"""
시장환경 신호등 + ETF 신호등
- 매일 장 마감 후 시장 전체를 🟢 공격 / 🟡 선별 / 🔴 방어 로 판정
  (QQQ·SPY 추세, 분산일, 50일선 위 종목 비율, 신고가-신저가, VIX)
- etfs.txt의 ETF마다 🟢 강세 / 🟡 중립 / 🔴 약세 판정
  (단기 추세, 장기 추세, 분산일, SPY 대비 상대강도 순위, 52주 고점 거리)
- 신호가 바뀐 ETF는 따로 모아서 알려줍니다.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 ─────────────
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                       # 휴장일·주말에도 가장 최근 거래일 기준으로 실행

DIST_WINDOW = int(os.environ.get("DIST_WINDOW", "25"))            # 분산일을 세는 기간(거래일)
DIST_DROP = float(os.environ.get("DIST_DROP", "0.2"))             # 분산일 하락 기준(%)
DIST_EXPIRE = float(os.environ.get("DIST_EXPIRE", "5"))           # 분산일 이후 종가가 이만큼(%) 오르면 그 분산일은 지움 (IBD 방식)

BREADTH_GOOD = float(os.environ.get("BREADTH_GOOD", "60"))        # 50일선 위 종목 비율(%) 이 이상이면 +1
BREADTH_BAD = float(os.environ.get("BREADTH_BAD", "40"))          # 이 미만이면 -1
VIX_GOOD = float(os.environ.get("VIX_GOOD", "20"))                # VIX 이 미만이면서 5일 전보다 낮으면 +1
VIX_BAD = float(os.environ.get("VIX_BAD", "25"))                  # 이 이상이면 -1
VIX_SPIKE = float(os.environ.get("VIX_SPIKE", "20"))              # 하루에 이만큼(%) 급등해도 -1

FTD_PCT = float(os.environ.get("FTD_PCT", "1.5"))                 # 팔로우스루 데이 상승률 기준(%)
FTD_MIN_DD = float(os.environ.get("FTD_MIN_DD", "7"))             # 52주 고점 대비 이만큼(%) 이상 조정받은 뒤에만 팔로우스루를 봄

RS_DAYS = int(os.environ.get("RS_DAYS", "63"))                    # ETF 상대강도 기간 (63거래일 ≈ 3개월)
RS_TOP = float(os.environ.get("RS_TOP", "30"))                    # 상위 30% +1, 하위 30% -1
HIGH_NEAR = float(os.environ.get("HIGH_NEAR", "10"))              # 52주 고점 10% 이내 +1
HIGH_FAR = float(os.environ.get("HIGH_FAR", "25"))                # 25% 넘게 아래면 -1

# 레버리지·인버스: 52주 고점 거리 점수 제외 / 채권·원자재: 분산일 점수 제외
LEVERAGED = set(os.environ.get("LEVERAGED", "TQQQ,SQQQ,SOXL,SOXS,KORU").split(","))
NO_DIST = set(os.environ.get("NO_DIST", "TLT,IEF,LQD,HYG,GLD,DBC").split(","))
# ───────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (market-light)"}
LIGHT = {1: "🟢", 0: "🟡", -1: "🔴"}
MARKET_NAME = {1: "공격", 0: "선별", -1: "방어"}
MARKET_ADVICE = {1: "돌파 매수 적극, 정상 비중",
                 0: "가장 강한 셋업만, 비중 축소, 손절 짧게",
                 -1: "신규 롱 자제, 현금 확대 · 숏 스캐너가 유리한 구간"}
ETF_NAME = {1: "강세", 0: "중립", -1: "약세"}


# ───────────── 목록 ─────────────
def load_etfs() -> dict[str, str]:
    out = {}
    for ln in (ROOT / "etfs.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            t, _, desc = ln.partition(",")
            out[t.strip().upper()] = desc.strip()
    return out


def load_universe() -> list[str]:
    """시장 폭(50일선 위 비율, 신고가·신저가) 계산용: S&P 500 + watchlist.txt"""
    cache = STATE / "universe_sp500.csv"
    if not (cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400):
        try:
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            if len(t) < 400:
                raise ValueError("종목 수가 비정상적으로 적음")
            t[["Symbol"]].rename(columns={"Symbol": "ticker"}).to_csv(cache, index=False)
        except Exception as e:                                       # 실패하면 지난 목록 사용
            print(f"[경고] S&P 500 목록 갱신 실패: {e}")
    tickers = []
    if cache.exists():
        tickers += pd.read_csv(cache)["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    p = ROOT / "watchlist.txt"
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tickers.append(re.sub(r"\.([A-Z])$", r"-\1", ln.partition(",")[0].strip().upper()))
    return [t for t in dict.fromkeys(tickers) if re.fullmatch(r"[A-Z\-]{1,6}", t)]


# ───────────── 시세 ─────────────
def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                df = data[t].dropna(subset=["Close"])
                if len(df):
                    out[t] = df
        time.sleep(1)
    return out


# ───────────── 지표 ─────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def trend(df: pd.DataFrame) -> tuple[int, str]:
    """21일 EMA·50일선 기준 단기 추세"""
    c = df["Close"]
    last, e21, s50 = c.iloc[-1], ema(c, 21).iloc[-1], c.rolling(50).mean().iloc[-1]
    if np.isnan(s50):
        return 0, "데이터 부족"
    a = "위" if last > e21 else "아래"
    b = "위" if last > s50 else "아래"
    score = -1 if last < s50 else (1 if last > e21 else 0)
    return score, f"21EMA {a} / 50SMA {b}"


def long_trend(df: pd.DataFrame) -> int:
    """50일선 > 200일선 정배열 + 200일선 상승(20일 전 대비)"""
    c = df["Close"]
    if len(c) < 220:
        return 0
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    up_align, rising = s50.iloc[-1] > s200.iloc[-1], s200.iloc[-1] > s200.iloc[-21]
    return 1 if up_align and rising else (-1 if not up_align and not rising else 0)


def distribution_days(df: pd.DataFrame) -> int:
    """최근 N거래일 중: 종가 0.2% 이상 하락 + 거래량 전날보다 많음.
    그 뒤 종가가 그날 종가보다 5% 넘게 오르면 지워짐 (IBD 방식)"""
    c, v = df["Close"], df["Volume"]
    if len(c) < DIST_WINDOW + 2 or v.tail(DIST_WINDOW).sum() == 0:
        return 0
    chg = c.pct_change() * 100
    is_dist = (chg <= -DIST_DROP) & (v > v.shift(1))
    n, cv = 0, c.values
    for i in range(len(c) - DIST_WINDOW, len(c)):
        if is_dist.iloc[i]:
            after = cv[i + 1:]
            if not (len(after) and after.max() >= cv[i] * (1 + DIST_EXPIRE / 100)):
                n += 1
    return n


def dist_score(n: int) -> int:
    return 1 if n <= 3 else (0 if n <= 5 else -1)


def follow_through(df: pd.DataFrame) -> dict | None:
    """조정 중이라면 반등 시도 몇 일째인지, 오늘이 팔로우스루 데이인지"""
    c, v = df["Close"], df["Volume"]
    if len(c) < 260:
        return None
    win = c.tail(25)
    low_pos = len(c) - 25 + int(np.argmin(win.values))               # 최근 25일 최저 종가 = 반등 시도 1일째
    hi = c.iloc[max(0, low_pos - 251):low_pos + 1].max()
    dd = (c.iloc[low_pos] / hi - 1) * 100
    if dd > -FTD_MIN_DD:
        return None                                                  # 충분히 조정받지 않음
    day = len(c) - low_pos                                           # 저점 당일 = 1일째
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    is_ftd = 4 <= day <= 20 and chg >= FTD_PCT and v.iloc[-1] > v.iloc[-2]
    return {"low_date": c.index[low_pos].date(), "day": day, "dd": dd, "chg": chg, "ftd": is_ftd}


def breadth(data: dict[str, pd.DataFrame], universe: list[str], session: dt.date) -> dict:
    above50 = above200 = n50 = n200 = 0
    nh = nl = nh5 = nl5 = 0
    for t in universe:
        df = data.get(t)
        if df is None or len(df) < 60 or df.index[-1].date() != session:
            continue
        c = df["Close"]
        n50 += 1
        above50 += c.iloc[-1] > c.rolling(50).mean().iloc[-1]
        if len(c) >= 200:
            n200 += 1
            above200 += c.iloc[-1] > c.rolling(200).mean().iloc[-1]
        if len(df) >= 257:
            hh = df["High"].rolling(252).max()
            ll = df["Low"].rolling(252).min()
            new_hi = (df["High"] >= hh).tail(5)
            new_lo = (df["Low"] <= ll).tail(5)
            nh += bool(new_hi.iloc[-1]); nl += bool(new_lo.iloc[-1])
            nh5 += int(new_hi.sum()); nl5 += int(new_lo.sum())
    return {"n": n50, "p50": above50 / n50 * 100 if n50 else np.nan,
            "p200": above200 / n200 * 100 if n200 else np.nan,
            "nh": nh, "nl": nl, "nh5": nh5, "nl5": nl5}


def light_of(score: int) -> int:
    return 1 if score >= 3 else (-1 if score <= -3 else 0)


# ───────────── 시장 판정 ─────────────
def market_eval(data, universe, session) -> dict:
    rows, total = [], 0
    for t in ("QQQ", "SPY"):
        s, label = trend(data[t])
        rows.append((t, label, s)); total += s
    dq, ds = distribution_days(data["QQQ"]), distribution_days(data["SPY"])
    s = dist_score(max(dq, ds))
    rows.append(("분산일", f"QQQ {dq}일 · SPY {ds}일", s)); total += s

    b = breadth(data, universe, session)
    s = 0 if np.isnan(b["p50"]) else (1 if b["p50"] >= BREADTH_GOOD else (-1 if b["p50"] < BREADTH_BAD else 0))
    rows.append(("50일선↑", f"{b['p50']:.0f}%  ({b['n']}종목 중)", s)); total += s
    rows.append(("200일선↑", f"{b['p200']:.0f}%", None))
    s = 1 if b["nh5"] >= max(5, 2 * b["nl5"]) else (-1 if b["nl5"] > b["nh5"] else 0)
    rows.append(("신고/신저", f"{b['nh']} / {b['nl']}  (5일 {b['nh5']} / {b['nl5']})", s)); total += s

    vix = data.get("^VIX")
    if vix is not None and len(vix) > 6:
        vc = vix["Close"]
        v, v1, v5 = vc.iloc[-1], vc.iloc[-2], vc.iloc[-6]
        jump = (v / v1 - 1) * 100
        s = -1 if v >= VIX_BAD or jump >= VIX_SPIKE else (1 if v < VIX_GOOD and v < v5 else 0)
        rows.append(("VIX", f"{v:.1f}  ({v - v1:+.1f})", s)); total += s
    else:
        rows.append(("VIX", "데이터 없음", 0))

    ftd = {t: follow_through(data[t]) for t in ("QQQ", "SPY")}
    return {"score": total, "light": light_of(total), "rows": rows, "ftd": ftd}


# ───────────── ETF 판정 ─────────────
def etf_eval(data, etfs, session) -> tuple[list[dict], list[str]]:
    res, missing = [], []
    for t, desc in etfs.items():
        df = data.get(t)
        if df is None or len(df) < 60 or df.index[-1].date() != session:
            missing.append(t)
            continue
        c = df["Close"]
        p1, _ = trend(df)
        p2 = long_trend(df)
        nd = distribution_days(df)
        p3 = 0 if t in NO_DIST else dist_score(nd)
        hi_dist = (c.iloc[-1] / c.tail(252).max() - 1) * 100
        p5 = 0 if t in LEVERAGED else (1 if hi_dist >= -HIGH_NEAR else (-1 if hi_dist < -HIGH_FAR else 0))
        ret = (c.iloc[-1] / c.iloc[-RS_DAYS - 1] - 1) * 100 if len(c) > RS_DAYS else np.nan
        res.append({"t": t, "desc": desc, "parts": [p1, p2, p3, 0, p5], "dist": nd,
                    "hi": hi_dist, "ret": ret, "chg": (c.iloc[-1] / c.iloc[-2] - 1) * 100})
    ranked = sorted([r for r in res if not np.isnan(r["ret"])], key=lambda r: -r["ret"])
    n = len(ranked)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        pct = (i + 1) / n * 100
        r["parts"][3] = 1 if pct <= RS_TOP else (-1 if pct > 100 - RS_TOP else 0)
    for r in res:
        r.setdefault("rank", None)
        r["score"] = sum(r["parts"])
        r["light"] = light_of(r["score"])
    return res, missing


# ───────────── 디스코드 ─────────────
def post(content: str):
    if not WEBHOOK:
        print(content)
        print("-" * 40)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def fit(s: str, w: int, right: bool = False) -> str:
    out = ""
    for ch in s:
        if width(out + ch) > w:
            break
        out += ch
    pad = " " * (w - width(out))
    return pad + out if right else out + pad


def send_block(title: str, head: str | None, lines: list[str]):
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first and title else ""
        h = f"{head}\n" if head else ""
        post(f"{header}```\n{h}" + "\n".join(block) + "\n```")
        first = False


def etf_line(r: dict) -> str:
    rank = f"{r['rank']}" if r["rank"] else "-"
    mark = "*" if r["t"] in LEVERAGED or r["t"] in NO_DIST else " "
    sc = f"{r['score']:+d}" if r["score"] else "0"
    return (fit(r["t"] + mark, 7) + fit(r["desc"], 16) + fit(sc, 4, True) + fit(rank, 5, True)
            + f"{r['ret']:>+7.1f}{r['hi']:>7.1f}{r['dist']:>5d}")


ETF_HEAD = (fit("TICKER", 7) + fit("설명", 16) + fit("점수", 4, True) + fit("RS", 5, True)
            + fit("3M%", 7, True) + fit("고점%", 7, True) + fit("분산", 5, True))


# ───────────── 실행 ─────────────
def main():
    etfs = load_etfs()
    universe = load_universe()
    need = list(dict.fromkeys(["QQQ", "SPY", "^VIX"] + list(etfs) + universe))
    data = download(need)
    if "QQQ" not in data or "SPY" not in data:
        post("⚠️ 시장 신호등: QQQ·SPY 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return 1
    session = data["SPY"].index[-1].date()
    today = dt.datetime.now(NY).date()
    if session != today and not FORCE:
        print(f"오늘({today})은 휴장일로 보임 (최근 거래일 {session}). 종료.")
        return

    m = market_eval(data, universe, session)
    etf_res, missing = etf_eval(data, etfs, session)

    # 어제 신호와 비교 (같은 날 다시 실행해도 비교 대상은 그대로 유지)
    sf = STATE / "lights.json"
    st = json.loads(sf.read_text()) if sf.exists() else {}
    prev = st.get("prev", {}) if st.get("date") == str(session) else st.get("cur", {})
    cur = {"_market": m["light"], **{r["t"]: r["light"] for r in etf_res}}
    sf.write_text(json.dumps({"date": str(session), "cur": cur, "prev": prev}))

    # ① 시장
    pm = prev.get("_market")
    change = ""
    if pm is not None and pm != m["light"]:
        change = f"   ← 어제 {LIGHT[pm]}에서 {'상승' if m['light'] > pm else '하락'}"
    lines = []
    for label, desc, s in m["rows"]:
        sc = "참고" if s is None else f"{s:+d}" if s else "0"
        lines.append(fit(label, 10) + fit(desc, 30) + fit(sc, 4, True))
    for t, f in m["ftd"].items():
        if not f:
            continue
        if f["ftd"] and m["light"] < 1:
            lines.append(f"✅ {t} 오늘 팔로우스루 데이 ({f['chg']:+.1f}%, 거래량 증가) → 시험 매수 구간")
        elif f["day"] > 1:
            lines.append(f"{t} 조정 {f['dd']:.0f}% 후 반등 시도 {f['day']}일째 (저점 {f['low_date']:%m/%d})")
    lines.append(f"→ {MARKET_ADVICE[m['light']]}")
    post(f"**🚦 시장환경 | {session} 마감**\n"
         f"## {LIGHT[m['light']]} {MARKET_NAME[m['light']]} ({m['score']:+d}점){change}")
    send_block("", None, lines)

    # ② 신호 바뀐 ETF
    changed = [r for r in etf_res if r["t"] in prev and prev[r["t"]] != r["light"]]
    if changed:
        changed.sort(key=lambda r: (-(r["light"] - prev[r["t"]]), -r["score"]))
        send_block(f"🔄 신호 바뀐 ETF ({len(changed)})", None,
                   [fit(r["t"], 7) + fit(r["desc"], 16) + f"{LIGHT[prev[r['t']]]}→{LIGHT[r['light']]}"
                    f"  {r['score']:+d}점" for r in changed])

    # ③ ETF 표 (신호별, 점수·RS 순)
    for lv in (1, 0, -1):
        rows = [r for r in etf_res if r["light"] == lv]
        if not rows:
            continue
        rows.sort(key=lambda r: (-r["score"], r["rank"] or 999))
        send_block(f"{LIGHT[lv]} {ETF_NAME[lv]} ({len(rows)})", ETF_HEAD, [etf_line(r) for r in rows])

    post("점수 = 단기추세 + 장기추세 + 분산일 + RS + 52주고점 (각 -1~+1) · +3 이상 🟢, -3 이하 🔴\n"
         f"RS = 3개월 수익률 순위 ({len(etf_res)}개 중) · 고점% = 52주 고점 대비 · 분산 = 최근 {DIST_WINDOW}일 분산일\n"
         "\\* 레버리지는 고점% 점수 제외, 채권·금·원자재는 분산일 점수 제외")
    if missing:
        post("⚠️ 시세를 받지 못한 ETF: " + ", ".join(missing))
    print(f"완료: 시장 {m['score']:+d}, ETF {len(etf_res)}개, 바뀐 신호 {len(changed)}개")


if __name__ == "__main__":
    sys.exit(main())
