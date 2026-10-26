import time
import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="MTF EMA Screener", page_icon="📈", layout="wide")

TFS = ["15m", "1h", "4h", "1d", "1w"]
PRICE_OPTS = ["Any", "Above", "Below"]
CROSS_OPTS = ["Any", "50>200", "50<200"]
DEFAULT_RULES = {
    "15m": ("Any", "Any", "Any"),
    "1h":  ("Any", "Below", "Any"),
    "4h":  ("Any", "Any", "Any"),
    "1d":  ("Above", "Any", "Any"),
    "1w":  ("Above", "Any", "Any"),
}

# ── HTTP ────────────────────────────────────────────────────────
def fetch(bases, path, params=None, retries=2):
    for _ in range(retries):
        for b in bases:
            try:
                r = requests.get(b + path, params=params, timeout=15,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.ok:
                    return r.json()
            except Exception:
                pass
        time.sleep(0.4)
    return None

# ── Providers ───────────────────────────────────────────────────
BN = ["https://data-api.binance.vision", "https://api.binance.com"]
BY = ["https://api.bybit.com", "https://api.bytick.com"]
OK = ["https://www.okx.com", "https://aws.okx.com"]
BY_TF = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W"}
OK_TF = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"}

def binance_universe():
    t = fetch(BN, "/api/v3/ticker/24hr")
    return [(x["symbol"], float(x["quoteVolume"])) for x in t] if isinstance(t, list) else []

def binance_closes(sym, tf):
    k = fetch(BN, "/api/v3/klines", {"symbol": sym, "interval": tf, "limit": 500})
    return [float(c[4]) for c in k] if isinstance(k, list) else None

def bybit_universe():
    d = fetch(BY, "/v5/market/tickers", {"category": "spot"})
    try:
        return [(x["symbol"], float(x["turnover24h"])) for x in d["result"]["list"]]
    except Exception:
        return []

def bybit_closes(sym, tf):
    d = fetch(BY, "/v5/market/kline",
              {"category": "spot", "symbol": sym, "interval": BY_TF[tf], "limit": 500})
    try:
        return [float(c[4]) for c in reversed(d["result"]["list"])]
    except Exception:
        return None

def okx_universe():
    d = fetch(OK, "/api/v5/market/tickers", {"instType": "SPOT"})
    try:
        return [(x["instId"].replace("-", ""), float(x["volCcy24h"]))
                for x in d["data"] if x["instId"].endswith("-USDT")]
    except Exception:
        return []

def okx_closes(sym, tf):
    d = fetch(OK, "/api/v5/market/candles",
              {"instId": sym[:-4] + "-USDT", "bar": OK_TF[tf], "limit": 300})
    try:
        return [float(c[4]) for c in reversed(d["data"])]
    except Exception:
        return None

PROVIDERS = {
    "binance": (binance_universe, binance_closes),
    "bybit": (bybit_universe, bybit_closes),
    "okx": (okx_universe, okx_closes),
}

# ── Indicator logic ─────────────────────────────────────────────
def ema_series(vals, n):
    if len(vals) < n:
        return []
    k = 2 / (n + 1)
    e = sum(vals[:n]) / n
    out = [e]
    for v in vals[n:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def evaluate(rule, p, e50, e200, max_dist):
    m50, m200, mx = rule
    ok50 = m50 == "Any" or (p > e50 if m50 == "Above" else p < e50)
    ok200 = m200 == "Any" or (p > e200 if m200 == "Above" else p < e200)
    okx = mx == "Any" or (e50 > e200 if mx == "50>200" else e50 < e200)
    d50, d200 = (p / e50 - 1) * 100, (p / e200 - 1) * 100
    dok50 = max_dist == 0 or m50 == "Any" or abs(d50) <= max_dist
    dok200 = max_dist == 0 or m200 == "Any" or abs(d200) <= max_dist
    return ok50 and ok200 and okx and dok50 and dok200, d50, d200

def scan_symbol(sym, closes_fn, rules, lf, ls, max_dist, live):
    row, passed, act = {"Symbol": sym}, 0, 0
    for tf, rule in rules:
        if all(x == "Any" for x in rule):
            continue
        closes = closes_fn(sym, tf)
        if not closes or len(closes) < ls + 2:
            return None
        idx = -1 if live else -2
        p = closes[idx]
        e50, e200 = ema_series(closes, lf)[idx], ema_series(closes, ls)[idx]
        ok, d50, d200 = evaluate(rule, p, e50, e200, max_dist)
        row[tf] = "✅" if ok else "❌"
        row[f"{tf} %50"] = round(d50, 2)
        row[f"{tf} %200"] = round(d200, 2)
        act += 1
        passed += ok
    if act == 0:
        return None
    row["Passed"] = f"{passed}/{act}"
    row["Status"] = "MATCH" if passed == act else ("WATCH" if act >= 2 and passed == act - 1 else "-")
    row["_score"] = passed / act
    return row

@st.cache_data(ttl=120, show_spinner=False)
def run_scan(order, rules, lf, ls, max_dist, live, top_n, min_vol):
    bad = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
    best = ("none", [])
    for name in order:
        uni_fn, closes_fn = PROVIDERS[name]
        pairs = [(s, v) for s, v in uni_fn()
                 if s.endswith("USDT") and not s.endswith(bad) and v >= min_vol]
        pairs.sort(key=lambda x: x[1], reverse=True)
        syms = [s for s, _ in pairs[:top_n]]
        if not syms:
            continue
        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = [r for r in ex.map(
                lambda s: scan_symbol(s, closes_fn, rules, lf, ls, max_dist, live), syms) if r]
        if len(rows) > len(best[1]):
            best = (name, rows)
        if len(rows) >= 0.5 * len(syms):
            return name, rows
    return best

def telegram(msg):
    try:
        tok, chat = st.secrets["TG_TOKEN"], st.secrets["TG_CHAT"]
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": msg}, timeout=10)
    except Exception:
        pass

# ── Sidebar settings ────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
live = st.sidebar.toggle("Use live price (off = last closed candle)", True)
lf = st.sidebar.number_input("EMA fast", 1, 500, 50)
ls = st.sidebar.number_input("EMA slow", 1, 500, 200)
max_dist = st.sidebar.number_input("Max distance from EMA (%), 0 = off", 0.0, 100.0, 0.0, 0.5)
top_n = st.sidebar.slider("Top N coins by volume", 20, 300, 100, 10)
min_vol = st.sidebar.number_input("Min 24h volume (USDT)", 0, 10**10, 5_000_000, 1_000_000)
order = st.sidebar.multiselect("Data sources (priority order)", ["okx", "bybit", "binance"],
                               default=["okx", "bybit", "binance"])
refresh = st.sidebar.slider("Auto-refresh (minutes, 0 = off)", 0, 30, 0)
alerts = st.sidebar.toggle("Telegram alerts (needs secrets)", False)

st.sidebar.subheader("Rules per timeframe")
rules = []
for tf in TFS:
    d = DEFAULT_RULES[tf]
    st.sidebar.markdown(f"**{tf}**")
    c1, c2, c3 = st.sidebar.columns(3)
    a = c1.selectbox("P/50", PRICE_OPTS, PRICE_OPTS.index(d[0]), key=f"a{tf}")
    b = c2.selectbox("P/200", PRICE_OPTS, PRICE_OPTS.index(d[1]), key=f"b{tf}")
    c = c3.selectbox("50/200", CROSS_OPTS, CROSS_OPTS.index(d[2]), key=f"c{tf}")
    rules.append((tf, (a, b, c)))

if refresh:
    st_autorefresh(interval=refresh * 60_000, key="auto")

# ── Main ────────────────────────────────────────────────────────
st.title("📈 MTF EMA Screener")
if st.button("🔄 Scan now"):
    run_scan.clear()

if not order:
    st.warning("Select at least one data source.")
    st.stop()

with st.spinner("Scanning..."):
    provider, rows = run_scan(tuple(order), tuple(rules), lf, ls, max_dist,
                              live, top_n, min_vol)

if not rows:
    st.error("All data sources failed. Try another source order or retry in a minute.")
    st.stop()

df = pd.DataFrame(rows).sort_values(["_score", "Symbol"], ascending=[False, True]).drop(columns="_score")
match = df[df["Status"] == "MATCH"]
watch = df[df["Status"] == "WATCH"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scanned", len(df))
c2.metric("MATCH", len(match))
c3.metric("WATCH", len(watch))
c4.metric("Source", provider)
st.caption(f"Updated {time.strftime('%H:%M:%S')} (UTC)")

# Telegram on new matches
now = set(match["Symbol"])
prev = st.session_state.get("prev_matches")
if alerts and prev is not None:
    for s in now - prev:
        telegram(f"ENTRY: {s} - all MTF EMA rules matched")
    for s in prev - now:
        telegram(f"EXIT: {s} - MTF EMA rules no longer matched")
st.session_state["prev_matches"] = now

t1, t2, t3 = st.tabs(["✅ MATCH", "👀 WATCH", "All"])
for tab, data in ((t1, match), (t2, watch), (t3, df)):
    with tab:
        if data.empty:
            st.info("Nothing here right now.")
        else:
            st.dataframe(data, use_container_width=True, hide_index=True)
