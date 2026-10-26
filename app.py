import time
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="MTF EMA Screener Pro", page_icon="📈", layout="wide")

# ── Style ───────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px; padding: 10px 14px;}
.hero {background: linear-gradient(90deg,#0f766e,#1d4ed8); color:#fff; padding:16px 22px;
    border-radius:14px; margin-bottom:14px;}
.hero h2 {margin:0; font-size:1.6rem;} .hero span {opacity:.85; font-size:.9rem;}
.lvl {display:flex; justify-content:space-between; padding:7px 12px; margin:4px 0;
    border-radius:8px; font-weight:600; font-size:.95rem;}
.lvl small {font-weight:400; opacity:.85;}
.entry {background:rgba(59,130,246,.18); border-left:4px solid #3b82f6;}
.sl {background:rgba(239,68,68,.18); border-left:4px solid #ef4444;}
.tp {background:rgba(34,197,94,.16); border-left:4px solid #22c55e;}
.exp {background:rgba(245,158,11,.18); border-left:4px solid #f59e0b;}
</style>
""", unsafe_allow_html=True)

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

# Presets use the same per-timeframe rule structure:
# (Price vs EMA50, Price vs EMA200, EMA50 vs EMA200).
PRESETS = {
    "Custom": None,
    "Full stack (4H–1W)": {
        "15m": ("Any", "Any", "Any"),
        "1h":  ("Any", "Any", "Any"),
        "4h":  ("Above", "Above", "50>200"),
        "1d":  ("Above", "Above", "50>200"),
        "1w":  ("Above", "Above", "50>200"),
    },
    "Pullback in uptrend": {
        "15m": ("Any", "Any", "Any"),
        "1h":  ("Below", "Above", "50>200"),
        "4h":  ("Below", "Above", "50>200"),
        "1d":  ("Above", "Above", "50>200"),
        "1w":  ("Above", "Above", "50>200"),
    },
    "4H reclaim, 1D still below": {
        "15m": ("Any", "Any", "Any"),
        "1h":  ("Any", "Any", "Any"),
        "4h":  ("Above", "Above", "50>200"),
        "1d":  ("Any", "Below", "Any"),
        "1w":  ("Any", "Any", "Any"),
    },
    "Full bear stack": {
        "15m": ("Any", "Any", "Any"),
        "1h":  ("Any", "Any", "Any"),
        "4h":  ("Below", "Below", "50<200"),
        "1d":  ("Below", "Below", "50<200"),
        "1w":  ("Below", "Below", "50<200"),
    },
}
TF_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}

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

def binance_ohlc(sym, tf):
    k = fetch(BN, "/api/v3/klines", {"symbol": sym, "interval": tf, "limit": 500})
    return [[c[0], c[1], c[2], c[3], c[4]] for c in k] if isinstance(k, list) else None

def bybit_universe():
    d = fetch(BY, "/v5/market/tickers", {"category": "spot"})
    try:
        return [(x["symbol"], float(x["turnover24h"] or 0)) for x in d["result"]["list"]]
    except Exception:
        return []

def bybit_ohlc(sym, tf):
    d = fetch(BY, "/v5/market/kline",
              {"category": "spot", "symbol": sym, "interval": BY_TF[tf], "limit": 500})
    try:
        return [c[:5] for c in reversed(d["result"]["list"])]
    except Exception:
        return None

def okx_universe():
    d = fetch(OK, "/api/v5/market/tickers", {"instType": "SPOT"})
    try:
        return [(x["instId"].replace("-", ""), float(x["volCcy24h"] or 0))
                for x in d["data"] if x["instId"].endswith("-USDT")]
    except Exception:
        return []

def okx_ohlc(sym, tf):
    d = fetch(OK, "/api/v5/market/candles",
              {"instId": sym[:-4] + "-USDT", "bar": OK_TF[tf], "limit": 300})
    try:
        return [c[:5] for c in reversed(d["data"])]
    except Exception:
        return None

PROVIDERS = {
    "binance": (binance_universe, binance_ohlc),
    "bybit": (bybit_universe, bybit_ohlc),
    "okx": (okx_universe, okx_ohlc),
}

@st.cache_resource
def _store():
    return {}

def get_closes(store, name, sym, tf, ttl=90):
    key = (name, sym, tf)
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    raw = PROVIDERS[name][1](sym, tf)
    if not raw:
        return None
    closes = [float(c[4]) for c in raw]
    store[key] = (time.time(), closes)
    return closes

def get_candles(name, sym, tf):
    raw = PROVIDERS[name][1](sym, tf)
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close"])
    df["t"] = pd.to_datetime(df["t"].astype("int64"), unit="ms")
    return df.astype({c: float for c in ["open", "high", "low", "close"]}).reset_index(drop=True)

# ── Indicators ──────────────────────────────────────────────────
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

def scan_symbol(sym, closes_fn, rules, lf, ls, max_dist, live, fast):
    row, passed, act, fails = {"Symbol": sym}, 0, 0, 0
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
        fails += (not ok)
        row["Price"] = p
        if fast and fails >= 2:      # can no longer be MATCH or WATCH -> stop fetching
            break
    if act == 0:
        return None
    total = sum(1 for _, r in rules if not all(x == "Any" for x in r))
    row["Passed"] = f"{passed}/{total}"
    row["Status"] = "MATCH" if passed == total else ("WATCH" if total >= 2 and passed == total - 1 else "-")
    row["_score"] = passed / total
    return row

@st.cache_data(ttl=120, show_spinner=False)
def run_scan(order, rules, lf, ls, max_dist, live, top_n, min_vol, fast):
    bad = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT")
    store = _store()
    best = ("none", [])
    for name in order:
        uni_fn, _ = PROVIDERS[name]
        pairs = [(s, v) for s, v in uni_fn()
                 if s.endswith("USDT") and not s.endswith(bad) and v >= min_vol]
        pairs.sort(key=lambda x: x[1], reverse=True)
        syms = [s for s, _ in pairs[:top_n]]
        if not syms:
            continue
        fn = partial(get_closes, store, name)
        with ThreadPoolExecutor(max_workers=24) as ex:
            rows = [r for r in ex.map(
                lambda s: scan_symbol(s, fn, rules, lf, ls, max_dist, live, fast), syms) if r]
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

# ── Trade plan + Monte Carlo ────────────────────────────────────
def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def trade_plan(df, direction, sl_mode, atr_mult, rr):
    entry = df["close"].iloc[-1]
    a = atr(df).iloc[-1]
    long = direction == "Long"
    if sl_mode == "Swing (10 bars)":
        sl = df["low"].iloc[-10:].min() - 0.1 * a if long else df["high"].iloc[-10:].max() + 0.1 * a
    else:
        sl = entry - atr_mult * a if long else entry + atr_mult * a
    risk = abs(entry - sl)
    sign = 1 if long else -1
    tps = [entry + sign * risk * r for r in rr]
    return entry, sl, tps, risk

def monte_carlo(close, entry, sl, tps, rr, long, horizon, n_paths, drift, seed=42):
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(close))[-500:]
    if not drift:
        r = r - r.mean()
    paths = entry * np.exp(np.cumsum(r[rng.integers(0, len(r), (n_paths, horizon))], axis=1))

    def first(mask):
        hit = mask.any(axis=1)
        return np.where(hit, mask.argmax(axis=1), horizon + 1)

    t_sl = first(paths <= sl) if long else first(paths >= sl)
    t_tp = [first(paths >= tp) if long else first(paths <= tp) for tp in tps]
    term = paths[:, -1]
    sign = 1 if long else -1
    risk = abs(entry - sl)
    term_r = np.clip(sign * (term - entry) / risk, -1, None)
    # 1/3 of the position exits at each TP, SL fixed, leftovers marked to market at horizon end
    ev = 0.0
    for t, R in zip(t_tp, rr):
        out = np.where((t < t_sl) & (t <= horizon), R,
                       np.where((t_sl < t) & (t_sl <= horizon), -1.0, term_r))
        ev += out.mean() / len(rr)
    return {
        "paths": paths,
        "p_tp": [float(((t < t_sl) & (t <= horizon)).mean()) for t in t_tp],
        "p_sl": float(((t_sl < np.minimum.reduce(t_tp)) & (t_sl <= horizon)).mean()),
        "exp_price": float(term.mean()), "median": float(np.median(term)),
        "p5": float(np.percentile(term, 5)), "p95": float(np.percentile(term, 95)),
        "ev_r": float(ev),
    }

def fmt(x):
    return f"{x:,.6f}" if x < 1 else (f"{x:,.4f}" if x < 100 else f"{x:,.2f}")

def build_chart(df, lf, ls, entry, sl, tps, mc, tf, long, bars=160):
    c = df["close"].tolist()
    pad = lambda s: [None] * (len(c) - len(s)) + s
    d = df.copy()
    d["e50"], d["e200"] = pad(ema_series(c, lf)), pad(ema_series(c, ls))
    d = d.iloc[-bars:]
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    H = mc["paths"].shape[1]
    ft = [d["t"].iloc[-1] + step * i for i in range(H + 1)]
    pct = np.percentile(mc["paths"], [5, 25, 50, 75, 95], axis=0)
    pct = np.hstack([np.full((5, 1), entry), pct])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d["t"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
    fig.add_trace(go.Scatter(x=d["t"], y=d["e50"], name=f"EMA{lf}", line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Scatter(x=d["t"], y=d["e200"], name=f"EMA{ls}", line=dict(color="#a855f7", width=1.5)))
    # Monte Carlo fan
    fig.add_trace(go.Scatter(x=ft, y=pct[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[0], fill="tonexty", fillcolor="rgba(96,165,250,.14)",
                             line=dict(width=0), name="MC 5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[1], fill="tonexty", fillcolor="rgba(96,165,250,.28)",
                             line=dict(width=0), name="MC 25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[2], name="MC median", line=dict(color="#60a5fa", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=[ft[0], ft[-1]], y=[entry, mc["exp_price"]], name="Expected price",
                             line=dict(color="#f59e0b", width=3)))
    # levels
    x0, x1 = d["t"].iloc[0], ft[-1]
    fig.add_hrect(y0=min(entry, sl), y1=max(entry, sl), fillcolor="rgba(239,68,68,.10)", line_width=0)
    fig.add_hrect(y0=min(entry, tps[-1]), y1=max(entry, tps[-1]), fillcolor="rgba(34,197,94,.07)", line_width=0)
    levels = [("ENTRY", entry, "#3b82f6"), ("SL", sl, "#ef4444")] + \
             [(f"TP{i+1}", tp, "#22c55e") for i, tp in enumerate(tps)] + [("EXP", mc["exp_price"], "#f59e0b")]
    for name, y, col in levels:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=col, width=1.4,
                      dash="solid" if name in ("ENTRY", "SL") else "dash"))
        fig.add_annotation(x=x1, y=y, text=f"<b>{name}</b> {fmt(y)}", showarrow=False, xanchor="left",
                           font=dict(color=col, size=11))
    fig.update_layout(height=640, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=130, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04), title=f"{'LONG' if long else 'SHORT'} plan")
    return fig

# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
live = st.sidebar.toggle("Use live price (off = last closed candle)", True)
lf = st.sidebar.number_input("EMA fast", 1, 500, 50)
ls = st.sidebar.number_input("EMA slow", 1, 500, 200)
max_dist = st.sidebar.number_input("Max distance from EMA (%), 0 = off", 0.0, 100.0, 0.0, 0.5)
top_n = st.sidebar.slider("Top N coins by volume", 20, 1000, 400, 20)
min_vol = st.sidebar.number_input("Min 24h volume (USDT)", 0, 10**10, 1_000_000, 500_000)
fast = st.sidebar.toggle("Fast mode (stop after 2 failed TFs)", True,
                         help="Much fewer API calls; MATCH/WATCH results are identical.")
order = st.sidebar.multiselect("Data sources (priority order)", ["okx", "bybit", "binance"],
                               default=["okx", "bybit", "binance"])
refresh = st.sidebar.slider("Auto-refresh (minutes, 0 = off)", 0, 30, 0)
alerts = st.sidebar.toggle("Telegram alerts (needs secrets)", False)

st.sidebar.subheader("EMA criteria presets")
preset_name = st.sidebar.selectbox(
    "Preset",
    list(PRESETS.keys()),
    index=0,
    help="Choose a preset from the reference image, or Custom to set each timeframe manually."
)

st.sidebar.subheader("Rules per timeframe")
manual_rules = []
for tf in TFS:
    d = DEFAULT_RULES[tf]
    st.sidebar.markdown(f"**{tf}**")
    c1, c2, c3 = st.sidebar.columns(3)
    a = c1.selectbox("Price / EMA50", PRICE_OPTS, PRICE_OPTS.index(d[0]), key=f"a{tf}")
    b = c2.selectbox("Price / EMA200", PRICE_OPTS, PRICE_OPTS.index(d[1]), key=f"b{tf}")
    c = c3.selectbox("EMA50 / EMA200", CROSS_OPTS, CROSS_OPTS.index(d[2]), key=f"c{tf}")
    manual_rules.append((tf, (a, b, c)))

# A selected preset takes precedence; Custom uses the manual controls above.
if PRESETS[preset_name] is None:
    rules = manual_rules
else:
    rules = [(tf, PRESETS[preset_name][tf]) for tf in TFS]
    st.sidebar.caption(f"Active preset: {preset_name}")

if refresh:
    st_autorefresh(interval=refresh * 60_000, key="auto")

# ── Main ────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h2>📈 MTF EMA Screener Pro</h2>'
            '<span>Multi-timeframe EMA scan · trade planner · Monte Carlo</span></div>', unsafe_allow_html=True)
if st.button("🔄 Scan now"):
    run_scan.clear()

if not order:
    st.warning("Select at least one data source.")
    st.stop()

t0 = time.time()
with st.spinner(f"Scanning up to {top_n} coins..."):
    provider, rows = run_scan(tuple(order), tuple(rules), lf, ls, max_dist, live, top_n, min_vol, fast)

if not rows:
    st.error("All data sources failed. Try another source order or retry in a minute.")
    st.stop()

df = pd.DataFrame(rows).sort_values(["_score", "Symbol"], ascending=[False, True]).drop(columns="_score")
match = df[df["Status"] == "MATCH"]
watch = df[df["Status"] == "WATCH"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scanned", len(df))
m2.metric("🟢 MATCH", len(match))
m3.metric("👀 WATCH", len(watch))
m4.metric("Source", provider.upper())
m5.metric("Scan time", f"{time.time() - t0:.1f}s")
st.caption(f"Updated {time.strftime('%H:%M:%S')} (UTC)")

now = set(match["Symbol"])
prev = st.session_state.get("prev_matches")
if alerts and prev is not None:
    for s in now - prev:
        telegram(f"ENTRY: {s} - all MTF EMA rules matched")
    for s in prev - now:
        telegram(f"EXIT: {s} - MTF EMA rules no longer matched")
st.session_state["prev_matches"] = now

pct_cols = {c: st.column_config.NumberColumn(c, format="%.2f%%") for c in df.columns if "%" in c}
cfg = {"Price": st.column_config.NumberColumn("Price", format="%.6g"), **pct_cols}
t1, t2, t3, t4 = st.tabs(["✅ MATCH", "👀 WATCH", "All", "🎯 Trade planner"])
for tab, data in ((t1, match), (t2, watch), (t3, df)):
    with tab:
        if data.empty:
            st.info("Nothing here right now.")
        else:
            st.dataframe(data, use_container_width=True, hide_index=True, column_config=cfg)
            st.download_button("⬇️ CSV", data.to_csv(index=False), "screener.csv", key=f"dl{id(tab)}")

# ── Trade planner ───────────────────────────────────────────────
with t4:
    pool = list(match["Symbol"]) + list(watch["Symbol"])
    if not pool:
        st.info("No MATCH/WATCH coins to plan right now.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        sym = k1.selectbox("Coin", pool, format_func=lambda s: f"{'🟢' if s in now else '👀'} {s}")
        ctf = k2.selectbox("Chart / entry TF", TFS, index=2)
        dirn = k3.selectbox("Direction", ["Auto", "Long", "Short"])
        sl_mode = k4.selectbox("Stop-loss", ["ATR", "Swing (10 bars)"])
        k5, k6, k7, k8 = st.columns(4)
        atr_mult = k5.number_input("ATR multiple", 0.5, 10.0, 1.5, 0.25)
        rr_txt = k6.text_input("TP in R (TP1,TP2,TP3)", "1,2,3")
        horizon = k7.slider("MC horizon (bars)", 10, 200, 60, 5)
        n_paths = k8.select_slider("MC paths", [1000, 2000, 5000, 10000], 5000)
        k9, k10, k11 = st.columns(3)
        drift = k9.toggle("Include historical drift", True)
        equity = k10.number_input("Account size (USDT)", 10.0, 1e9, 1000.0, 100.0)
        risk_pct = k11.number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1)

        try:
            rr = [float(x) for x in rr_txt.split(",")][:3]
            assert len(rr) == 3 and all(x > 0 for x in rr)
        except Exception:
            st.error("Enter three positive R multiples, e.g. 1,2,3")
            st.stop()

        cdf = get_candles(provider, sym, ctf)
        if cdf is None or len(cdf) < ls + 5:
            st.error("Could not load candles for this coin/timeframe.")
        else:
            closes = cdf["close"].tolist()
            e50, e200 = ema_series(closes, lf)[-1], ema_series(closes, ls)[-1]
            p = closes[-1]
            auto = "Long" if (p > e200 and e50 >= e200) else "Short" if (p < e200 and e50 <= e200) else \
                   ("Long" if p > e200 else "Short")
            direction = auto if dirn == "Auto" else dirn
            long = direction == "Long"
            entry, sl, tps, risk = trade_plan(cdf, direction, sl_mode, atr_mult, rr)
            mc = monte_carlo(cdf["close"].values, entry, sl, tps, rr, long, horizon, n_paths, drift)
            qty = (equity * risk_pct / 100) / risk

            left, right = st.columns([3, 1.1])
            left.plotly_chart(build_chart(cdf, lf, ls, entry, sl, tps, mc, ctf, long),
                              use_container_width=True)
            with right:
                st.markdown(f"### {sym} · {'🟢 LONG' if long else '🔴 SHORT'}")
                st.markdown(f'<div class="lvl entry">ENTRY<span>{fmt(entry)}</span></div>'
                            f'<div class="lvl sl">SL<span>{fmt(sl)} <small>({abs(sl/entry-1)*100:.2f}%)</small></span></div>',
                            unsafe_allow_html=True)
                for i, (tp, R, pr) in enumerate(zip(tps, rr, mc["p_tp"])):
                    st.markdown(f'<div class="lvl tp">TP{i+1} · {R:g}R<span>{fmt(tp)} '
                                f'<small>P={pr*100:.0f}%</small></span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="lvl exp">Expected price<span>{fmt(mc["exp_price"])} '
                            f'<small>({(mc["exp_price"]/entry-1)*100:+.2f}%)</small></span></div>',
                            unsafe_allow_html=True)
                st.metric("P(SL before any TP)", f"{mc['p_sl']*100:.0f}%")
                st.metric("Expected value (R)", f"{mc['ev_r']:+.2f}R",
                          help="1/3 of the position exits at each TP; leftovers marked at horizon end.")
                st.metric("Median / 5–95% range",
                          f"{fmt(mc['median'])}", f"{fmt(mc['p5'])} – {fmt(mc['p95'])}", delta_color="off")
                st.metric("Position size", f"{qty:,.4f} {sym[:-4]}",
                          f"≈ {qty * entry:,.0f} USDT · risk {equity * risk_pct / 100:,.2f}", delta_color="off")
            st.caption("Monte Carlo = bootstrap of the last ≤500 log-returns on the chosen timeframe; "
                       "hits are checked on closes. A statistical estimate, not a prediction or financial advice.")
