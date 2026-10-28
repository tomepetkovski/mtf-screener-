import threading
import time
from functools import partial
from urllib.parse import quote

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

MARKETS = {"Crypto (USDT spot)": "crypto", "US stocks (NASDAQ + NYSE)": "stocks"}
STOCK_EXCH = ["NASDAQ", "NYSE"]
STOCK_WORKERS = 12      # Yahoo rate-limits hard; raise carefully
CRYPTO_WORKERS = 32
CLOSES_KEPT = 700       # enough warm-up for EMA200, keeps the cache small
ENRICH_MAX = 150        # extra signals are computed for at most this many MATCH/WATCH rows
ANALYST_MAX = 100       # Finnhub free tier = 60 calls/min; results are cached for 24h
ANALYST_MIN = {"Hold+": -0.4, "Buy+": 0.4, "Strong Buy": 1.2}

# ── HTTP ────────────────────────────────────────────────────────
def fetch(bases, path, params=None, retries=2, headers=None):
    hdr = headers or {"User-Agent": "Mozilla/5.0"}
    for _ in range(retries):
        for b in bases:
            try:
                r = requests.get(b + path, params=params, timeout=15, headers=hdr)
                if r.ok:
                    return r.json()
                if r.status_code == 404:      # unknown symbol: no point retrying
                    return None
                if r.status_code == 429:      # rate limited: back off
                    time.sleep(1.5)
            except Exception:
                pass
        time.sleep(0.4)
    return None

# ── Crypto providers ────────────────────────────────────────────
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
    return [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in k] if isinstance(k, list) else None

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
        return [c[:6] for c in reversed(d["result"]["list"])]
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
        return [c[:6] for c in reversed(d["data"])]
    except Exception:
        return None

# ── US stocks (NASDAQ + NYSE) ───────────────────────────────────
# Universe: nasdaq.com screener API (has price + volume -> ranked by dollar volume),
# falling back to the official NASDAQ Trader symbol directory (no volume).
# Prices: Yahoo Finance chart API. Yahoo has no 4h bars, so 4h is built from 60m bars.
NQ = ["https://api.nasdaq.com"]
NQ_HDR = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*",
          "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
YH = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
YH_TF = {"15m": ("15m", "60d"), "1h": ("60m", "1y"), "4h": ("60m", "1y"),
         "1d": ("1d", "5y"), "1w": ("1wk", "10y")}

def _num(s):
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except Exception:
        return 0.0

def _clean_sym(sym):
    s = (sym or "").strip().upper()
    if not s or "^" in s or " " in s:
        return None
    return s.replace("/", "-").replace(".", "-")        # BRK/B, BRK.B -> BRK-B (Yahoo style)

def _txt_lines(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        return r.text.splitlines() if r.ok else []
    except Exception:
        return []

def _stocks_fallback(exchanges):
    out = {}
    if "NASDAQ" in exchanges:
        # Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
        for ln in _txt_lines("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")[1:]:
            p = ln.split("|")
            if len(p) > 6 and p[3] == "N" and p[6] == "N":
                s = _clean_sym(p[0])
                if s:
                    out[s] = 0.0
    if "NYSE" in exchanges:
        # ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
        for ln in _txt_lines("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")[1:]:
            p = ln.split("|")
            if len(p) > 6 and p[2] == "N" and p[4] == "N" and p[6] == "N":
                s = _clean_sym(p[0])
                if s:
                    out[s] = 0.0
    return list(out.items())

def stocks_universe(exchanges):
    """[(symbol, dollar_volume)] for all listed stocks on the chosen exchanges (cached 1h)."""
    store = _store()
    key = ("__uni__", "stocks", tuple(sorted(exchanges)))
    with _CACHE_LOCK:
        hit = store.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    out = {}
    for ex in exchanges:
        d = fetch(NQ, "/api/screener/stocks",
                  {"tableonly": "true", "limit": 10000, "offset": 0,
                   "exchange": ex.lower(), "download": "true"}, headers=NQ_HDR)
        try:
            data = (d or {}).get("data") or {}
            rows = data.get("rows") or (data.get("table") or {}).get("rows") or []
        except Exception:
            rows = []
        for r in rows:
            s = _clean_sym(r.get("symbol"))
            if s:
                out[s] = _num(r.get("volume")) * _num(r.get("lastsale"))
    pairs = list(out.items()) if out else _stocks_fallback(exchanges)
    if pairs:
        with _CACHE_LOCK:
            store[key] = (time.time(), pairs)
    return pairs

def _yahoo_chart(sym, interval, rng):
    d = fetch(YH, f"/v8/finance/chart/{quote(sym)}",
              {"interval": interval, "range": rng, "includePrePost": "false"})
    try:
        res = d["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        off = (res.get("meta") or {}).get("gmtoffset", 0) or 0
        vols = q.get("volume") or [0] * len(res["timestamp"])
        rows = [[t, o, h, l, c, v or 0] for t, o, h, l, c, v in
                zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], vols)
                if None not in (o, h, l, c)]
        return rows, off
    except Exception:
        return None, 0

def _resample_4h(rows, off):
    """Group 60m bars of each session day in blocks of 4 (09:30-13:30, 13:30-16:00 ET)."""
    out, day, n, cur = [], None, 0, None
    for t, o, h, l, c, v in rows:
        d = (t + off) // 86400
        if d != day:
            day, n = d, 0
        if n % 4 == 0:
            if cur:
                out.append(cur)
            cur = [t, o, h, l, c, v]
        else:
            cur[2], cur[3], cur[4], cur[5] = max(cur[2], h), min(cur[3], l), c, cur[5] + v
        n += 1
    if cur:
        out.append(cur)
    return out

def stocks_ohlc(sym, tf):
    interval, rng = YH_TF[tf]
    rows, off = _yahoo_chart(sym, interval, rng)
    if not rows:
        return None
    if tf == "4h":
        rows = _resample_4h(rows, off)
    return [[r[0] * 1000, r[1], r[2], r[3], r[4], r[5]] for r in rows]     # ms, like the crypto feeds

PROVIDERS = {
    "binance": (binance_universe, binance_ohlc),
    "bybit": (bybit_universe, bybit_ohlc),
    "okx": (okx_universe, okx_ohlc),
    "yahoo": (None, stocks_ohlc),
}

@st.cache_resource
def _store():
    return {}

_SHARED = _store()

# Shared cache is written by many worker threads -> guard with a lock and prune it.
_CACHE_LOCK = threading.Lock()
_FH_LOCK = threading.Lock()          # serialises Finnhub calls (rate limit)
_CACHE_MAX = 20000

def get_closes(store, name, sym, tf, ttl=90):
    key = (name, sym, tf)
    now = time.time()
    with _CACHE_LOCK:
        hit = store.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1].tolist()
    try:
        raw = PROVIDERS[name][1](sym, tf)
        if not raw:
            return None
        arr = np.asarray([float(c[4]) for c in raw][-CLOSES_KEPT:], dtype="float64")
    except Exception:
        return None
    with _CACHE_LOCK:
        if len(store) >= _CACHE_MAX:
            for k in [k for k, v in store.items() if now - v[0] >= ttl]:
                store.pop(k, None)
            if len(store) >= _CACHE_MAX:             # still full: drop the oldest quarter
                for k in sorted(store, key=lambda k: store[k][0])[: _CACHE_MAX // 4]:
                    store.pop(k, None)
        store[key] = (now, arr)
    return arr.tolist()

def get_candles(name, sym, tf):
    try:
        raw = PROVIDERS[name][1](sym, tf)
        if not raw:
            return None
        raw = [(list(r) + [0.0] * 6)[:6] for r in raw]          # [t, o, h, l, c, volume]
        df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "volume"])
        df["t"] = pd.to_datetime(df["t"].astype("int64"), unit="ms")
        return df.astype({c: float for c in ["open", "high", "low", "close", "volume"]}).reset_index(drop=True)
    except Exception:
        return None

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
    active = [(tf, r) for tf, r in rules if not all(x == "Any" for x in r)]
    if not active:
        return None
    total = len(active)
    need = max(lf, ls) + 2
    idx = -1 if live else -2
    res, passed, fails = {}, 0, 0
    # Highest timeframe first: fewest bars to download, and the most selective filter,
    # so fast mode drops most symbols after only 1-2 requests.
    for tf, rule in sorted(active, key=lambda x: TF_MINUTES[x[0]], reverse=True):
        closes = closes_fn(sym, tf)
        if not closes or len(closes) < need:
            return None
        s_fast, s_slow = ema_series(closes, lf), ema_series(closes, ls)
        if not s_fast or not s_slow:
            return None
        p, e50, e200 = closes[idx], s_fast[idx], s_slow[idx]
        if e50 <= 0 or e200 <= 0:
            return None
        ok, d50, d200 = evaluate(rule, p, e50, e200, max_dist)
        res[tf] = (ok, p, d50, d200)
        passed += ok
        fails += (not ok)
        if fast and fails >= 2:      # can no longer be MATCH or WATCH -> stop fetching
            break
    row, price_set = {"Symbol": sym, "Price": 0.0}, False
    for tf, _ in active:             # columns in the user's timeframe order
        if tf not in res:
            continue
        ok, p, d50, d200 = res[tf]
        row[tf] = "✅" if ok else "❌"
        row[f"{tf} %50"] = round(d50, 2)
        row[f"{tf} %200"] = round(d200, 2)
        if not price_set:            # one consistent price (first active TF)
            row["Price"] = p
            price_set = True
    row["Passed"] = f"{passed}/{total}"
    row["Status"] = "MATCH" if passed == total else ("WATCH" if total >= 2 and passed == total - 1 else "-")
    row["_score"] = passed / total
    return row

def safe_scan_symbol(*args):
    """Crash protection: one bad symbol must never kill the whole scan."""
    try:
        return scan_symbol(*args)
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def run_scan(market, order, exchanges, rules, lf, ls, max_dist, live, top_n, min_vol, fast):
    store = _store()

    if market == "stocks":
        try:
            pairs = stocks_universe(exchanges)
        except Exception:
            pairs = []
        if any(v > 0 for _, v in pairs):             # volume known -> apply the floor
            pairs = [(s, v) for s, v in pairs if v >= min_vol]
        pairs.sort(key=lambda x: x[1], reverse=True)
        syms = [s for s, _ in pairs[:top_n]]
        if not syms:
            return "none", []
        fn = partial(get_closes, store, "yahoo", ttl=300)
        with ThreadPoolExecutor(max_workers=STOCK_WORKERS) as ex:
            rows = [r for r in ex.map(
                lambda s: safe_scan_symbol(s, fn, rules, lf, ls, max_dist, live, fast), syms) if r]
        return "yahoo", rows

    bad = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT")
    best = ("none", [])
    for name in order:
        try:
            uni_fn, _ = PROVIDERS[name]
            pairs = [(s, v) for s, v in uni_fn()
                     if s.endswith("USDT") and not s.endswith(bad) and v >= min_vol]
            pairs.sort(key=lambda x: x[1], reverse=True)
            syms = [s for s, _ in pairs[:top_n]]
            if not syms:
                continue
            fn = partial(get_closes, store, name)
            with ThreadPoolExecutor(max_workers=CRYPTO_WORKERS) as ex:
                rows = [r for r in ex.map(
                    lambda s: safe_scan_symbol(s, fn, rules, lf, ls, max_dist, live, fast), syms) if r]
        except Exception:
            continue
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

def trade_plan(df, direction, sl_mode, atr_mult, rr, entry=None):
    entry = df["close"].iloc[-1] if entry is None else entry
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
    # 1/3 of the position exits at each TP, SL fixed, leftovers marked to market at horizon end.
    # If SL and a TP are hit on the same bar, SL wins (conservative).
    ev = 0.0
    for t, R in zip(t_tp, rr):
        out = np.where((t < t_sl) & (t <= horizon), R,
                       np.where((t_sl <= t) & (t_sl <= horizon), -1.0, term_r))
        ev += out.mean() / len(rr)
    return {
        "paths": paths,
        "p_tp": [float(((t < t_sl) & (t <= horizon)).mean()) for t in t_tp],
        "p_sl": float(((t_sl <= np.minimum.reduce(t_tp)) & (t_sl <= horizon)).mean()),
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

# ── Extra signals: breakout · crash risk · open interest · analyst rating ──
BY_OI = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1d"}
BN_OI = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1d"}
BN_F = ["https://fapi.binance.com"]
FH = ["https://finnhub.io"]

def _clip01(x):
    return float(min(max(x, 0.0), 1.0))

def breakout_signal(df, n=20, vol_mult=1.5, live=True):
    """Close vs the previous n-bar high/low. +VOL = volume >= vol_mult x 20-bar average,
    +SQZ = Bollinger bandwidth was in its lowest 20% just before the break.
    (A live bar has only partial volume; turn 'live price' off for closed-bar confirmation.)"""
    d = df if live else df.iloc[:-1]
    if len(d) < n + 25:
        return None
    last_close = float(d["close"].iloc[-1])
    prev = d.iloc[-(n + 1):-1]
    hi, lo = float(prev["high"].max()), float(prev["low"].min())
    vavg = float(d["volume"].iloc[-21:-1].mean())
    vol_ok = vavg > 0 and float(d["volume"].iloc[-1]) >= vol_mult * vavg
    c = d["close"]
    bw = (c.rolling(20).std() * 4 / c.rolling(20).mean()).iloc[:-1].dropna().iloc[-120:]
    squeeze = len(bw) >= 40 and float(bw.iloc[-1]) <= float(bw.quantile(0.2))
    if last_close > hi:
        return "🚀 BREAKOUT" + (" +VOL" if vol_ok else "") + (" +SQZ" if squeeze else "")
    if last_close < lo:
        return "🔻 BREAKDOWN" + (" +VOL" if vol_ok else "")
    if hi > 0 and last_close >= hi * 0.99:
        return f"⏳ near high ({(last_close / hi - 1) * 100:+.1f}%)"
    return "-"

def crash_score(df, live=True):
    """0-100 crash-risk score (higher = more dangerous). Everything is ATR-normalised so it
    is comparable across coins and stocks. Returns (score, detail) or None."""
    d = df if live else df.iloc[:-1]
    if len(d) < 60:
        return None
    c = d["close"]
    a = atr(d)
    a_now = float(a.iloc[-1])
    if not np.isfinite(a_now) or a_now <= 0:
        return None
    price = float(c.iloc[-1])
    s_dd = _clip01((float(d["high"].iloc[-60:].max()) - price) / a_now / 10)       # drawdown, 10 ATR = max
    s_vol = _clip01((a_now / float(a.iloc[-100:].mean()) - 1) / 1.5)                # volatility spike
    vavg = d["volume"].rolling(20).mean().shift(1)
    big_red = ((d["open"] - d["close"]) >= a) & (d["volume"] > 1.5 * vavg)         # heavy-volume red bars
    s_red = _clip01(float(big_red.iloc[-10:].sum()) / 3)
    cl = c.tolist()
    e50, e200 = ema_series(cl, 50), ema_series(cl, 200)
    s_trend = 0.0
    if e50 and price < e50[-1]:
        s_trend += 0.4
    if e200 and price < e200[-1]:
        s_trend += 0.4
    if e50 and e200 and e50[-1] < e200[-1]:
        s_trend += 0.2
    s_mom = _clip01(-(price - float(c.iloc[-6])) / a_now / 4)                       # 5-bar fall, 4 ATR = max
    score = 100 * (0.25 * s_dd + 0.15 * s_vol + 0.20 * s_red + 0.25 * s_trend + 0.15 * s_mom)
    detail = (f"drawdown {s_dd * 100:.0f} · volatility {s_vol * 100:.0f} · red/volume {s_red * 100:.0f} · "
              f"trend loss {s_trend * 100:.0f} · momentum {s_mom * 100:.0f}")
    return int(round(score)), detail

def oi_history(sym, tf):
    """Open-interest series (oldest -> newest) from a USDT perpetual, or None if there is none.
    Bybit first, Binance futures as fallback (Binance is geo-blocked in some regions)."""
    d = fetch(BY, "/v5/market/open-interest",
              {"category": "linear", "symbol": sym, "intervalTime": BY_OI[tf], "limit": 200})
    try:
        vals = [float(x["openInterest"]) for x in reversed(d["result"]["list"])]
        if len(vals) >= 3:
            return vals
    except Exception:
        pass
    d = fetch(BN_F, "/futures/data/openInterestHist", {"symbol": sym, "period": BN_OI[tf], "limit": 200})
    try:
        vals = [float(x["sumOpenInterest"]) for x in d]
        if len(vals) >= 3:
            return vals
    except Exception:
        pass
    return None

def oi_signal(sym, tf, cdf, lb, live):
    oi = oi_history(sym, tf)
    if not oi:
        return "n/a (no perp)"
    if not live:
        oi = oi[:-1]
    d = cdf if live else cdf.iloc[:-1]
    ob = lb * 7 if tf == "1w" else lb          # weekly uses daily OI
    if len(oi) <= ob or len(d) <= lb or oi[-1 - ob] <= 0:
        return "n/a"
    oc = (oi[-1] / oi[-1 - ob] - 1) * 100
    up = float(d["close"].iloc[-1]) > float(d["close"].iloc[-1 - lb])
    if oc >= 2:
        return f"🟢 OI↑ {oc:+.1f}% (longs building)" if up else f"🔴 OI↑ {oc:+.1f}% (shorts building)"
    if oc <= -2:
        return f"🟡 OI↓ {oc:+.1f}% (short covering)" if up else f"🟠 OI↓ {oc:+.1f}% (long exits)"
    return f"➖ OI flat {oc:+.1f}%"

def _analyst_label(r):
    sb, b, h = r.get("strongBuy") or 0, r.get("buy") or 0, r.get("hold") or 0
    s, ss = r.get("sell") or 0, r.get("strongSell") or 0
    n = sb + b + h + s + ss
    if n == 0:
        return "n/a", None
    sc = (2 * sb + b - s - 2 * ss) / n                       # -2 .. +2
    lab = ("Strong Buy" if sc >= 1.2 else "Buy" if sc >= 0.4 else "Hold" if sc > -0.4
           else "Sell" if sc > -1.2 else "Strong Sell")
    return f"{lab} ({n})", sc

def _analyst_trend(d):
    """Latest Finnhub recommendation vs the previous month: label gets ▲ / ▼ when
    the consensus score moved by >= 0.15 (on the -2..+2 scale)."""
    rows = sorted(d, key=lambda r: r.get("period", ""), reverse=True)
    label, sc = _analyst_label(rows[0])
    if sc is None or len(rows) < 2:
        return label, sc
    _, prev = _analyst_label(rows[1])
    if prev is not None:
        if sc - prev >= 0.15:
            label += " ▲"
        elif sc - prev <= -0.15:
            label += " ▼"
    return label, sc

def analyst_rating(sym, key):
    """(label, score) from Finnhub recommendation trends (with ▲/▼ trend); cached 24h, ~1 call/second."""
    store = _store()
    k = ("__fh__", sym)
    with _CACHE_LOCK:
        hit = store.get(k)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    with _FH_LOCK:
        time.sleep(1.05)
        d = fetch(FH, "/api/v1/stock/recommendation", {"symbol": sym.replace("-", "."), "token": key})
    if not isinstance(d, list):
        return "n/a", None                                   # error / rate limit: do not cache
    res = _analyst_trend(d) if d else ("n/a", None)
    with _CACHE_LOCK:
        store[k] = (time.time(), res)
    return res

# ── Volume flow: how unusual is volume, and which side is it on? ─────────────
def _robust_z(x, window=50):
    """z-score of the last value vs the previous `window` values using median/MAD.
    Volume is heavy-tailed, so mean/std z-scores are dominated by one spike."""
    s = np.asarray(x, dtype="float64")
    s = s[np.isfinite(s)]
    if len(s) < window + 2:
        return 0.0
    ref = s[-window - 1:-1]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    if mad <= 1e-12:
        mad = float(ref.std()) or 1.0
    return float((s[-1] - med) / mad)

def volume_flow(df, live=True, window=50, k=3):
    """Returns (label, vol_z, delta_z) or None.
    vol_z   = robust z of log-volume (is volume unusually high/low?)
    delta_z = robust z of the k-bar net 'delta proxy' (is that volume buying or selling?)
    delta proxy per bar = volume * CLV, CLV = (2C - H - L)/(H - L) in [-1, +1].
    (Exact buy/sell volume needs taker data: Binance klines column 9 'taker buy base volume'.)"""
    d = df if live else df.iloc[:-1]
    if len(d) < window + k + 5:
        return None
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    clv = ((2 * d["close"] - d["high"] - d["low"]) / rng).fillna(0.0)
    delta = clv * d["volume"]
    vz = _robust_z(np.log1p(d["volume"].values), window)
    dz = _robust_z(delta.rolling(k).sum().dropna().values, window)
    if vz >= 1.5 and dz >= 1.0:
        lab = "🟢 buy surge"
    elif vz >= 1.5 and dz <= -1.0:
        lab = "🔴 sell surge"
    elif dz >= 1.5:
        lab = "🟢 accumulation"
    elif dz <= -1.5:
        lab = "🔴 distribution"
    elif vz <= -1.0:
        lab = "💤 volume dry-up"
    else:
        lab = "➖ neutral"
    return lab, vz, dz

def flow_text(f):
    return "n/a" if not f else f"{f[0]} (vz {f[1]:+.1f} · dz {f[2]:+.1f})"
def rules_direction(rules):
    """True = the rule set is long-biased (Above / 50>200), False = short-biased."""
    up = dn = 0
    for _, (a, b, c) in rules:
        up += (a == "Above") + (b == "Above") + (c == "50>200")
        dn += (a == "Below") + (b == "Below") + (c == "50<200")
    return up >= dn

def conviction(ex, long, status, regime):
    """0-100 setup score + breakdown. Direction-aware: the same signal helps a long and hurts a short.
    ex keys (all optional): Breakout, Crash, OI, _dz (flow z), _imb (weighted book imbalance), _as (analyst)."""
    sg = 1 if long else -1
    clip = lambda x: max(-1.0, min(1.0, float(x)))
    ok = lambda v: v is not None and np.isfinite(v)
    p = {"MTF": 20 if status == "MATCH" else 8 if status == "WATCH" else 0}
    if ok(ex.get("_dz")):
        p["Flow"] = 12 * clip(sg * ex["_dz"] / 3)
    if ok(ex.get("_imb")):
        p["Book"] = 8 * clip(sg * ex["_imb"] / 0.4)
    v = {"🟢": 8, "🟡": 3, "🟠": -5, "🔴": -8}.get(str(ex.get("OI") or "")[:1])
    if v is not None:
        p["OI"] = sg * v
    bo = str(ex.get("Breakout") or "")
    if bo.startswith("🚀"):
        p["Break"] = sg * (10 + 3 * ("+VOL" in bo) + 2 * ("+SQZ" in bo))
    elif bo.startswith("🔻"):
        p["Break"] = -sg * (10 + 3 * ("+VOL" in bo))
    if ok(ex.get("Crash")):
        p["Crash"] = sg * (5 - 15 * ex["Crash"] / 100)
    if regime:
        p["Regime"] = sg * {"RISK-ON": 10, "CAUTION": 0, "RISK-OFF": -15}[regime["state"]]
    if ok(ex.get("_as")):
        p["Analyst"] = 6 * clip(sg * ex["_as"] / 1.2)
    score = int(round(max(0, min(100, 50 + sum(p.values())))))
    return score, p

def grade(s):
    return "A" if s >= 80 else "B" if s >= 65 else "C" if s >= 50 else "D"

def blocked_tps(lv, entry, tps, long):
    """Targets that sit behind an opposing wall: [(tp_number, wall)]."""
    if not lv:
        return []
    out = []
    for i, tp in enumerate(tps):
        lo, hi = (entry, tp) if long else (tp, entry)
        for w in (lv["ask_walls"] if long else lv["bid_walls"]):
            if lo <= w["price"] <= hi:
                out.append((i + 1, w))
                break
    return out

def fill_prob(close, price, limit, long, horizon, n_paths=3000, seed=7):
    """Chance that a limit order is touched within the horizon (bootstrap, drift removed, close-based)."""
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(close))[-500:]
    r = r - r.mean()
    paths = price * np.exp(np.cumsum(r[rng.integers(0, len(r), (n_paths, horizon))], axis=1))
    return float((paths.min(axis=1) <= limit).mean() if long else (paths.max(axis=1) >= limit).mean())
    
# ── Order-book walls (crypto only: stocks have no free L2 feed) ──────────────
def _lv(rows):
    return [(float(x[0]), float(x[1])) for x in rows]
def _ttl(key, ttl, fn):
    store, now = _SHARED, time.time()
    with _CACHE_LOCK:
        hit = store.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    if val is not None:
        with _CACHE_LOCK:
            store[key] = (now, val)
    return val

def get_orderbook(src, sym):
    return _ttl(("__ob__", src, sym), 15, lambda: _get_orderbook_raw(src, sym))

def _tag_persistence(bw, aw, src, sym, keep=6, min_gap=20, max_age=900):
    """Adds w['persist'] = share of recent snapshots that had a wall at the same place
    (None = no history yet). Spoofed walls tend to vanish, real ones stay."""
    store, now = _SHARED, time.time()
    key = ("__obh__", src, sym)
    with _CACHE_LOCK:
        hit = store.get(key)
    hist = [h for h in (hit[1] if hit else []) if now - h[0] <= max_age]
    for side, ws in (("b", bw), ("a", aw)):
        for w in ws:
            w["persist"] = (sum(any(s == side and not (w["hi"] < lo or w["lo"] > hi) for s, lo, hi in h[1])
                                for h in hist) / len(hist)) if hist else None
    if not hist or now - hist[-1][0] >= min_gap:
        hist.append((now, [("b", w["lo"], w["hi"]) for w in bw] + [("a", w["lo"], w["hi"]) for w in aw]))
    with _CACHE_LOCK:
        store[key] = (now, hist[-keep:])

def _pers(w):
    return "" if w.get("persist") is None else f" · seen {w['persist']:.0%}"
    
def _get_orderbook_raw(src, sym):
    """(bids, asks) as [(price, qty)], or None."""
    try:
        if src == "binance":
            d = fetch(BN, "/api/v3/depth", {"symbol": sym, "limit": 1000})
            return _lv(d["bids"]), _lv(d["asks"])
        if src == "bybit":
            d = fetch(BY, "/v5/market/orderbook", {"category": "spot", "symbol": sym, "limit": 200})["result"]
            return _lv(d["b"]), _lv(d["a"])
        if src == "okx":
            d = fetch(OK, "/api/v5/market/books", {"instId": sym[:-4] + "-USDT", "sz": 400})["data"][0]
            return _lv(d["bids"]), _lv(d["asks"])
    except Exception:
        pass
    return None

def _usd(x):
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}k"
    
def orderbook_levels(src, sym, price, atr_abs=0.0, band=0.03, bins=40, z=3.0):
    """Bucket resting orders (USD notional) around price and flag buckets that are
    statistical outliers (robust z >= z) as walls. Also returns book imbalance (over the range
    both sides of the snapshot cover, max +-1%) and a wall-based limit-entry suggestion."""
    ob = get_orderbook(src, sym)
    if not ob or not ob[0] or not ob[1]:
        return None
    bids, asks = ob
    min_bid = min(p for p, _ in bids)
    max_ask = max(p for p, _ in asks)
    # depth snapshots only cover a limited range on liquid coins: never bucket beyond what we have
    lo = max(price * (1 - band), min_bid)
    hi = min(price * (1 + band), max_ask)
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2

    def bucket(levels):
        arr = np.array([(p, p * q) for p, q in levels if lo <= p <= hi])
        if arr.size == 0:
            return np.zeros(bins)
        idx = np.clip(np.searchsorted(edges, arr[:, 0], side="right") - 1, 0, bins - 1)
        return np.bincount(idx, weights=arr[:, 1], minlength=bins)

    b, a = bucket(bids), bucket(asks)
    nz = np.concatenate([b, a])
    nz = nz[nz > 0]
    if len(nz) < 8:
        return None
    med = float(np.median(nz))
    mad = float(np.median(np.abs(nz - med)) * 1.4826) or float(nz.std()) or 1.0
    mad = max(mad, 0.05 * med)          # floor: a very uniform book must not turn noise into "walls"

    def walls(arr):
        out, used = [], []
        for i in np.argsort(arr)[::-1]:
            zs = (arr[i] - med) / mad
            if zs < z or len(out) >= 3:
                break
            if any(abs(i - j) <= 1 for j in used):          # skip neighbours of an accepted wall
                continue
            used.append(i)
            # absorb adjacent buckets that are themselves elevated, so a wall split across
            # a bucket edge is reported with its full size
            i0 = i - 1 if i > 0 and (arr[i - 1] - med) / mad >= z / 2 else i
            i1 = i + 1 if i < bins - 1 and (arr[i + 1] - med) / mad >= z / 2 else i
            usd = float(arr[i0:i1 + 1].sum())
            ctr = float(np.average(mid[i0:i1 + 1], weights=arr[i0:i1 + 1]))
            out.append({"price": ctr, "lo": float(edges[i0]), "hi": float(edges[i1 + 1]),
                        "usd": usd, "z": float(zs),
                        "dist_pct": (ctr / price - 1) * 100})
        return out

    bw, aw = walls(b), walls(a)
    _tag_persistence(bw, aw, src, sym)
    # imbalance over the range BOTH sides actually cover (max 1%), so the sides are comparable
    cov = min(0.01, (price - min_bid) / price, (max_ask - price) / price)
    if cov <= 0:
        return None

    def near(lv, weighted=False):
        tot = 0.0
        for p, q in lv:
            d = abs(p / price - 1)
            if d <= cov:
                tot += p * q * ((1 - 0.5 * d / cov) if weighted else 1.0)
        return tot

    B, A = near(bids), near(asks)
    imb = (B - A) / (B + A) if (B + A) > 0 else 0.0
    Bw, Aw = near(bids, True), near(asks, True)          # closer orders count more (less spoof-prone)
    imb_w = (Bw - Aw) / (Bw + Aw) if (Bw + Aw) > 0 else 0.0

    buf = 0.1 * atr_abs
    long_entry = short_entry = None
    below = [w for w in bw if w["hi"] <= price * 1.001]
    above = [w for w in aw if w["lo"] >= price * 0.999]
    # prefer big walls, but penalise walls that are far away (less likely to be reached / fill)
    pw = lambda w: 1.0 if w.get("persist") is None else 0.5 + 0.5 * w["persist"]
    score = lambda w: w["usd"] * pw(w) / (1 + 100 * abs(w["price"] / price - 1))   # <- replaces old score
    if below:
        w = max(below, key=score)
        long_entry = {"entry": min(w["hi"] + buf, price), "invalid_below": w["lo"] - buf, "wall": w}
    if above:
        w = max(above, key=score)
        short_entry = {"entry": max(w["lo"] - buf, price), "invalid_above": w["hi"] + buf, "wall": w}
    return {"bid_walls": bw, "ask_walls": aw, "imbalance": float(imb), "imbalance_w": float(imb_w),
            "long_entry": long_entry, "short_entry": short_entry}

def book_label(lv, price):
    if not lv:
        return "n/a"
    imb = lv["imbalance"]
    icon = "🟢" if imb > 0.15 else "🔴" if imb < -0.15 else "➖"
    parts = [f"{icon} imb {imb:+.0%}"]
    if lv["bid_walls"]:
        w = max(lv["bid_walls"], key=lambda x: x["usd"])
        parts.append(f"bid {_usd(w['usd'])} @ {(w['price'] / price - 1) * 100:+.1f}%")
    if lv["ask_walls"]:
        w = max(lv["ask_walls"], key=lambda x: x["usd"])
        parts.append(f"ask {_usd(w['usd'])} @ {(w['price'] / price - 1) * 100:+.1f}%")
    return " · ".join(parts)

def add_walls(fig, lv):
    """Draw the walls on the trade-planner chart."""
    if not lv:
        return fig
    for w in lv["bid_walls"]:
        fig.add_hrect(y0=w["lo"], y1=w["hi"], fillcolor="rgba(34,197,94,.22)", line_width=0,
                      annotation_text=f"BID wall {_usd(w['usd'])} (z {w['z']:.1f}){_pers(w)}",
                      annotation_position="bottom left", annotation_font_color="#22c55e")
    for w in lv["ask_walls"]:
        fig.add_hrect(y0=w["lo"], y1=w["hi"], fillcolor="rgba(239,68,68,.22)", line_width=0,
                      annotation_text=f"ASK wall {_usd(w['usd'])} (z {w['z']:.1f}){_pers(w)}",
                      annotation_position="top left", annotation_font_color="#ef4444")
    return fig

# ── Footprint-style volume at price from recent trades (crypto) ──────────────
def _get_trades_raw(src, sym):
    """[(price, qty, taker_is_buy)] for the most recent trades, or None.
    Bybit spot recent-trade is capped at ~60 rows, so the linear perp (up to 1000) is tried first."""
    try:
        if src == "binance":
            d = fetch(BN, "/api/v3/aggTrades", {"symbol": sym, "limit": 1000})
            return [(float(t["p"]), float(t["q"]), not t["m"]) for t in d]   # m=True -> seller was taker
        if src == "bybit":
            d = fetch(BY, "/v5/market/recent-trade", {"category": "linear", "symbol": sym, "limit": 1000})
            if not (isinstance(d, dict) and (d.get("result") or {}).get("list")):
                d = fetch(BY, "/v5/market/recent-trade", {"category": "spot", "symbol": sym, "limit": 60})
            return [(float(t["price"]), float(t["size"]), t["side"] == "Buy") for t in d["result"]["list"]]
        if src == "okx":
            base = sym[:-4] if sym.endswith("USDT") else sym
            d = fetch(OK, "/api/v5/market/trades", {"instId": f"{base}-USDT", "limit": 500})
            return [(float(t["px"]), float(t["sz"]), t["side"] == "buy") for t in d["data"]]
    except Exception:
        pass
    return None
    
def get_trades(src, sym):
    return _ttl(("__tr__", src, sym), 10, lambda: _get_trades_raw(src, sym))

def footprint(src, sym, bins=40, imb_ratio=3.0):
    """Buy/sell volume per price bucket from the latest trades (a mini footprint).
    Returns (DataFrame[price, buy, sell, delta, total, imbalance], poc_price, cvd) or None.
    Extra info is stored in fp.attrs: vah, val (70% value area), delta_z (CVD vs trade-size noise)."""
    tr = get_trades(src, sym)
    if not tr or len(tr) < 50:
        return None
    p = np.array([t[0] for t in tr]); q = np.array([t[1] for t in tr])
    isb = np.array([t[2] for t in tr], dtype=bool)
    lo, hi = p.min(), p.max()
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    buy = np.bincount(idx, weights=q * isb, minlength=bins)
    sell = np.bincount(idx, weights=q * ~isb, minlength=bins)
    fp = pd.DataFrame({"price": (edges[:-1] + edges[1:]) / 2, "buy": buy, "sell": sell})
    fp["delta"], fp["total"] = fp["buy"] - fp["sell"], fp["buy"] + fp["sell"]

    # imbalance only counts on buckets with meaningful volume (avoids "BUY" on 0.0001 vs 0)
    active = fp["total"] >= np.median(fp["total"][fp["total"] > 0])
    buy_dom = fp["buy"] >= imb_ratio * fp["sell"].clip(lower=1e-12)
    sell_dom = fp["sell"] >= imb_ratio * fp["buy"].clip(lower=1e-12)
    fp["imbalance"] = np.where(active & buy_dom, "BUY", np.where(active & sell_dom, "SELL", ""))

    poc_i = int(fp["total"].idxmax())
    # value area: expand from the POC until 70% of the volume is covered
    tot, target = float(fp["total"].sum()), 0.70 * float(fp["total"].sum())
    l = r = poc_i
    acc = float(fp["total"].iloc[poc_i])
    while acc < target and (l > 0 or r < bins - 1):
        up = fp["total"].iloc[r + 1] if r < bins - 1 else -1
        dn = fp["total"].iloc[l - 1] if l > 0 else -1
        if up >= dn:
            r += 1; acc += up
        else:
            l -= 1; acc += dn
    # z-score of net signed flow: how far CVD is from what random buy/sell would give
    signed = np.where(isb, q, -q)
    sd = float(signed.std() * np.sqrt(len(signed))) or 1.0
    fp.attrs.update(vah=float(fp["price"].iloc[r]), val=float(fp["price"].iloc[l]),
                    delta_z=float(signed.sum() / sd))
    return fp, float(fp["price"].iloc[poc_i]), float(fp["delta"].sum())

def footprint_chart(fp, poc, entry=None):
    fig = go.Figure()
    h = (fp["price"].iloc[1] - fp["price"].iloc[0]) * 0.9
    fig.add_trace(go.Bar(y=fp["price"], x=fp["buy"], orientation="h", name="Aggressive buy",
                         marker_color="#22c55e", width=h))
    fig.add_trace(go.Bar(y=fp["price"], x=-fp["sell"], orientation="h", name="Aggressive sell",
                         marker_color="#ef4444", width=h))
    fig.add_hline(y=poc, line_dash="dot", line_color="#f59e0b", annotation_text="POC")
    attrs = getattr(fp, "attrs", {}) or {}
    for k, name in (("vah", "VAH"), ("val", "VAL")):
        if k in attrs:
            fig.add_hline(y=attrs[k], line_dash="dash", line_width=1, line_color="#94a3b8",
                          annotation_text=name, annotation_position="top right")
    if entry:
        fig.add_hline(y=entry, line_color="#3b82f6", annotation_text="ENTRY")
    title = "Volume at price (buy right / sell left)"
    if "delta_z" in attrs:
        title += f" · flow z {attrs['delta_z']:+.1f}"
    fig.update_layout(barmode="relative", template="plotly_dark", height=420,
                      margin=dict(l=10, r=10, t=30, b=10), title=title)
    return fig

# ── Stocks: no free order book -> high-volume nodes as proxy support/resistance ──
def volume_nodes(df, bins=40, top=3):
    """Volume-at-price from candles (typical price). POC + top high-volume nodes act as
    support/resistance. A proxy only: it is traded volume, not resting orders."""
    d = df.iloc[-300:]
    tp = ((d["high"] + d["low"] + d["close"]) / 3).values
    hist, edges = np.histogram(tp, bins=bins, weights=d["volume"].values)
    mid = (edges[:-1] + edges[1:]) / 2
    order = np.argsort(hist)[::-1]
    nodes, used = [], []
    for i in order:
        if any(abs(i - j) <= 1 for j in used):
            continue
        used.append(i)
        nodes.append((float(mid[i]), float(hist[i])))
        if len(nodes) >= top:
            break
    return nodes          # first = POC

# ── Enrichment for MATCH/WATCH rows ─────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def enrich_rows(market, src, syms, tf, n_break, vol_mult, live, oi_lb, fh_key):
    stocks = market == "stocks"

    def one(s):
        out = {"Breakout": "n/a", "Crash": None}
        try:
            cdf = get_candles(src, s, tf)
            if cdf is None or len(cdf) < 30:
                return s, out
            out["Breakout"] = breakout_signal(cdf, n_break, vol_mult, live) or "n/a"
            cs = crash_score(cdf, live)
            out["Crash"] = cs[0] if cs else None
            fl = volume_flow(cdf, live)
            out["Flow"] = flow_text(fl)
            out["_dz"] = fl[2] if fl else None
            if not stocks:
                out["OI"] = oi_signal(s, tf, cdf, oi_lb, live)
                try:
                    px = float(cdf["close"].iloc[-1])
                    lv_ = orderbook_levels(src, s, px, float(atr(cdf).iloc[-1]))
                    out["Book"] = book_label(lv_, px)
                    out["_imb"] = lv_["imbalance_w"] if lv_ else None
                except Exception:
                    out["Book"] = "n/a"
        except Exception:
            pass
        return s, out

    with ThreadPoolExecutor(max_workers=8 if stocks else 16) as ex:
        res = dict(ex.map(one, syms))
    if stocks and fh_key:
        for i, s in enumerate(syms):
            if i >= ANALYST_MAX:
                res[s]["Analyst"] = "limit"
                continue
            label, score = analyst_rating(s, fh_key)
            res[s]["Analyst"], res[s]["_as"] = label, score
    return res

@st.cache_data(ttl=300, show_spinner=False)
def market_regime(market, src):
    """Benchmark trend: BTC (crypto) or SPY (stocks) on 1D."""
    sym = "SPY" if market == "stocks" else "BTCUSDT"
    df = get_candles(src, sym, "1d")
    if df is None or len(df) < 210:
        return None
    c = df["close"].tolist()
    p, e50, e200 = c[-1], ema_series(c, 50)[-1], ema_series(c, 200)[-1]
    cs = crash_score(df, True)
    state = "RISK-OFF" if p < e200 else ("CAUTION" if p < e50 else "RISK-ON")
    return {"sym": sym, "state": state, "d200": (p / e200 - 1) * 100, "crash": cs[0] if cs else None}

# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
market = MARKETS[st.sidebar.radio("Market", list(MARKETS))]
is_stocks = market == "stocks"
ccy = "USD" if is_stocks else "USDT"

live = st.sidebar.toggle("Use live price (off = last closed candle)", True)
lf = st.sidebar.number_input("EMA fast", 1, 500, 50)
ls = st.sidebar.number_input("EMA slow", 1, 500, 200)
max_dist = st.sidebar.number_input("Max distance from EMA (%), 0 = off", 0.0, 100.0, 0.0, 0.5)
top_n = st.sidebar.slider("Top N by volume", 20, 8000, 1500, 50,
                          help="Crypto exchanges list a few hundred to ~1000 USDT pairs. "
                               "NASDAQ + NYSE together have ~6,000+ tickers: use the maximum to scan all of them.")
if is_stocks:
    min_vol = st.sidebar.number_input("Min daily $ volume (USD)", 0, 10**11, 1_000_000, 500_000)
    exch = st.sidebar.multiselect("Exchanges", STOCK_EXCH, default=STOCK_EXCH)
    order = []
else:
    min_vol = st.sidebar.number_input("Min 24h volume (USDT)", 0, 10**10, 100_000, 50_000)
    order = st.sidebar.multiselect("Data sources (priority order)", ["okx", "bybit", "binance"],
                                   default=["okx", "bybit", "binance"])
    exch = []
fast = st.sidebar.toggle("Fast mode (stop after 2 failed TFs)", True,
                         help="Much fewer API calls; MATCH/WATCH results are identical.")
refresh = st.sidebar.slider("Auto-refresh (minutes, 0 = off)", 0, 30, 0)
alerts = st.sidebar.toggle("Telegram alerts (needs secrets)", False)

st.sidebar.subheader("Extra signals")
extras_on = st.sidebar.toggle("Breakout · crash risk · flow · OI · book · analyst (MATCH/WATCH rows)", True,
                              help="Computed only for MATCH/WATCH rows, so the extra API cost is small.")
sig_tf = st.sidebar.selectbox("Signal timeframe", TFS, index=3)
n_break = st.sidebar.slider("Breakout lookback (bars)", 10, 100, 20)
vol_mult = st.sidebar.number_input("Breakout volume x average", 1.0, 10.0, 1.5, 0.1)
oi_lb = st.sidebar.slider("OI / price change lookback (bars)", 2, 30, 5)
crash_max = st.sidebar.slider("Crash protector: hide MATCH/WATCH with risk ≥ (100 = off)", 10, 100, 100, 5,
                              help="Meant for long setups. Leave at 100 when scanning for shorts.")
req_break = st.sidebar.toggle("Only breakouts", False)
req_oi = False
min_analyst = "Any"
fh_key = ""
if is_stocks:
    min_analyst = st.sidebar.selectbox("Min analyst rating", ["Any"] + list(ANALYST_MIN))
    try:
        fh_key = st.secrets["FINNHUB_KEY"]
    except Exception:
        fh_key = st.sidebar.text_input("Finnhub API key (free) for analyst ratings", type="password")
else:
    req_oi = st.sidebar.toggle("Only OI rising with price up", False)

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
            '<span>Multi-timeframe EMA scan · crypto + US stocks · trade planner · Monte Carlo</span></div>',
            unsafe_allow_html=True)
if st.button("🔄 Scan now"):
    run_scan.clear()

if is_stocks and not exch:
    st.warning("Select at least one exchange.")
    st.stop()
if not is_stocks and not order:
    st.warning("Select at least one data source.")
    st.stop()

if all(all(x == "Any" for x in r) for _, r in rules):
    st.warning("All rules are set to 'Any' - pick at least one condition.")
    st.stop()

t0 = time.time()
spin = (f"Scanning up to {top_n} stocks... the first full scan can take several minutes "
        "(later scans reuse the cache)") if is_stocks else f"Scanning up to {top_n} coins..."
with st.spinner(spin):
    try:
        provider, rows = run_scan(market, tuple(order), tuple(exch), tuple(rules), lf, ls,
                                  max_dist, live, top_n, min_vol, fast)
    except Exception as e:
        provider, rows = "none", []
        st.error(f"Scan crashed: {e}")

if not rows:
    run_scan.clear()   # never keep a failed/empty result cached for 2 minutes
    st.error("All data sources failed. Try another source order or retry in a minute.")
    st.stop()

data_src = "yahoo" if is_stocks else provider      # provider key used for candle downloads

df = pd.DataFrame(rows).sort_values(["_score", "Symbol"], ascending=[False, True]).drop(columns="_score")
_lead = ["Symbol", "Price"] + [c for tf in TFS for c in (tf, f"{tf} %50", f"{tf} %200") if c in df.columns]
df = df[_lead + [c for c in df.columns if c not in _lead]]          # stable column order (fast mode skips TFs)
match = df[df["Status"] == "MATCH"]
watch = df[df["Status"] == "WATCH"]

# ── Extra signals (MATCH/WATCH rows only) + filters ─────────────
extra, as_map, regime, hidden = {}, {}, None, 0
if extras_on:
    pool_syms = (list(match["Symbol"]) + list(watch["Symbol"]))[:ENRICH_MAX]
    if pool_syms:
        with st.spinner("Computing extra signals..."):
            try:
                extra = enrich_rows(market, data_src, tuple(pool_syms), sig_tf, n_break, vol_mult,
                                    live, oi_lb, fh_key)
            except Exception as e:
                st.warning(f"Extra signals failed: {e}")
    try:
        regime = market_regime(market, data_src)
    except Exception:
        regime = None
    for col in ("Breakout", "Crash", "Flow", "OI", "Book", "Analyst"):
        if any(col in v for v in extra.values()):
            df[col] = df["Symbol"].map(lambda s, c=col: extra.get(s, {}).get(c))
    if "Crash" in df.columns:
        df["Crash"] = pd.to_numeric(df["Crash"], errors="coerce")
    if extra:
        long_bias = rules_direction(rules)
        st_map = dict(zip(df["Symbol"], df["Status"]))
        df["Conviction"] = df["Symbol"].map(
    lambda s: conviction(extra[s], long_bias, st_map.get(s), regime)[0] if s in extra else None)

    as_map = {s: v.get("_as") for s, v in extra.items()}

    def keep(frame):
        f = frame
        if "Crash" in f.columns and crash_max < 100:
            f = f[~(f["Crash"] >= crash_max)]                       # NaN (unknown) is kept
        if req_break and "Breakout" in f.columns:
            f = f[f["Breakout"].astype(str).str.startswith("🚀")]
        if req_oi and "OI" in f.columns:
            f = f[f["OI"].astype(str).str.startswith("🟢")]
        if min_analyst != "Any" and any(v is not None for v in as_map.values()):
            f = f[pd.to_numeric(f["Symbol"].map(as_map), errors="coerce") >= ANALYST_MIN[min_analyst]]
        return f

    match = df[df["Status"] == "MATCH"]
    watch = df[df["Status"] == "WATCH"]
    n_before = len(match) + len(watch)
    match, watch = keep(match), keep(watch)
    hidden = n_before - len(match) - len(watch)
    if "Conviction" in df.columns:
        match = match.sort_values("Conviction", ascending=False, na_position="last")
        watch = watch.sort_values("Conviction", ascending=False, na_position="last")


m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scanned", len(df))
m2.metric("🟢 MATCH", len(match))
m3.metric("👀 WATCH", len(watch))
m4.metric("Source", provider.upper())
m5.metric("Scan time", f"{time.time() - t0:.1f}s")
st.caption(f"Updated {time.strftime('%H:%M:%S', time.gmtime())} (UTC)")
if regime:
    icon = {"RISK-ON": "🟢", "CAUTION": "🟡", "RISK-OFF": "🔴"}[regime["state"]]
    rc = "n/a" if regime["crash"] is None else f"{regime['crash']}/100"
    st.markdown(f"{icon} **Market regime: {regime['state']}** · {regime['sym']} 1D is "
                f"{regime['d200']:+.1f}% vs EMA200 · benchmark crash score {rc}")
if hidden:
    st.caption(f"{hidden} MATCH/WATCH rows hidden by the extra-signal filters (still visible in the All tab).")

now = set(match["Symbol"])
pkey = f"prev_matches_{market}"          # separate state per market -> no false alerts on switch
prev = st.session_state.get(pkey)
if alerts and prev is not None:
    for s in now - prev:
        telegram(f"ENTRY: {s} - all MTF EMA rules matched")
    for s in prev - now:
        telegram(f"EXIT: {s} - MTF EMA rules no longer matched")
st.session_state[pkey] = now

pct_cols = {c: st.column_config.NumberColumn(c, format="%.2f%%") for c in df.columns if "%" in c}
cfg = {"Price": st.column_config.NumberColumn("Price", format="%.6g"), **pct_cols}
if "Crash" in df.columns:
    cfg["Crash"] = st.column_config.ProgressColumn("Crash risk", min_value=0, max_value=100, format="%d")
if "Conviction" in df.columns:
    cfg["Conviction"] = st.column_config.ProgressColumn("Conviction", min_value=0, max_value=100, format="%d")

t1, t2, t3, t4 = st.tabs(["✅ MATCH", "👀 WATCH", "All", "🎯 Trade planner"])
for key, tab, data in (("match", t1, match), ("watch", t2, watch), ("all", t3, df)):
    with tab:
        if data.empty:
            st.info("Nothing here right now.")
        else:
            st.dataframe(data, width="content", hide_index=True, column_config=cfg)
            st.download_button("⬇️ CSV", data.to_csv(index=False), f"screener_{market}_{key}.csv",
                               key=f"dl_{key}")

# ── Trade planner ───────────────────────────────────────────────
with t4:
    pool = list(match["Symbol"]) + list(watch["Symbol"])
    if not pool:
        st.info("No MATCH/WATCH symbols to plan right now.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        sym = k1.selectbox("Symbol", pool, format_func=lambda s: f"{'🟢' if s in now else '👀'} {s}")
        ctf = k2.selectbox("Chart / entry TF", TFS, index=2)
        dirn = k3.selectbox("Direction", ["Auto", "Long", "Short"])
        sl_mode = k4.selectbox("Stop-loss", ["ATR", "Swing (10 bars)", "Wall invalidation"])
        k5, k6, k7, k8 = st.columns(4)
        atr_mult = k5.number_input("ATR multiple", 0.5, 10.0, 1.5, 0.25)
        rr_txt = k6.text_input("TP in R (TP1,TP2,TP3)", "1,2,3")
        horizon = k7.slider("MC horizon (bars)", 10, 200, 60, 5)
        n_paths = k8.select_slider("MC paths", [1000, 2000, 5000, 10000], 5000)
        k9, k10, k11 = st.columns(3)
        k12, k13 = st.columns(2)
        entry_mode = k12.selectbox("Entry", ["Market", "Wall limit"],
                                   help="Wall limit = rest an order just in front of the strongest wall.")
        cost_bps = k13.number_input("Round-trip cost (bps, fees + slippage)", 0.0, 200.0, 2.0 if is_stocks else 15.0, 1.0)
        drift = k9.toggle("Include historical drift", True)
        equity = k10.number_input(f"Account size ({ccy})", 10.0, 1e9, 1000.0, 100.0)
        risk_pct = k11.number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1)

        try:
            rr = [float(x) for x in rr_txt.split(",")][:3]
            assert len(rr) == 3 and all(x > 0 for x in rr)
        except Exception:
            rr = None
            st.error("Enter three positive R multiples, e.g. 1,2,3")

        cdf = get_candles(data_src, sym, ctf) if rr else None
        if rr is None:
            pass
        elif cdf is None or len(cdf) < max(lf, ls) + 5:
            st.error("Could not load candles for this symbol/timeframe.")
        else:
            try:
                closes = cdf["close"].tolist()
                e50, e200 = ema_series(closes, lf)[-1], ema_series(closes, ls)[-1]
                p = closes[-1]
                auto = "Long" if (p > e200 and e50 >= e200) else "Short" if (p < e200 and e50 <= e200) else \
                       ("Long" if p > e200 else "Short")
                direction = auto if dirn == "Auto" else dirn
                long = direction == "Long"
                px_last = float(cdf["close"].iloc[-1])
                lv = None
                if not is_stocks:
                    try:
                        lv = orderbook_levels(data_src, sym, px_last, float(atr(cdf).iloc[-1]))
                    except Exception:
                        lv = None
                sug = (lv["long_entry"] if long else lv["short_entry"]) if lv else None
                limit_px = sug["entry"] if (sug and entry_mode == "Wall limit") else None
                if entry_mode == "Wall limit" and not sug:
                    st.caption("No wall found on this side: using market entry.")
                entry, sl, tps, risk = trade_plan(cdf, direction, sl_mode, atr_mult, rr, entry=limit_px)
                if sl_mode == "Wall invalidation" and sug:
                    sl = sug.get("invalid_below") or sug.get("invalid_above")
                    risk = abs(entry - sl)
                    tps = [entry + (1 if long else -1) * risk * r for r in rr]
                if not np.isfinite(risk) or risk <= entry * 1e-6 or (long and sl >= entry) or (not long and sl <= entry):
                    raise ValueError("Stop is ~0 or on the wrong side of the entry. Try another SL mode / entry / TF.")
                mc = monte_carlo(cdf["close"].values, entry, sl, tps, rr, long, horizon, n_paths, drift)
                cost_r = cost_bps / 1e4 * entry / risk
                ev_net = mc["ev_r"] - cost_r
                fillp = fill_prob(cdf["close"].values, px_last, limit_px, long, horizon) if limit_px else None
                exs = extra.get(sym, {})
                fl = volume_flow(cdf, live)
                bo = breakout_signal(cdf, n_break, vol_mult, live)
                cs = crash_score(cdf, live)
                ex_live = {"Breakout": bo, "Crash": cs[0] if cs else None, "OI": exs.get("OI"),
                           "_dz": fl[2] if fl else None, "_imb": lv["imbalance_w"] if lv else None,
                           "_as": exs.get("_as")}
                cv, parts = conviction(ex_live, long, "MATCH" if sym in now else "WATCH", regime)
                blocked = blocked_tps(lv, entry, tps, long)
                qty = (equity * risk_pct / 100) / risk
                if is_stocks:
                    qty = float(int(qty))
                pos_txt = f"{qty:,.0f} shares" if is_stocks else f"{qty:,.4f} {sym[:-4]}"

                left, right = st.columns([3, 1.1])
                fig = build_chart(cdf, lf, ls, entry, sl, tps, mc, ctf, long)
                if lv:
                    add_walls(fig, lv)
                if sug and limit_px is None:
                    fig.add_hline(y=sug["entry"], line_dash="dot", line_color="#38bdf8",
                                  annotation_text="wall limit", annotation_position="bottom right")
                left.plotly_chart(fig, width="stretch")
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
                    st.metric("Net EV after costs", f"{ev_net:+.2f}R", f"costs {cost_r:.2f}R", delta_color="off")
                    st.metric("Setup grade", f"{grade(cv)} · {cv}/100",
                              help=" · ".join(f"{k} {v:+.0f}" for k, v in parts.items()))
                    if fillp is not None:
                        st.metric("Limit fill chance", f"{fillp * 100:.0f}%",
                                  f"EV per signal {fillp * ev_net:+.2f}R", delta_color="off")
                    for i, w in blocked:
                        st.caption(f"⚠️ TP{i} sits behind a {_usd(w['usd'])} "
                                   f"{'ask' if long else 'bid'} wall @ {fmt(w['price'])}")
                    st.metric("Median / 5–95% range",
                              f"{fmt(mc['median'])}", f"{fmt(mc['p5'])} – {fmt(mc['p95'])}", delta_color="off")
                    st.metric("Position size", pos_txt,
                              f"≈ {qty * entry:,.0f} {ccy} · risk {equity * risk_pct / 100:,.2f}", delta_color="off")
                    if lv:
                        st.metric("Book imbalance (near price)", f"{lv['imbalance']:+.0%}")
                        sug = lv["long_entry"] if long else lv["short_entry"]
                        if sug:
                            inv = sug.get("invalid_below") or sug.get("invalid_above")
                            st.metric("Wall-based limit entry", fmt(sug["entry"]),
                                      f"{_usd(sug['wall']['usd'])} wall · invalid beyond {fmt(inv)}",
                                      delta_color="off")
                    elif is_stocks:
                        nodes = volume_nodes(cdf)
                        st.metric("Volume POC (support/resist.)", fmt(nodes[0][0]))
                    st.metric(f"Breakout ({ctf})", bo or "n/a")
                    st.metric(f"Crash risk ({ctf})", f"{cs[0]}/100" if cs else "n/a",
                              help=cs[1] if cs else None)
                    if exs.get("OI"):
                        st.metric(f"Open interest ({sig_tf})", exs["OI"])
                    if exs.get("Analyst"):
                        st.metric("Analyst rating", exs["Analyst"])
                st.caption("Monte Carlo = bootstrap of the last ≤500 log-returns on the chosen timeframe; "
                           "hits are checked on closes. A statistical estimate, not a prediction or financial advice.")
                if is_stocks:
                    st.caption("Stocks: regular-session bars only; 4H bars are built from 1H "
                               "(09:30–13:30 and 13:30–16:00 ET); times shown in UTC. "
                               "Stops/gaps overnight are not modelled.")
                if not is_stocks:
                    with st.expander("🦶 Footprint: volume at price (latest trades)"):
                        res_fp = footprint(data_src, sym)
                        if res_fp:
                            fp, poc, cvd = res_fp
                            st.plotly_chart(footprint_chart(fp, poc, entry), width="stretch")
                            st.caption(f"Net aggressive delta: {cvd:+,.2f} · POC {fmt(poc)} · "
                                       "short sample (latest trades only)")
                        else:
                            st.info("No recent trade data available for this symbol.")
            except Exception as e:
                st.error(f"Trade planner error: {e}")
