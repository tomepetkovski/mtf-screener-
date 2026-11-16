"""
zone_engine.py  -  "Pre-breakout radar" for MTF EMA Screener Pro
=================================================================
Finds coiled, accumulated setups BEFORE the move, maps the key zones around them and
builds a zone-based trade plan. Daily candles only (1 request per symbol), so it can run
over the WHOLE scanned universe, not only over the EMA MATCH/WATCH rows.

Ingredients (all causal, no look-ahead, all ATR / range normalised so crypto and stocks are comparable):
  COIL       squeeze (Bollinger inside Keltner), bandwidth percentile, range contraction, volume dry-up,
             tight 20-bar box, progressive contractions (VCP), higher lows, NR7, proximity to the pivot
  ACCUM      up/down volume ratio (50 bars) + OBV making a new high while price has not yet
  RS         return vs BTC / SPY and "RS line at a new high before price" (leader tell)
  ZONES      volume profile (POC / value area / HVN / LVN), anchored VWAPs (swing low, volume climax),
             unmitigated demand zone (last base before an impulse), swing highs/lows (stop pools)
  ROOM       R-multiple to the first obstacle ahead (HVN / swing level): no room = no trade
  PHASE      COILED (set-up, buy-stop above pivot) | TRIGGER (fresh break out of a coil) |
             FORMING | EXTENDED (do not chase)
Short side = the same logic on a mirrored price series.

Public API
  analyze(df, long=True, bench=None)            -> dict (live read of one symbol, incl. zones + plan)
  scan_universe(symbols, candle_fn, ...)        -> list[dict] rows for a DataFrame
  backtest(symbols, candle_fn, ...), summarize(t) -> does the coil score really add edge vs. a baseline?
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

_LOCK = threading.Lock()
_DC: dict = {}


def cached(key, ttl, fn):
    now = time.time()
    with _LOCK:
        hit = _DC.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    v = fn()
    if v is not None:
        with _LOCK:
            if len(_DC) > 12000:
                for k in sorted(_DC, key=lambda k: _DC[k][0])[:3000]:
                    _DC.pop(k, None)
            _DC[key] = (now, v)
    return v


def _clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


# ── mirror trick: shorts are longs on K - price ───────────────────────────────
def prep(df, long=True):
    """Returns (d, back). d is the series the long-logic runs on; back() maps a price back."""
    if long:
        return df.reset_index(drop=True), (lambda x: x)
    K = float(2 * df["high"].max())
    m = df.reset_index(drop=True).copy()
    m["open"], m["close"] = K - df["open"].values, K - df["close"].values
    m["high"], m["low"] = K - df["low"].values, K - df["high"].values
    return m, (lambda x: K - x)


def _tr(d):
    pc = d["close"].shift()
    return pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)


def _rank_last(s, win):
    a = s.values.astype(float)
    out = np.full(len(a), np.nan)
    if len(a) >= win:
        w = np.lib.stride_tricks.sliding_window_view(a, win)
        out[win - 1:] = (w[:, :-1] <= w[:, -1:]).mean(axis=1)
    out[~np.isfinite(a)] = np.nan
    return pd.Series(out, index=s.index)


# ── 1. features for EVERY bar (live = last row, backtest = all rows) ──────────
def coil_features(d):
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    tr = _tr(d)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr20 = tr.rolling(20).mean()
    std = c.rolling(20).std()
    sq = (2 * std < 1.5 * atr20)
    sq_len = sq.astype(int).groupby((~sq).cumsum()).cumsum()
    bbw_rank = _rank_last(4 * std / atr, 120)
    rc = tr.rolling(10).mean() / tr.rolling(50).mean()
    rng = h - l
    nr7 = (rng <= rng.rolling(7).min()).rolling(5).max().astype(bool)
    vr = v.rolling(5).mean() / v.rolling(50).mean()
    tight = (h.rolling(20).max() - l.rolling(20).min()) / atr
    r10 = h.rolling(10).max() - l.rolling(10).min()
    prog = ((r10 < r10.shift(10)).astype(int) + (r10.shift(10) < r10.shift(20)).astype(int)) / 2
    ll10 = l.rolling(10).min()
    hl = ll10 > ll10.shift(10)
    pivot = h.rolling(40).max().shift(1)
    dist = (pivot - c) / atr

    coil = (15 * sq + 5 * _clip(sq_len / 8) + 15 * _clip((0.5 - bbw_rank) / 0.4) + 15 * _clip((1 - rc) / 0.4)
            + 10 * _clip((1 - vr) / 0.4) + 10 * _clip((6 - tight) / 3) + 10 * prog + 5 * hl + 5 * nr7
            + 15 * _clip(1 - dist / 2.5) * (dist >= 0))
    coil = coil.clip(upper=100)

    up = c.diff()
    uv = (v * (up > 0)).rolling(50).sum()
    dv = (v * (up < 0)).rolling(50).sum()
    udr = uv / dv.replace(0, np.nan)
    obv = (np.sign(up).fillna(0) * v).cumsum()
    obv_lead = (obv >= obv.rolling(60).max()) & (c < h.rolling(60).max() - 0.25 * atr)
    accum = 60 * _clip((udr - 0.9) / 0.7) + 40 * obv_lead

    e50, e200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    hh, ll = h.rolling(252, min_periods=120).max(), l.rolling(252, min_periods=120).min()
    pos = (c - ll) / (hh - ll).replace(0, np.nan)
    trend_ok = (c > e50) & (e50 > e200) & (e200 > e200.shift(20)) & (pos >= 0.70)
    ext = (c - e50) / atr
    rvol = v / v.rolling(20).mean().shift(1)
    broke = c > pivot
    recent = coil.shift(1).rolling(5).max()

    phase = np.select(
        [trend_ok & broke & (rvol >= 1.3) & (recent >= 55) & (ext <= 2.5),
         trend_ok & broke & (ext > 3),
         trend_ok & ~broke & (coil >= 60) & (dist <= 2.0),
         trend_ok & ~broke & (coil >= 45) & (dist <= 4.0)],
        ["TRIGGER", "EXTENDED", "COILED", "FORMING"], default="")
    return pd.DataFrame({"atr": atr, "coil": coil, "accum": accum, "dist": dist, "pivot": pivot, "ll10": ll10,
                         "ext": ext, "rvol": rvol, "trend_ok": trend_ok, "phase": phase, "udr": udr,
                         "sq_len": sq_len, "bbw_rank": bbw_rank, "vr": vr, "tight": tight})


# ── 2. zones (live only) ──────────────────────────────────────────────────────
def volume_profile(d, win=120, bins=48):
    x = d.iloc[-win:]
    lo, hi = float(x["low"].min()), float(x["high"].max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(bins)
    for h_, l_, v_ in zip(x["high"].values, x["low"].values, x["volume"].values):
        i0 = int(np.clip(np.searchsorted(edges, l_, "right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, h_, "right") - 1, 0, bins - 1))
        hist[i0:i1 + 1] += v_ / (i1 - i0 + 1)
    sm = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    poc = int(sm.argmax())
    tot, acc, a, b = sm.sum(), sm[poc], poc, poc
    while acc < 0.7 * tot and (a > 0 or b < bins - 1):
        up_ = sm[b + 1] if b < bins - 1 else -1
        dn_ = sm[a - 1] if a > 0 else -1
        if up_ >= dn_:
            b += 1; acc += up_
        else:
            a -= 1; acc += dn_
    m = sm.mean()
    hvn = [float(mid[i]) for i in range(1, bins - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] >= 1.25 * m]
    lvn = [float(mid[i]) for i in range(1, bins - 1) if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] <= 0.6 * m]
    return {"poc": float(mid[poc]), "vah": float(mid[b]), "val": float(mid[a]), "hvn": hvn, "lvn": lvn}


def avwap(d, i0):
    x = d.iloc[i0:]
    tp, v = (x["high"] + x["low"] + x["close"]) / 3, x["volume"]
    return float(((tp * v).cumsum() / v.cumsum().replace(0, np.nan)).iloc[-1])


def demand_zone(d, atr, lookback=120):
    """Most recent UNMITIGATED base before an impulse (body >= 1.5 ATR, volume spike, breaks 10-bar high)."""
    c, o, h, l, v = (d[k].values for k in ("close", "open", "high", "low", "volume"))
    n = len(d)
    vavg = pd.Series(v).rolling(20).mean().shift(1).values
    hmax = pd.Series(h).rolling(10).max().shift(1).values
    for i in range(n - 2, max(n - lookback, 12), -1):
        if (c[i] - o[i]) >= 1.5 * atr and v[i] >= 1.3 * vavg[i] and c[i] > hmax[i]:
            j = i - 1
            while j >= i - 5 and c[j] >= o[j]:
                j -= 1
            j = j if j >= i - 5 else i - 1
            zlo, zhi = float(l[j]), float(max(o[j], c[j]))
            if (c[i + 1:] < zlo).any() or zhi >= c[-1]:
                continue
            return {"lo": zlo, "hi": zhi, "age": n - 1 - i, "retested": bool((l[i + 1:] <= zhi).any())}
    return None


def swings(d, k=3, lookback=160):
    h, l = d["high"].values, d["low"].values
    n = len(d)
    hi_, lo_ = [], []
    for i in range(max(k, n - lookback), n - k):
        if h[i] == h[i - k:i + k + 1].max():
            hi_.append(float(h[i]))
        if l[i] == l[i - k:i + k + 1].min():
            lo_.append(float(l[i]))
    return hi_, lo_


# ── 3. one symbol: live read + zone plan ──────────────────────────────────────
def _rs(c, b, lb=60):
    n = min(len(c), len(b))
    if n < lb + 5:
        return None, False
    c, b = np.asarray(c[-n:], float), np.asarray(b[-n:], float)
    line = c / b
    ret = (c[-1] / c[-1 - lb] - 1) - (b[-1] / b[-1 - lb] - 1)
    lead = bool(line[-1] >= line[-lb:].max() * 0.999 and c[-1] < c[-lb:].max() * 0.995)
    return float(ret), lead


def grade(s):
    return "A" if s >= 80 else "B" if s >= 65 else "C" if s >= 50 else "D"


def analyze(df, long=True, bench=None, feats=None):
    """df: daily candles [t, open, high, low, close, volume]. bench: list/array of benchmark daily closes."""
    if df is None or len(df) < 220:
        return None
    d, back = prep(df, long)
    f = feats if feats is not None else coil_features(d)
    r = f.iloc[-1]
    a, px = float(r["atr"]), float(d["close"].iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    vp = volume_profile(d)
    if vp is None:
        return None
    n = len(d)
    low_i = int(d["low"].iloc[-120:].values.argmin()) + n - min(120, n)
    vol_i = int(d["volume"].iloc[-120:].values.argmax()) + n - min(120, n)
    av_low, av_vol = avwap(d, low_i), avwap(d, vol_i)
    dz = demand_zone(d, a)
    sh, sl_ = swings(d)
    ahead_sw = sorted(x for x in sh if x > px + 0.25 * a)
    behind_sw = sorted((x for x in sl_ if x < px - 0.25 * a), reverse=True)
    hvn_ahead = sorted(x for x in vp["hvn"] if x > px + 0.25 * a)
    lvn_ahead = sorted(x for x in vp["lvn"] if x > px)
    pivot = float(r["pivot"]) if np.isfinite(r["pivot"]) else px
    phase = str(r["phase"])

    # entry / stop / targets (in long-space)
    entry = px if phase in ("TRIGGER", "EXTENDED") else pivot + 0.1 * a
    base_low = float(r["ll10"]) if np.isfinite(r["ll10"]) else px - 2 * a
    stop = base_low - 0.25 * a
    if dz and dz["lo"] < base_low and entry - dz["lo"] <= 3.5 * a:
        stop = min(stop, dz["lo"] - 0.25 * a)                 # a demand zone just under the base = better invalidation
    stop = float(np.clip(stop, entry - 3.5 * a, entry - 1.0 * a))
    risk = entry - stop
    base_hi_lo = float(d["low"].iloc[-40:].min())
    measured = pivot + max(pivot - base_hi_lo, 2 * a)          # measured move of the base
    obst = sorted(set(round(x, 8) for x in hvn_ahead + ahead_sw if x > entry + 0.5 * a))
    room_r = min(9.0, (obst[0] - entry) / risk) if obst else 9.0
    tps, labels = [], []
    for lvl, nm in [(x, "HVN/swing") for x in obst] + [(measured, "measured move")]:
        if lvl >= entry + 1.0 * risk and all(abs(lvl - t) > 0.5 * a for t in tps):
            tps.append(float(lvl)); labels.append(nm)
        if len(tps) == 3:
            break
    for m_ in (1.5, 2.5, 4.0):                                 # fill missing targets with R multiples
        if len(tps) < 3:
            t_ = entry + m_ * risk
            if all(abs(t_ - t) > 0.5 * a for t in tps):
                tps.append(float(t_)); labels.append(f"{m_:g}R")
    tps = sorted(tps)[:3]

    rs, rs_lead = None, False
    if bench is not None:
        r_, lead_ = _rs(df["close"].values, bench, 60)
        if long:
            rs, rs_lead = r_, lead_
        elif r_ is not None:                 # short: underperforming the benchmark is the good direction
            rs = -r_
    rs_score = 50.0 if rs is None else 100 * float(_clip(0.5 + rs / 0.4))
    rs_score = min(100.0, rs_score + (25 if rs_lead else 0))
    room_score = 100 * float(_clip((room_r - 0.5) / 2.5))
    struct = 50 * (px > av_low) + 25 * (px > vp["poc"]) + 25 * (px > av_vol)
    radar = 0.35 * float(r["coil"]) + 0.15 * float(r["accum"]) + 0.15 * rs_score + 0.20 * room_score + 0.15 * struct
    if phase in ("", "EXTENDED"):
        radar *= 0.4

    notes = []
    if r["sq_len"] >= 5: notes.append(f"squeeze {int(r['sq_len'])} bars")
    if r["bbw_rank"] <= 0.15: notes.append("bandwidth in lowest 15%")
    if r["vr"] <= 0.75: notes.append("volume dry-up")
    if r["accum"] >= 60: notes.append("accumulation (up-vol > down-vol / OBV leads)")
    if rs_lead: notes.append("RS line at new high before price")
    if room_r < 1.5: notes.append(f"only {room_r:.1f}R of room before an obstacle")
    if dz and not dz["retested"]: notes.append("fresh untested demand zone below")
    if phase == "EXTENDED": notes.append("extended: wait for a pullback to the zone")
    if phase == "TRIGGER": notes.append(f"broke the pivot on {r['rvol']:.1f}x volume")

    B = back
    zs = lambda lst: [float(B(x)) for x in lst]
    return {
        "side": "Long" if long else "Short", "phase": phase, "radar": round(float(radar), 1), "grade": grade(radar),
        "coil": round(float(r["coil"]), 1), "accum": round(float(r["accum"]), 1),
        "rs60": None if rs is None else round(100 * rs, 1), "rs_lead": rs_lead,
        "dist_atr": round(float(r["dist"]), 2), "ext_atr": round(float(r["ext"]), 2), "rvol": round(float(r["rvol"]), 2),
        "price": float(B(px)), "atr": a, "pivot": float(B(pivot)),
        "entry": float(B(entry)), "sl": float(B(stop)), "tps": [float(B(t)) for t in tps], "tp_src": labels,
        "room_r": round(room_r, 2), "risk_pct": round(100 * risk / abs(B(entry)), 2),
        "poc": float(B(vp["poc"])), "vah": float(B(vp["vah"])), "val": float(B(vp["val"])),
        "hvn_ahead": zs(hvn_ahead[:3]), "lvn_ahead": zs(lvn_ahead[:3]),
        "avwap_swing": float(B(av_low)), "avwap_climax": float(B(av_vol)),
        "demand": None if not dz else {"lo": float(min(B(dz["lo"]), B(dz["hi"]))), "hi": float(max(B(dz["lo"]), B(dz["hi"]))),
                                       "age": dz["age"], "retested": dz["retested"]},
        "swing_ahead": zs(ahead_sw[:3]), "swing_behind": zs(behind_sw[:3]),
        "notes": notes,
    }


# ── 4. whole-market scan ──────────────────────────────────────────────────────
def scan_universe(symbols, candle_fn, long=True, bench=None, min_coil=45, min_dollar_vol=0.0,
                  phases=("COILED", "TRIGGER", "FORMING"), workers=16, ttl=900, progress=None):
    """candle_fn(symbol) -> daily DataFrame. Candles are cached for `ttl` s, so re-scans are cheap.
    Cheap vectorised prefilter first; zones / plan only for the survivors."""
    rows = []

    def one(s):
        try:
            df = cached(("d1", s), ttl, lambda: candle_fn(s))
            if df is None or len(df) < 220:
                return None
            if min_dollar_vol and float((df["close"] * df["volume"]).iloc[-20:].mean()) < min_dollar_vol:
                return None
            d, _ = prep(df, long)
            f = coil_features(d)
            r = f.iloc[-1]
            if r["phase"] not in phases or r["coil"] < min_coil and r["phase"] != "TRIGGER":
                return None
            res = analyze(df, long, bench, feats=f)
            if res:
                res["symbol"] = s
            return res
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, symbols):
            if r:
                rows.append(r)
    return sorted(rows, key=lambda x: -x["radar"])


# ── 5. backtest: does the coil score add edge over simply buying uptrends? ────
def _sim(h, l, c, j, entry, sl, rr, timeout):
    risk = entry - sl
    end = min(len(h), j + timeout)
    if risk <= 0 or end <= j:
        return None
    tps = [entry + risk * x for x in rr]
    hit, state = 0, None
    for k in range(j, end):
        if l[k] <= sl:
            state = "SL"; break
        while hit < 3 and h[k] >= tps[hit]:
            hit += 1
        if hit == 3:
            state = "TP3"; break
    if state is None:
        if end - j < timeout:
            return None
        rest = max(-1.0, (c[end - 1] - entry) / risk)
    else:
        rest = -1.0 if state == "SL" else 0.0
    return sum(rr[:hit]) / 3 + (3 - hit) * rest / 3


def backtest_symbol(df, long=True, rr=(1.0, 2.0, 3.0), timeout=30, wait=10, cooldown=10):
    if df is None or len(df) < 300:
        return []
    d, back = prep(df, long)
    f = coil_features(d)
    h, l, o, c = (d[k].values for k in ("high", "low", "open", "close"))
    t = df["t"].values
    out = []
    for kind, mask in (("COILED", (f["phase"] == "COILED").values), ("baseline", f["trend_ok"].values)):
        last = -10 ** 9
        for i in range(250, len(d) - 2):
            if not mask[i] or i - last < cooldown:
                continue
            a, piv = float(f["atr"].iat[i]), float(f["pivot"].iat[i])
            if not np.isfinite(a) or a <= 0:
                continue
            if kind == "COILED":                                  # buy-stop above the pivot, wait up to `wait` bars
                trig, j = piv + 0.1 * a, None
                for k in range(i + 1, min(i + 1 + wait, len(d) - 1)):
                    if h[k] >= trig:
                        j, entry = k, max(o[k], trig)
                        break
                if j is None:
                    continue
            else:
                j, entry = i + 1, float(o[i + 1])
            sl = float(f["ll10"].iat[i]) - 0.25 * a
            sl = float(np.clip(sl, entry - 3.5 * a, entry - 1.0 * a))
            r = _sim(h, l, c, j, entry, sl, rr, timeout)
            if r is None:
                continue
            last = i
            risk_pct = (entry - sl) / abs(back(entry))
            out.append({"kind": kind, "date": t[i], "r": r, "risk_pct": risk_pct,
                        "coil": float(f["coil"].iat[i]), "accum": float(f["accum"].iat[i]),
                        "dist": float(f["dist"].iat[i])})
    return out


def backtest(symbols, candle_fn, long=True, cost_bps=15.0, **kw):
    rows = []

    def one(s):
        try:
            return [dict(x, symbol=s) for x in backtest_symbol(cached(("d1", s), 900, lambda: candle_fn(s)), long, **kw)]
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one, symbols):
            rows += r
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    t["net"] = t["r"] - cost_bps / 1e4 / t["risk_pct"]
    return t


def _wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    dd = 1 + z * z / n
    cc = (p + z * z / (2 * n)) / dd
    hh = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return max(0.0, cc - hh), min(1.0, cc + hh)


def _row(g):
    n = len(g)
    w = int((g["net"] > 0).sum())
    lo, hi = _wilson(w, n)
    return pd.Series({"n": n, "Win %": round(100 * w / n, 1) if n else np.nan, "95% CI": f"{lo * 100:.0f}-{hi * 100:.0f}%",
                      "Avg R (net)": round(g["net"].mean(), 3) if n else np.nan, "Total R": round(g["net"].sum(), 1)})


def summarize(t):
    """{'by_kind': ..., 'by_coil': ..., 'by_accum': ...}  -> the table to read is by_kind: COILED must beat baseline."""
    if t is None or t.empty:
        return {}
    c = t[t["kind"] == "COILED"].copy()
    c["coil bucket"] = pd.cut(c["coil"], [0, 65, 75, 85, 101], labels=["60-65", "65-75", "75-85", "85+"], right=False)
    c["accum bucket"] = pd.cut(c["accum"], [-1, 30, 60, 101], labels=["low", "mid", "high"])
    return {"by_kind": t.groupby("kind").apply(_row),
            "by_coil": c.groupby("coil bucket", observed=True).apply(_row),
            "by_accum": c.groupby("accum bucket", observed=True).apply(_row)}
