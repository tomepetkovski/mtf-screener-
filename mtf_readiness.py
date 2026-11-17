"""
mtf_readiness.py  -  pure helpers for MTF EMA Screener Pro  [readiness-patch]

 * scan_symbol(...)      drop-in replacement of the app's scan_symbol:
                          - same MATCH / WATCH logic and columns
                          - only the EMAs a rule really needs are required (young coins are no longer dropped)
                          - per-timeframe candle mode: live=True / False / "mixed" (1D+1W last CLOSED, rest live)
                          - Readiness 0-100 + "Flip" text (what is closest to flipping, in ATR units)
                          - optional mirrored (opposite direction) readiness in the same pass
 * lanes_for(...)        Bias / Setup / Trigger lanes (-3..+3 per timeframe, arrows = turning) + HTF bias
 * readiness_table(...)  near-miss table for both directions
 * position_qty(...)     risk-based position size (fixes the old cash-as-quantity bug)
 * tv_list(...)          TradingView watchlist string
"""
import numpy as np
import pandas as pd

TF_MIN = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}
TF_W = {"15m": 1.0, "1h": 1.5, "4h": 2.0, "1d": 3.0, "1w": 3.0}
TF_ORDER = ["1w", "1d", "4h", "1h", "15m"]
HTF = ("1d", "1w")
LOOKBACK = 5
MIRROR = {"Above": "Below", "Below": "Above", "Any": "Any", "50>200": "50<200", "50<200": "50>200"}


def mirror_rule(rule):
    return tuple(MIRROR[x] for x in rule)


def idx_for(tf, mode):
    """Index of the bar used for a timeframe. mode: True = live, False = closed, 'mixed' = HTF closed, rest live."""
    if isinstance(mode, str):
        return -2 if tf in HTF else -1
    return -1 if mode else -2


# ── indicators ─────────────────────────────────────────────────
def ema_aligned(vals, n):
    x = np.asarray(vals, dtype=float)
    out = np.full(len(x), np.nan)
    if n < 1 or len(x) < n:
        return out
    k = 2.0 / (n + 1)
    e = float(x[:n].mean())
    out[n - 1] = e
    for i in range(n, len(x)):
        e = x[i] * k + e * (1 - k)
        out[i] = e
    return out


def _g(a, i):
    if i < 0 or i >= len(a):
        return float("nan")
    v = float(a[i])
    return v if np.isfinite(v) else float("nan")


def snapshot(closes, lf, ls, idx):
    arr = np.asarray(closes, dtype=float)
    end = len(arr) + idx + 1
    if end < 2 or not np.isfinite(arr[end - 1]):
        return None
    ef, es = ema_aligned(arr, lf), ema_aligned(arr, ls)
    i, j = end - 1, end - 1 - LOOKBACK
    seg = arr[max(0, end - 15):end]
    atrc = float(np.mean(np.abs(np.diff(seg)))) if len(seg) >= 3 else float("nan")
    p = float(arr[i])
    if not (np.isfinite(atrc) and atrc > 0):
        atrc = abs(p) * 0.002 or 1e-9
    return {"p": p, "atrc": atrc, "e50": _g(ef, i), "e200": _g(es, i), "p_prev": _g(arr, j),
            "e50_prev": _g(ef, j), "e200_prev": _g(es, j), "ef": ef, "es": es, "arr": arr, "end": end}


def evaluate(rule, p, e50, e200, max_dist):
    """Same as the app's evaluate(), NaN-safe for EMAs that a rule does not use."""
    m50, m200, mx = rule
    ok50 = m50 == "Any" or (p > e50 if m50 == "Above" else p < e50)
    ok200 = m200 == "Any" or (p > e200 if m200 == "Above" else p < e200)
    okx = mx == "Any" or (e50 > e200 if mx == "50>200" else e50 < e200)
    d50 = (p / e50 - 1) * 100 if e50 > 0 else float("nan")
    d200 = (p / e200 - 1) * 100 if e200 > 0 else float("nan")
    dok50 = max_dist == 0 or m50 == "Any" or abs(d50) <= max_dist
    dok200 = max_dist == 0 or m200 == "Any" or abs(d200) <= max_dist
    return bool(ok50 and ok200 and okx and dok50 and dok200), d50, d200


def cond_list(rule, s):
    """[(label, margin_ATR, margin_ATR_5_bars_ago)] for every condition the rule uses. margin < 0 = failing."""
    m50, m200, mx = rule
    a, out = s["atrc"], []
    if m50 != "Any":
        g = 1 if m50 == "Above" else -1
        out.append(("P/50", g * (s["p"] - s["e50"]) / a, g * (s["p_prev"] - s["e50_prev"]) / a))
    if m200 != "Any":
        g = 1 if m200 == "Above" else -1
        out.append(("P/200", g * (s["p"] - s["e200"]) / a, g * (s["p_prev"] - s["e200_prev"]) / a))
    if mx != "Any":
        g = 1 if mx == "50>200" else -1
        out.append(("50x200", g * (s["e50"] - s["e200"]) / a, g * (s["e50_prev"] - s["e200_prev"]) / a))
    return out


def cond_score(m, improving):
    if m >= 0:
        return 1.0
    prox = max(0.0, 1.0 - (-m) / 3.0)          # 0 when 3 ATR or more away
    return prox * (1.0 if improving else 0.5)


def eval_side(rule, s, max_dist, near_atr):
    ok, d50, d200 = evaluate(rule, s["p"], s["e50"], s["e200"], max_dist)
    conds = cond_list(rule, s)
    fails = [(lab, -m, bool(m > mp)) for lab, m, mp in conds if m < 0]
    if ok:
        score = 1.0
    elif conds:
        score = float(np.mean([cond_score(m, bool(m > mp)) for _, m, mp in conds]))
        if not fails:
            score = min(score, 0.5)             # fails only the max-distance filter (too extended)
    else:
        score = 0.0
    near = bool(fails) and all(f[1] <= near_atr for f in fails)
    return {"ok": ok, "d50": d50, "d200": d200, "score": score, "fails": fails, "near": near}


def _flip_text(items):
    items = sorted(items, key=lambda x: x[2])[:2]
    return " · ".join(f"{tf} {lab} {d:.1f}ATR{'↑' if imp else '↓'}" for tf, lab, d, imp in items)


# ── scan (drop-in for the app's scan_symbol) ───────────────────
def scan_symbol(sym, closes_fn, rules, lf, ls, max_dist, live, fast):
    opts = fast if isinstance(fast, tuple) else (fast, False, 1.5)
    fast_mode, mirror, near_atr = bool(opts[0]), bool(opts[1]), float(opts[2])
    active = [(tf, r) for tf, r in rules if not all(x == "Any" for x in r)]
    if not active:
        return None
    total = len(active)
    wsum = sum(TF_W[tf] for tf, _ in active)
    sides = {"A": {"passed": 0, "fails": 0, "far": False, "w": 0.0, "flips": []}}
    if mirror:
        sides["B"] = {"passed": 0, "fails": 0, "far": False, "w": 0.0, "flips": []}
    res, low_tf, low_closes = {}, None, None
    for tf, rule in sorted(active, key=lambda x: TF_MIN[x[0]], reverse=True):
        closes = closes_fn(sym, tf)
        if closes is None or len(closes) < 3:
            return None
        s = snapshot(closes, lf, ls, idx_for(tf, live))
        if s is None:
            return None
        need_f = rule[0] != "Any" or rule[2] != "Any"
        need_s = rule[1] != "Any" or rule[2] != "Any"
        if (need_f and not (np.isfinite(s["e50"]) and s["e50"] > 0)) or \
           (need_s and not (np.isfinite(s["e200"]) and s["e200"] > 0)):
            return None
        for key, r in (("A", rule), ("B", mirror_rule(rule))):
            if key not in sides:
                continue
            e = eval_side(r, s, max_dist, near_atr)
            sd = sides[key]
            sd["passed"] += e["ok"]
            sd["fails"] += (not e["ok"])
            sd["w"] += TF_W[tf] * e["score"]
            sd["flips"] += [(tf, lab, d, imp) for lab, d, imp in e["fails"]]
            if not e["ok"] and not e["near"]:
                sd["far"] = True
            if key == "A":
                res[tf] = (e["ok"], s["p"], e["d50"], e["d200"])
        if low_tf is None or TF_MIN[tf] < TF_MIN[low_tf]:
            low_tf, low_closes = tf, closes
        if fast_mode and all(sd["fails"] >= 2 and (sd["far"] or sd["fails"] >= 3) for sd in sides.values()):
            break
    A = sides["A"]
    live_flag = bool(live)                         # True and "mixed" are both live
    price = float(low_closes[-1] if live_flag else low_closes[-2]) if low_closes is not None and len(low_closes) > 2 else 0.0
    row = {"Symbol": sym, "Price": price}
    for tf, _ in active:
        if tf not in res:
            continue
        ok, p, d50, d200 = res[tf]
        row[tf] = "✅" if ok else "❌"
        row[f"{tf} %50"] = round(d50, 2) if np.isfinite(d50) else np.nan
        row[f"{tf} %200"] = round(d200, 2) if np.isfinite(d200) else np.nan
    row["Passed"] = f"{A['passed']}/{total}"
    row["Status"] = "MATCH" if A["passed"] == total else ("WATCH" if total >= 2 and A["passed"] == total - 1 else "-")
    row["_score"] = A["passed"] / total
    row["Ready"] = int(round(100 * A["w"] / wsum))
    row["Flip"] = "✅ all rules met" if A["passed"] == total else _flip_text(A["flips"])
    if "B" in sides:
        B = sides["B"]
        row["Ready ⇄"] = int(round(100 * B["w"] / wsum))
        row["Flip ⇄"] = "✅ all rules met" if B["passed"] == total else _flip_text(B["flips"])
    return row


# ── lanes: Bias / Setup / Trigger ──────────────────────────────
def lane_state(closes, lf, ls, idx):
    """(score now, score LOOKBACK bars ago), each -3..+3, from P vs EMA50, P vs EMA200, EMA50 vs EMA200."""
    s = snapshot(closes, lf, ls, idx)
    if s is None:
        return None

    def score(p, e50, e200):
        t = []
        if np.isfinite(e50):
            t.append(1 if p > e50 else -1)
        if np.isfinite(e200):
            t.append(1 if p > e200 else -1)
        if np.isfinite(e50) and np.isfinite(e200):
            t.append(1 if e50 > e200 else -1)
        return None if not t else int(round(sum(t) * 3 / len(t)))

    now = score(s["p"], s["e50"], s["e200"])
    prev = score(s["p_prev"], s["e50_prev"], s["e200_prev"]) if np.isfinite(s["p_prev"]) else None
    return None if now is None else (now, prev if prev is not None else now)


def lanes_for(sym, closes_fn, lf, ls, live):
    """(text, HTF bias) e.g. ('1W +3 · 1D +3 · 4H +1↑ · 1H -2↑ · 15m +3', 'Long')."""
    parts, bias = [], []
    for tf in TF_ORDER:
        try:
            c = closes_fn(sym, tf)
            st = lane_state(c, lf, ls, idx_for(tf, live)) if c is not None and len(c) > 3 else None
        except Exception:
            st = None
        if not st:
            continue
        now, prev = st
        arrow = "↑" if now - prev >= 1 else "↓" if now - prev <= -1 else ""
        lab = tf.upper() if tf != "15m" else "15m"
        parts.append(f"{lab} {now:+d}{arrow}")
        if tf in HTF:
            bias.append(now)
    if not parts:
        return "n/a", "?"
    b = float(np.mean(bias)) if bias else 0.0
    return " · ".join(parts), ("Long" if b >= 1.5 else "Short" if b <= -1.5 else "Mixed")


def bias_check(side, htf_bias):
    if htf_bias in ("?", "Mixed"):
        return "➖ HTF mixed"
    return "✅ with HTF bias" if side == htf_bias else "⚠️ counter-trend"


def readiness_table(df, long_bias, min_ready=55, side_sel="Both", max_rows=30):
    cols = ["Symbol", "Side", "Ready", "Status", "Flip", "Price"]
    A, B = ("Long", "Short") if long_bias else ("Short", "Long")
    parts = []
    if "Ready" in df.columns and side_sel in ("Both", A):
        a = df[df["Status"] != "MATCH"]
        parts.append(pd.DataFrame({"Symbol": a["Symbol"], "Side": A, "Ready": a["Ready"],
                                   "Status": a["Status"].replace("-", "near"), "Flip": a["Flip"], "Price": a["Price"]}))
    if "Ready ⇄" in df.columns and side_sel in ("Both", B):
        b = df
        parts.append(pd.DataFrame({"Symbol": b["Symbol"], "Side": B, "Ready": b["Ready ⇄"],
                                   "Status": np.where(b["Ready ⇄"] >= 100, "MATCH", "near"),
                                   "Flip": b["Flip ⇄"], "Price": b["Price"]}))
    if not parts:
        return pd.DataFrame(columns=cols)
    t = pd.concat(parts, ignore_index=True)
    t = t[pd.to_numeric(t["Ready"], errors="coerce") >= min_ready]
    return t.sort_values(["Ready", "Symbol"], ascending=[False, True]).head(max_rows).reset_index(drop=True)[cols]


# ── sizing ─────────────────────────────────────────────────────
def edge_scale(r_results, min_n=30):
    """0.5 when the journal shows a non-positive average result over >= min_n closed signals, else 1."""
    r = [float(x) for x in r_results if x is not None and np.isfinite(x)]
    if len(r) < min_n:
        return 1.0
    return 0.5 if float(np.mean(r)) <= 0 else 1.0


def position_qty(equity, risk_pct, risk, entry=None, conviction=50, ai_p=None, regime_state=None,
                 edge=1.0, long=True, max_notional_x=1.0):
    """Quantity = (equity * risk% * scales) / risk-per-unit, capped by notional (spot: 1x equity).
    ai_p is a probability 0..1. The regime scale only cuts size when the trade is AGAINST the regime."""
    if not (risk and risk > 0 and np.isfinite(risk)):
        return 0.0
    against = "RISK-OFF" if long else "RISK-ON"
    r_scale = 0.5 if regime_state == against else 0.75 if regime_state == "CAUTION" else 1.0
    c = 1.0 if conviction >= 80 else 0.85 if conviction >= 65 else 0.65 if conviction >= 50 else \
        0.4 if conviction >= 35 else 0.25
    if ai_p is not None and np.isfinite(ai_p):
        a = min(1.0, max(0.3, (float(ai_p) - 0.35) / 0.30))
        c = 0.6 * c + 0.4 * a
    qty = equity * (risk_pct / 100.0) * r_scale * c * edge / risk
    if entry and entry > 0:
        qty = min(qty, max_notional_x * equity / entry)
    return float(max(qty, 0.0))


# ── TradingView list ───────────────────────────────────────────
def tv_list(symbols, market, src):
    pre = {"binance": "BINANCE:", "bybit": "BYBIT:", "okx": "OKX:"}.get(src, "")
    if market == "stocks":
        return ",".join(str(s).replace("-", ".") for s in symbols)
    return ",".join(f"{pre}{s}" for s in symbols)
