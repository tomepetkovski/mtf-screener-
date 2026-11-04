"""
decision_engine.py: decision layer for "MTF EMA Screener Pro"
==============================================================
Turns a long list of MATCH/WATCH rows into a short list of actionable, non-redundant trades.

  1. rank_pct          : cross-sectional rank inside today's candidates (the 0-100 conviction grade saturates:
                         when 77 of 83 rows are >= 65, "grade B+" no longer separates anything)
  2. build_decisions   : TAKE / CANDIDATE / WAIT / SKIP + reasons + risk multiplier per symbol
                         (TAKE is only issued when the validated AI gate is open; otherwise CANDIDATE)
  3. select_diversified: greedy pick of top-ranked names whose RESIDUAL returns (market factor removed)
                         are not near-duplicates, so 5 picks are not one sector bet
  4. nasdaq_earnings_calendar: free earnings dates (no Finnhub key) so the earnings gap penalty is never blind
  5. portfolio_risk    : total risk budget check for the suggested picks
"""
import time
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

_LOCK = threading.Lock()
_CACHE = {}


# ── 1. cross-sectional rank ───────────────────────────────────────────────────
def rank_pct(values):
    """Percentile rank 0-100 (NaN stays NaN). values: dict or Series."""
    s = pd.Series(values, dtype="float64")
    return (s.rank(pct=True) * 100).round(0)


# ── 2. decision per symbol ────────────────────────────────────────────────────
def _ok(v):
    return v is not None and isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def risk_multiplier(pred, rank, ai_active, take_ev=0.10, macro_h=None):
    """Scale of the base risk-per-trade. AI active: proportional to validated EV (EV = take_ev -> 1.0x, capped at 1.5x).
    AI not active: never above 1.0x (nothing is validated), 0.5x for the lower half of today's candidates."""
    if ai_active and pred:
        m = float(np.clip(pred["ev"] / max(take_ev, 1e-6), 0.0, 1.5))
        m = 0.0 if pred["ev"] <= 0 else max(m, 0.25)
    else:
        m = 1.0 if (_ok(rank) and rank >= 75) else 0.75 if (_ok(rank) and rank >= 50) else 0.5
    if macro_h is not None and macro_h <= 24:
        m *= 0.5
    return round(m, 2)


def decide(status, long, rank, pred, ai_active, ex, regime_state, macro_h, take_ev=0.10):
    """Returns (decision, why, risk_x). Hard blockers -> SKIP, soft flags -> WAIT, else TAKE / CANDIDATE."""
    sg = 1 if long else -1
    hard, soft = [], []
    ext = ex.get("_ext")
    dz = ex.get("_dz")
    ed = ex.get("_ed")
    crowd = ex.get("_crowd")
    crash = ex.get("Crash")
    ext_d = sg * ext if _ok(ext) else None
    dz_d = sg * dz if _ok(dz) else None

    if _ok(ed) and ed <= 3:
        hard.append(f"earnings in {int(ed)}d (gap over stop)")
    if macro_h is not None and macro_h <= 6:
        hard.append("macro event < 6h")
    if ai_active and pred and pred["call"].endswith("SKIP"):
        hard.append("AI: negative expected R")
    if ext_d is not None and ext_d > 3:
        hard.append(f"extended {ext_d:.1f} ATR (chasing)")

    if ext_d is not None and 2 < ext_d <= 3:
        soft.append(f"{ext_d:.1f} ATR above EMA: wait for pullback")
    if dz_d is not None and dz_d <= -1.5:
        soft.append("volume flow against you")
    if regime_state and ((long and regime_state == "RISK-OFF") or ((not long) and regime_state == "RISK-ON")):
        soft.append(f"against regime ({regime_state})")
    if _ok(crowd) and ((long and crowd >= 0.5) or ((not long) and crowd <= -0.5)):
        soft.append("crowded in your direction")
    if _ok(crash) and long and crash >= 40:
        soft.append(f"crash risk {int(crash)}")
    if _ok(ed) and 3 < ed <= 7:
        soft.append(f"earnings in {int(ed)}d")
    if status == "WATCH" and not ai_active:
        soft.append("WATCH (one TF missing)")

    rx = risk_multiplier(pred, rank, ai_active, take_ev, macro_h)
    if hard:
        return "⛔ SKIP", "; ".join(hard), 0.0
    if soft:
        return "⏳ WAIT", "; ".join(soft), round(rx * 0.5, 2)
    if ai_active and pred:
        if pred["call"].endswith("TAKE"):
            return "✅ TAKE", f"AI TAKE, P {pred['p'] * 100:.0f}%, EV {pred['ev']:+.2f}R", rx
        return "➖ NEUTRAL", "AI sees no edge above threshold", round(rx * 0.5, 2)
    if _ok(rank) and rank >= 75 and status == "MATCH":
        return "🟡 CANDIDATE", f"top quartile today (rank {int(rank)}), AI gate closed: unvalidated", rx
    return "➖ NEUTRAL", "mid/low rank today", round(rx * 0.5, 2)


def build_decisions(df, extra, conv_map, ai_pred, ai_active, regime, macro_h, long_bias, take_ev=0.10):
    """df: scanner frame with Symbol/Status. Returns DataFrame indexed by Symbol: Rank %, Decision, Risk x, Why."""
    cand = df[df["Status"].isin(["MATCH", "WATCH"])]
    syms = list(cand["Symbol"])
    if not syms:
        return pd.DataFrame(columns=["Rank %", "Decision", "Risk x", "Why"])
    stat = dict(zip(cand["Symbol"], cand["Status"]))
    conv = {s: (conv_map.get(s) or (None,))[0] for s in syms}
    evs = {s: ai_pred[s]["ev"] for s in syms if s in ai_pred}
    base = evs if (ai_active and len(evs) >= 0.5 * len(syms)) else conv
    rk = rank_pct({s: v for s, v in base.items() if v is not None})
    rows = {}
    rs = regime["state"] if regime else None
    for s in syms:
        r = rk.get(s, np.nan)
        d, why, rx = decide(stat[s], long_bias, r, ai_pred.get(s), ai_active, extra.get(s, {}), rs, macro_h, take_ev)
        rows[s] = {"Rank %": r, "Decision": d, "Risk x": rx, "Why": why}
    return pd.DataFrame.from_dict(rows, orient="index")


# ── 3. diversified selection ──────────────────────────────────────────────────
def residual_corr(closes, lookback=120):
    """Correlation of daily log-returns after removing the cross-sectional mean (market factor)
    so that only sector / theme co-movement remains. closes: {sym: list/array of daily closes}."""
    R = {}
    for s, c in closes.items():
        c = np.asarray(c if c is not None else [], dtype="float64")
        if len(c) >= lookback + 1 and np.all(c[-lookback - 1:] > 0):
            R[s] = np.diff(np.log(c[-lookback - 1:]))
    if len(R) < 3:
        return pd.DataFrame(R).corr() if R else pd.DataFrame()
    M = pd.DataFrame(R)
    M = M.sub(M.mean(axis=1), axis=0)
    return M.corr()


def select_diversified(order, closes, k=8, max_corr=0.5, lookback=120):
    """order: symbols best-first. Returns (picked, skipped) where skipped = {sym: (twin, corr)}.
    Symbols without enough history are kept (cannot be proven redundant)."""
    C = residual_corr({s: closes.get(s) for s in order}, lookback)
    picked, skipped = [], {}
    for s in order:
        if len(picked) >= k:
            break
        twin = None
        if s in C.columns:
            for t in picked:
                if t in C.columns and np.isfinite(C.loc[s, t]) and C.loc[s, t] >= max_corr:
                    twin = (t, float(C.loc[s, t]))
                    break
        if twin:
            skipped[s] = twin
        else:
            picked.append(s)
    return picked, skipped


# ── 4. free earnings calendar (api.nasdaq.com, same host the app already uses) ─
def nasdaq_earnings_calendar(fetch, base, headers, clean_sym, days=7, ttl=21600):
    """{symbol: ('YYYY-MM-DD', 'bmo'|'amc'|'')} for the next `days` days, or None if every call failed.
    fetch(bases, path, params, headers=...) is the app's fetch(); base = NQ; headers = NQ_HDR; clean_sym = _clean_sym."""
    key = ("earn_nasdaq", days)
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    out, any_ok = {}, False
    today = datetime.now(timezone.utc).date()
    for i in range(days + 1):
        d = today + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        j = fetch(base, "/api/calendar/earnings", {"date": d.isoformat()}, headers=headers, retries=1)
        try:
            rows = (j or {}).get("data", {}).get("rows")
        except Exception:
            rows = None
        if rows is None:
            continue
        any_ok = True
        for r in rows:
            s = clean_sym(r.get("symbol"))
            if not s or s in out:
                continue
            t = str(r.get("time") or "")
            out[s] = (d.isoformat(), "bmo" if "pre" in t else "amc" if "after" in t else "")
    if not any_ok:
        return None
    with _LOCK:
        _CACHE[key] = (now, out)
    return out


# ── 5. portfolio risk budget ──────────────────────────────────────────────────
def portfolio_risk(picks, risk_x, base_risk_pct, max_heat_pct=6.0, open_heat=0.0):
    """Scales the suggested risks down proportionally if their sum (plus risk already open) exceeds the budget.
    Returns ({sym: risk_pct}, total_pct, scale)."""
    raw = {s: base_risk_pct * float(risk_x.get(s, 1.0)) for s in picks}
    tot = sum(raw.values())
    room = max(0.0, max_heat_pct - open_heat)
    scale = 1.0 if tot <= room or tot == 0 else room / tot
    out = {s: round(v * scale, 3) for s, v in raw.items()}
    return out, round(sum(out.values()), 3), round(scale, 3)
