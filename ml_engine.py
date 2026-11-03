"""
ml_engine.py — AI layer for "MTF EMA Screener Pro"
===================================================
Predicts P(win), expected R and a TAKE / NEUTRAL / SKIP call for MATCH and WATCH setups, trained on
the rule-based signals only (MATCH, and optionally WATCH; manual plans and everything else are excluded):

  * live  : closed MATCH (+ WATCH) rows of your signal journal (manual "taken" plans excluded);
            the conviction component 'MTF' (20 = MATCH, 8 = WATCH) tells the model which kind it is
  * hist  : optional pre-training set = historical MATCH events produced by the Backtest tab
            (same rules, same exit logic, price features only, down-weighted)

Design (small, noisy, non-stationary data => the main job is NOT to fool yourself):
  1. Features are direction-signed, scale-free (ATR units / percentiles / z-scores) and taken on the
     last CLOSED bar, so live, journal and backtest rows are computed identically.
  2. Model = ridge logistic regression (numpy) + shallow gradient boosting (scikit-learn, optional),
     probabilities averaged, then Platt-calibrated on out-of-fold predictions.
  3. Validation = purged expanding-window walk-forward: a fold is only trained on trades that were
     already CLOSED before the fold starts (no look-ahead, no overlapping-trade leakage).
  4. Gate: the model is used only if OOF AUC (bootstrap 95% one-sided) > 0.5 AND calibrated Brier beats
     the base-rate forecast. Otherwise status = LEARNING and the app shows nothing actionable.
  5. Shrinkage toward the base rate by n/(n+k): small samples can never give extreme probabilities.
  6. Post-training live check: signals that closed AFTER the last training are true out-of-sample.

The model file is a local pickle written by this app: never load a pickle from an untrusted source.
"""
import json
import os
import pickle
import threading
import time

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    HAVE_SK = True
except Exception:                                   # scikit-learn is optional
    GradientBoostingClassifier = None
    ExtraTreesClassifier = None
    HAVE_SK = False

MODEL_PATH = os.environ.get("AI_MODEL_PATH", "ai_model.pkl")
HIST_PATH = os.environ.get("AI_HIST_PATH", "ai_hist.pkl")
MODEL_VERSION = 4
_TRAIN_LOCK = threading.Lock()
_MODEL_CACHE = {}

DEFAULT_CFG = {
    "min_n": 80,            # min closed signals in the evaluation pool before the gate can open
    "min_live_eval": 30,    # live closed signals needed to evaluate on live data (else historical)
    "l2": 20.0,             # ridge strength of the logistic model
    "hist_w": 0.35,         # sample weight of historical backtest rows relative to live rows
    "halflife_d": 730.0,    # recency half-life in days
    "folds": 5,
    "use_gb": True,
    "cost_bps": 15.0,       # round-trip cost used to turn R into net R for the labels
    "shrink_k": 150.0,      # shrinkage: p = prior + n/(n+k) * (p_model - prior)
    "hist_cap": 0.5,        # max shrink weight when only historically validated
    "include_watch": True,  # train on WATCH signals too (False = MATCH only)
    "take_ev": 0.10,        # AI call = TAKE needs net EV >= this many R (and P(win) above the base rate)
}

PRICE_COLS = ["ext", "rsi", "mom5", "mom20", "slope", "dist_slow", "dz",
              "adx", "atr_pct", "bbw_pct", "vol_z", "pullback", "room",
              "ret1", "ret5", "ret20", "ret_z", "range_atr", "body_atr",
              "close_loc", "wick_reject", "ema_gap"]
META_COLS = ["is_long", "is_stocks", "tf_log", "sl_atr"]
PART_KEYS = ["MTF", "Flow", "Book", "OI", "Break", "Crash", "Regime", "Analyst", "Crowd",
             "RelStr", "Earnings", "Event", "Extend", "Trend"]      # conviction components (AI is never one)
ALL_COLS = PRICE_COLS + META_COLS + ["p_" + k for k in PART_KEYS] + ["is_live"]
TF_MIN = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}

NICE = {
    "ext": "Extension from fast EMA (ATR)", "rsi": "RSI (with trade)", "mom5": "5-bar momentum (ATR)",
    "mom20": "20-bar momentum (ATR)", "slope": "Fast-EMA slope", "dist_slow": "Distance to slow EMA",
    "dz": "Volume delta z (with trade)", "adx": "Trend strength (ADX)", "atr_pct": "Volatility (log ATR %)",
    "bbw_pct": "Squeeze percentile", "vol_z": "Volume z", "pullback": "Pullback depth (ATR)",
    "room": "Room to 20-bar extreme (ATR)", "ret1": "1-bar return (ATR, with trade)",
    "ret5": "5-bar return (ATR, with trade)", "ret20": "20-bar return (ATR, with trade)",
    "ret_z": "Return anomaly Z-score (with trade)", "range_atr": "Bar range (ATR)",
    "body_atr": "Candle body (ATR, with trade)", "close_loc": "Close location (with trade)",
    "wick_reject": "Directional rejection wick (ATR)", "ema_gap": "Fast/slow EMA gap (ATR, with trade)",
    "is_long": "Long (vs short)", "is_stocks": "Stock (vs crypto)",
    "tf_log": "Timeframe", "sl_atr": "Stop size (ATR)", "is_live": "Live (vs backtest) row",
}
for _k in PART_KEYS:
    NICE["p_" + _k] = f"Conviction: {_k}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Features
# ══════════════════════════════════════════════════════════════════════════════
def raw_feature_frame(df, lf=50, ls=200):
    """Direction-agnostic features for EVERY bar of a candle frame (cols t,open,high,low,close,volume).
    Vectorised: used for live (row -2) and for every historical signal bar (no per-signal recompute)."""
    c, h, l, v = (df[k].astype(float) for k in ("close", "high", "low", "volume"))
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    ef = c.ewm(span=lf, adjust=False, min_periods=lf).mean()
    es = c.ewm(span=ls, adjust=False, min_periods=ls).mean()

    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()

    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls_ = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + g / ls_.replace(0, np.nan))
    rsi = rsi.where(ls_ > 0, 100.0)

    bw = c.rolling(20).std() * 4 / c.rolling(20).mean()
    bbw_pct = bw.rolling(120, min_periods=40).apply(lambda w: float((w[:-1] <= w[-1]).mean()), raw=True)
    lv = np.log1p(v)
    m50, s50 = lv.rolling(50).mean().shift(1), lv.rolling(50).std().shift(1)
    rng = (h - l).replace(0, np.nan)
    clv = ((2 * c - h - l) / rng).fillna(0.0)
    d3 = (clv * v).rolling(3).sum()
    dm, ds = d3.rolling(50).mean().shift(1), d3.rolling(50).std().shift(1)

    f = pd.DataFrame(index=df.index)
    f["ext"] = (c - ef) / a
    f["rsi"] = (rsi - 50) / 50
    f["mom5"] = (c - c.shift(5)) / a
    f["mom20"] = (c - c.shift(20)) / a
    f["slope"] = (ef - ef.shift(10)) / a / 10
    f["dist_slow"] = (c / es - 1) * 10                      # in units of 10 %
    f["dz"] = (d3 - dm) / ds.replace(0, np.nan)
    f["adx"] = adx / 100
    f["atr_pct"] = np.log((a / c * 100).clip(lower=1e-6))
    f["bbw_pct"] = bbw_pct
    f["vol_z"] = (lv - m50) / s50.replace(0, np.nan)
    # Scale-free additions; reference return statistics exclude the current bar.
    ret = c.pct_change()
    ret_mu = ret.rolling(50, min_periods=25).mean().shift(1)
    ret_sd = ret.rolling(50, min_periods=25).std().shift(1)
    op = df["open"].astype(float)
    f["ret1"] = c.diff() / a
    f["ret5"] = (c - c.shift(5)) / a
    f["ret20"] = (c - c.shift(20)) / a
    f["ret_z"] = (ret - ret_mu) / ret_sd.replace(0, np.nan)
    f["range_atr"] = (h - l) / a
    f["body_atr"] = (c - op) / a
    f["close_loc"] = clv
    lower_wick = (np.minimum(op, c) - l).clip(lower=0)
    upper_wick = (h - np.maximum(op, c)).clip(lower=0)
    f["lower_wick_atr"] = lower_wick / a
    f["upper_wick_atr"] = upper_wick / a
    f["ema_gap"] = (ef - es) / a
    f["dd60"] = (h.rolling(60).max() - c) / a
    f["ru60"] = (c - l.rolling(60).min()) / a
    f["room_up"] = (h.rolling(20).max().shift(1) - c) / a
    f["room_dn"] = (c - l.rolling(20).min().shift(1)) / a
    return f.replace([np.inf, -np.inf], np.nan)


def _cl(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        return None
    return None if not np.isfinite(x) else float(min(max(x, lo), hi))


def signed_features(row, long, meta=None):
    """One row of raw_feature_frame -> final feature dict (direction-signed, clipped). None = missing."""
    meta = meta or {}
    sg = 1.0 if long else -1.0
    g = lambda k: row.get(k) if hasattr(row, "get") else row[k]
    sgn = lambda k, lo, hi: (lambda v: None if v is None else sg * v)(_cl(g(k), lo, hi))
    out = {
        "ext": sgn("ext", -6, 6), "rsi": sgn("rsi", -1, 1), "mom5": sgn("mom5", -6, 6),
        "mom20": sgn("mom20", -12, 12), "slope": sgn("slope", -1, 1),
        "dist_slow": sgn("dist_slow", -4, 4), "dz": sgn("dz", -5, 5),
        "adx": _cl(g("adx"), 0, 1), "atr_pct": _cl(g("atr_pct"), -8, 6),
        "bbw_pct": _cl(g("bbw_pct"), 0, 1), "vol_z": _cl(g("vol_z"), -5, 5),
        "pullback": _cl(g("dd60") if long else g("ru60"), 0, 15),
        "room": _cl(g("room_up") if long else g("room_dn"), -10, 10),
        "ret1": sgn("ret1", -8, 8), "ret5": sgn("ret5", -8, 8),
        "ret20": sgn("ret20", -12, 12), "ret_z": sgn("ret_z", -6, 6),
        "range_atr": _cl(g("range_atr"), 0, 10),
        "body_atr": sgn("body_atr", -8, 8),
        "close_loc": sgn("close_loc", -1, 1),
        "wick_reject": _cl(g("lower_wick_atr") if long else g("upper_wick_atr"), 0, 8),
        "ema_gap": sgn("ema_gap", -12, 12),
        "is_long": 1.0 if long else 0.0,
        "is_stocks": 1.0 if meta.get("stocks") else 0.0,
        "tf_log": float(np.log2(TF_MIN.get(meta.get("tf", "1d"), 1440) / 15)),
        "sl_atr": float(meta.get("sl_atr", 1.5)),
    }
    return out


def live_features(df, long, lf=50, ls=200, meta=None):
    """Features of the last CLOSED bar (row -2; the last row is the still-forming candle)."""
    try:
        if df is None or len(df) < max(ls, 60) + 10:
            return None
        row = raw_feature_frame(df, lf, ls).iloc[-2]
        out = signed_features(row, long, meta)
        return out if out.get("ext") is not None else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Datasets (MATCH only)
# ══════════════════════════════════════════════════════════════════════════════
def _loadj(s):
    try:
        d = json.loads(s) if isinstance(s, str) and s else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def journal_frame(jdf, cost_bps=15.0, include_watch=True):
    """Closed MATCH (+ optionally WATCH) signals of the journal; manual 'taken' plans are excluded."""
    if jdf is None or len(jdf) == 0:
        return pd.DataFrame()
    j = jdf.copy()
    for c in ("feat", "parts", "extras", "status", "outcome", "closed_ts"):
        if c not in j.columns:
            j[c] = None
    ok_st = ["MATCH", "WATCH"] if include_watch else ["MATCH"]
    m = j["status"].isin(ok_st) & (j["outcome"].astype(str) != "OPEN") & j["r_result"].notna()
    m &= ~j["extras"].astype(str).str.contains('"taken"')
    j = j[m]
    if j.empty:
        return pd.DataFrame()
    risk = (j["entry"] - j["sl"]).abs().replace(0, np.nan)
    net = j["r_result"].astype(float) - cost_bps / 1e4 * j["entry"] / risk
    rows = []
    for ft, pt in zip(j["feat"], j["parts"]):
        r = {k: v for k, v in _loadj(ft).items() if k in PRICE_COLS + META_COLS}
        for k, v in _loadj(pt).items():
            if k in PART_KEYS:
                r["p_" + k] = v
        rows.append(r)
    X = pd.DataFrame(rows, index=j.index)
    head = pd.DataFrame({"src": "live", "status": j["status"].astype(str), "t_open": j["ts"].astype(float),
                         "t_close": pd.to_numeric(j["closed_ts"], errors="coerce").fillna(j["ts"]).astype(float),
                         "net": net, "y": (net > 0).astype(int)})
    out = pd.concat([head, X], axis=1).dropna(subset=["net"])
    out["has_feat"] = [bool(_loadj(s)) for s in j.loc[out.index, "feat"]]
    return out


def hist_available(path=HIST_PATH):
    return os.path.exists(path)


def load_hist(path=HIST_PATH):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def hist_info(path=HIST_PATH):
    h = load_hist(path)
    if not h:
        return None
    return {"n": len(h["df"]), "meta": h.get("meta", {}), "ts": h.get("ts")}


def save_hist(t, meta, path=HIST_PATH):
    """t = trades DataFrame from run_backtest (needs columns date, t_end, net, feat). Returns row count."""
    tt = t[t["feat"].notna()] if "feat" in t.columns else t.iloc[0:0]
    if tt.empty:
        return 0
    feats = pd.DataFrame(list(tt["feat"]), index=tt.index)
    head = pd.DataFrame({"src": "hist",
                         "t_open": [pd.Timestamp(d).timestamp() for d in tt["date"]],
                         "t_close": [pd.Timestamp(d).timestamp() for d in tt["t_end"]],
                         "net": tt["net"].astype(float), "y": (tt["net"] > 0).astype(int)})
    df = pd.concat([head, feats], axis=1).dropna(subset=["net"]).reset_index(drop=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"df": df, "meta": dict(meta or {}), "ts": time.time()}, fh)
    os.replace(tmp, path)
    return len(df)


def assemble(jdf, hist, cfg):
    live = journal_frame(jdf, cfg["cost_bps"], cfg["include_watch"])
    parts = [live] if len(live) else []
    if hist is not None and len(hist.get("df", [])):
        parts.append(hist["df"])
    if not parts:
        return pd.DataFrame()
    D = pd.concat(parts, ignore_index=True)
    for c in ALL_COLS:
        if c not in D.columns:
            D[c] = np.nan
    D["is_live"] = (D["src"] == "live").astype(float)
    D["status"] = D["status"].fillna("MATCH") if "status" in D.columns else "MATCH"
    D.loc[D["src"] == "hist", "p_MTF"] = 20.0                   # backtest events are MATCH by construction
    for c in ALL_COLS:
        D[c] = pd.to_numeric(D[c], errors="coerce")
    return D.sort_values("t_open").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Models
# ══════════════════════════════════════════════════════════════════════════════
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _fit_logit(X, y, w, l2, iters=60):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = w / max(w.mean(), 1e-12)
    p0 = float(np.clip((w * y).sum() / w.sum(), 0.02, 0.98))
    beta = np.zeros(d + 1)
    beta[0] = np.log(p0 / (1 - p0))
    pen = l2 * np.eye(d + 1)
    pen[0, 0] = 0.0
    for _ in range(iters):
        p = _sig(Xb @ beta)
        grad = Xb.T @ (w * (p - y)) + pen @ beta
        hess = Xb.T @ (Xb * (w * p * (1 - p))[:, None]) + pen + 1e-6 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def _weights(df, cfg, t_ref):
    base = np.where(df["src"].values == "live", 1.0, cfg["hist_w"])
    age_d = np.clip((t_ref - df["t_open"].values) / 86400.0, 0, None)
    w = base * np.clip(0.5 ** (age_d / cfg["halflife_d"]), 1e-3, 1.0)
    return w / max(w.mean(), 1e-12)


def _imputer(df, cols):
    imp = {}
    for c in cols:
        if c.startswith("p_") or c == "is_live":
            imp[c] = 0.0
        else:
            s = df[c].dropna()
            imp[c] = float(s.median()) if len(s) else 0.0
    return imp


def _matrix(df, cols, imp):
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    return X.fillna(pd.Series(imp)).values.astype(float)


def _scaler(X, w, cols):
    mu, sd = np.zeros(len(cols)), np.ones(len(cols))
    for j, c in enumerate(cols):
        if c.startswith("p_"):
            sd[j] = 10.0                                   # fixed: conviction points, 0 = neutral
        else:
            m = float(np.average(X[:, j], weights=w))
            s = float(np.sqrt(np.average((X[:, j] - m) ** 2, weights=w)))
            mu[j], sd[j] = m, (s if s > 1e-9 else 1.0)
    return mu, sd


def _fit_models(tr, cols, cfg):
    y = tr["y"].values.astype(float)
    if len(tr) < 25 or y.min() == y.max():
        return None
    w = _weights(tr, cfg, float(tr["t_open"].max()))
    imp = _imputer(tr, cols)
    X = _matrix(tr, cols, imp)
    mu, sd = _scaler(X, w, cols)
    Xs = np.clip((X - mu) / sd, -5, 5)
    beta = _fit_logit(Xs, y, w, cfg["l2"])
    gb, et = None, None
    if cfg["use_gb"] and HAVE_SK and len(tr) >= 60:
        try:
            gb = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.04, max_depth=2, subsample=0.75,
                min_samples_leaf=max(8, int(0.03 * len(tr))), max_features=0.8,
                random_state=7
            )
            gb.fit(Xs, y, sample_weight=w)
        except Exception:
            gb = None
        # Diverse tree family; conservative leaves reduce overfit on noisy labels.
        try:
            et = ExtraTreesClassifier(
                n_estimators=160, min_samples_leaf=max(5, int(0.04 * len(tr))),
                max_features=0.75, class_weight="balanced", random_state=17, n_jobs=1
            )
            et.fit(Xs, y, sample_weight=w)
        except Exception:
            et = None
    return {"cols": cols, "imp": imp, "mu": mu, "sd": sd, "beta": beta, "gb": gb, "et": et}


def _predict_raw(m, df):
    X = _matrix(df, m["cols"], m["imp"])
    Xs = np.clip((X - m["mu"]) / m["sd"], -5, 5)
    p_lr = _sig(m["beta"][0] + Xs @ m["beta"][1:])
    preds, weights, tree_preds = [p_lr], [0.30], []
    if m.get("gb") is not None:
        pg = m["gb"].predict_proba(Xs)[:, 1]
        preds.append(pg); weights.append(0.35); tree_preds.append(pg)
    if m.get("et") is not None:
        pe = m["et"].predict_proba(Xs)[:, 1]
        preds.append(pe); weights.append(0.35); tree_preds.append(pe)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    p = sum(w * pred for w, pred in zip(weights, preds))
    # Preserve the existing return contract: third item is the non-linear consensus.
    p_tree = np.mean(tree_preds, axis=0) if tree_preds else None
    return p, p_lr, p_tree, Xs


# ══════════════════════════════════════════════════════════════════════════════
# 4. Metrics + walk-forward
# ══════════════════════════════════════════════════════════════════════════════
def auc_score(y, p):
    y, p = np.asarray(y), np.asarray(p, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot_auc(y, p, n=300, seed=3):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = auc_score(y[i], p[i])
        if np.isfinite(a):
            vals.append(a)
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95))) if len(vals) > 20 else (float("nan"),) * 2


def walk_forward(D, cols, cfg):
    """Purged expanding-window walk-forward on the evaluation pool. Returns (oof DataFrame, scope)."""
    n_live = int((D["src"] == "live").sum())
    scope = "live" if n_live >= cfg["min_live_eval"] else "historical"
    E = D[D["src"] == ("live" if scope == "live" else "hist")].sort_values("t_open")
    K = int(cfg["folds"])
    if len(E) < max(20, K * 4):
        return pd.DataFrame(), scope
    out = []
    for k, blk in enumerate(np.array_split(np.arange(len(E)), K)):
        if k == 0 or len(blk) == 0:
            continue
        te = E.iloc[blk]
        tr = D[D["t_close"] < te["t_open"].min()]               # purge: only trades already closed
        m = _fit_models(tr, cols, cfg)
        if m is None:
            continue
        p, p_lr, p_gb, _ = _predict_raw(m, te)
        out.append(pd.DataFrame({"p": p, "base": float(tr["y"].mean()), "y": te["y"].values,
                                 "net": te["net"].values, "t": te["t_open"].values,
                                 "status": te["status"].values}))
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), scope


def _platt(p, y):
    z = _logit(p)
    b = _fit_logit(z.reshape(-1, 1), y.astype(float), np.ones(len(y)), l2=1.0)
    return float(b[0]), float(b[1])


def _tables(oof_p, y, net):
    order = np.argsort(-oof_p)
    th = []
    for frac in (0.2, 0.33, 0.5, 1.0):
        idx = order[: max(1, int(round(frac * len(y))))]
        th.append({"Top share": f"{int(frac * 100)}%", "n": len(idx), "Min P": round(float(oof_p[idx].min()), 3),
                   "Win %": round(100 * float(y[idx].mean()), 1), "Avg R (net)": round(float(net[idx].mean()), 3)})
    rel = []
    try:
        q = pd.qcut(pd.Series(oof_p).rank(method="first"), 5, labels=False)
        for b in range(5):
            m = (q == b).values
            rel.append({"Bucket": b + 1, "n": int(m.sum()), "Predicted %": round(100 * float(oof_p[m].mean()), 1),
                        "Actual %": round(100 * float(y[m].mean()), 1), "Avg R (net)": round(float(net[m].mean()), 3)})
    except Exception:
        pass
    return th, rel


def _by_status(oof, pc):
    rows = []
    for st_ in ("MATCH", "WATCH"):
        m = (oof["status"] == st_).values
        if m.sum() >= 5:
            y = oof["y"].values[m]
            rows.append({"Signal": st_, "n": int(m.sum()), "Win %": round(100 * float(y.mean()), 1),
                         "Avg R (net)": round(float(oof["net"].values[m].mean()), 3),
                         "Avg AI P(win) %": round(100 * float(pc[m].mean()), 1),
                         "AUC": round(auc_score(y, pc[m]), 3) if (m.sum() >= 20 and 0 < y.sum() < len(y)) else None})
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 5. Train / save / load / predict
# ══════════════════════════════════════════════════════════════════════════════
def train(jdf, hist=None, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    D = assemble(jdf, hist, cfg)
    n_live = int((D["src"] == "live").sum()) if len(D) else 0
    n_hist = int((D["src"] == "hist").sum()) if len(D) else 0
    if len(D) < 30:
        return {"status": "NO_DATA", "n_live": n_live, "n_hist": n_hist, "trained_ts": time.time(), "cfg": cfg,
                "gate": {"ok": False, "reasons": [f"only {len(D)} closed MATCH rows (need ≥ 30)"]},
                "version": MODEL_VERSION}
    cols = [c for c in ALL_COLS if D[c].notna().any() and D[c].fillna(0).nunique() > 1]
    if n_live == 0 and "is_live" in cols:
        cols.remove("is_live")
    oof, scope = walk_forward(D, cols, cfg)
    reasons, ok = [], False
    metrics = {}
    prior, a_cal, b_cal = float(D.loc[D["src"] == ("live" if scope == "live" else "hist"), "y"].mean()), 0.0, 1.0
    th, rel, by_status = [], [], []
    if len(oof) >= 20 and oof["y"].nunique() == 2:
        # Deploy calibration is fit on all OOF predictions; validation metrics use
        # expanding chronological calibration to avoid scoring an in-sample calibrator.
        y_all, p_all = oof["y"].values.astype(int), oof["p"].values.astype(float)
        net_all = oof["net"].values.astype(float)
        a_cal, b_cal = _platt(p_all, y_all)
        pc_eval = p_all.copy()
        eval_mask = np.zeros(len(oof), dtype=bool)
        calibration_mode = "raw-small-sample"
        if len(oof) >= 45:
            start_cal = max(20, len(oof) // 3)
            edges = np.linspace(start_cal, len(oof), min(5, len(oof) - start_cal + 1), dtype=int)
            for j in range(len(edges) - 1):
                lo_i, hi_i = int(edges[j]), int(edges[j + 1])
                if hi_i <= lo_i or len(np.unique(y_all[:lo_i])) < 2:
                    continue
                try:
                    aa, bb = _platt(p_all[:lo_i], y_all[:lo_i])
                    pc_eval[lo_i:hi_i] = _sig(aa + bb * _logit(p_all[lo_i:hi_i]))
                    eval_mask[lo_i:hi_i] = True
                except Exception:
                    continue
            calibration_mode = "expanding-crossfit"
        if not eval_mask.any():
            eval_mask[:] = True
            pc_eval = p_all.copy()
        y, pc, net = y_all[eval_mask], pc_eval[eval_mask], net_all[eval_mask]
        oof_eval = oof.iloc[np.where(eval_mask)[0]].reset_index(drop=True)
        lo, hi = _boot_auc(y, pc)
        brier, brier_base = float(np.mean((pc - y) ** 2)), float(np.mean((oof_eval["base"].values - y) ** 2))
        ll = float(-np.mean(y * np.log(np.clip(pc, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pc, 1e-6, 1))))
        metrics = {"auc": auc_score(y, pc), "auc_lo": lo, "auc_hi": hi, "brier": brier, "brier_base": brier_base,
                   "skill": 1 - brier / brier_base if brier_base > 0 else float("nan"), "logloss": ll,
                   "n_oof": len(y), "n_oof_total": len(oof), "calibration_mode": calibration_mode,
                   "win_all": float(y.mean())}
        th, rel = _tables(pc, y, net)
        by_status = _by_status(oof_eval, pc)
        n_eval = int((D["src"] == ("live" if scope == "live" else "hist")).sum())
        if n_eval < cfg["min_n"]:
            reasons.append(f"{n_eval} closed {scope} MATCH signals < {cfg['min_n']} required")
        if len(y) < 20:
            reasons.append(f"only {len(y)} out-of-sample evaluation predictions after calibration checks; need ≥ 20")
        if not (np.isfinite(lo) and lo > 0.5):
            reasons.append(f"AUC {metrics['auc']:.3f} (90% low {lo:.3f}) is not reliably above 0.5")
        if not (np.isfinite(metrics["skill"]) and metrics["skill"] > 0):
            reasons.append("calibrated Brier does not beat the base-rate forecast")
        ok = not reasons
    else:
        reasons.append("not enough out-of-fold predictions to validate yet")

    full = _fit_models(D, cols, cfg)
    if full is None:
        return {"status": "NO_DATA", "n_live": n_live, "n_hist": n_hist, "trained_ts": time.time(), "cfg": cfg,
                "gate": {"ok": False, "reasons": ["one-class labels"]}, "version": MODEL_VERSION}
    E = D[D["src"] == ("live" if scope == "live" else "hist")]
    wins, loss = E.loc[E["net"] > 0, "net"], E.loc[E["net"] <= 0, "net"]
    n_eval = len(E)
    shrink = n_eval / (n_eval + cfg["shrink_k"]) if ok else 0.0
    if scope != "live":
        shrink = min(shrink, cfg["hist_cap"])
    status = ("ACTIVE_LIVE" if scope == "live" else "ACTIVE_HIST") if ok else "LEARNING"

    imp_tab = []
    _tree_imps = [m.feature_importances_ for m in (full.get("gb"), full.get("et")) if m is not None]
    gbi = np.mean(_tree_imps, axis=0) if _tree_imps else None
    for j, c in enumerate(cols):
        imp_tab.append({"Feature": NICE.get(c, c), "LR coef (std)": round(float(full["beta"][j + 1]), 3),
                        "Tree importance": round(float(gbi[j]), 3) if gbi is not None else None})
    return {"version": MODEL_VERSION, "status": status, "trained_ts": time.time(), "cfg": cfg, "scope": scope,
            "model": full, "cols": cols, "calib": (a_cal, b_cal), "prior": prior, "shrink": float(shrink),
            "avg_win": float(wins.mean()) if len(wins) else 1.5,
            "avg_loss": float(max(0.2, -loss.mean())) if len(loss) else 0.9,
            "n_live": n_live, "n_hist": n_hist, "n_eval": n_eval, "metrics": metrics,
            "thresholds": th, "reliability": rel, "by_status": by_status, "importance": imp_tab,
            "gate": {"ok": ok, "reasons": reasons}, "has_gb": full["gb"] is not None, "has_et": full.get("et") is not None}


def save_model(pkg, path=MODEL_PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(pkg, fh)
    os.replace(tmp, path)
    _MODEL_CACHE.clear()


def load_model(path=MODEL_PATH):
    try:
        mt = os.path.getmtime(path)
    except Exception:
        return None
    hit = _MODEL_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        with open(path, "rb") as fh:
            pkg = pickle.load(fh)
        if pkg.get("version") != MODEL_VERSION:
            pkg = None
    except Exception:
        pkg = None
    _MODEL_CACHE[path] = (mt, pkg)
    return pkg


def train_and_save(jdf, hist=None, cfg=None):
    """Thread-safe: returns the new package, or None if another training is already running."""
    if not _TRAIN_LOCK.acquire(blocking=False):
        return None
    try:
        pkg = train(jdf, hist, cfg)
        if pkg["status"] != "NO_DATA":
            save_model(pkg)
        return pkg
    finally:
        _TRAIN_LOCK.release()


def needs_retrain(pkg, jdf, every_n, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    L = journal_frame(jdf, cfg["cost_bps"], cfg["include_watch"])
    if not pkg or pkg.get("status") == "NO_DATA":
        return len(L) >= cfg["min_live_eval"] or hist_available()
    if not every_n:
        return False
    new = int((L["t_close"] > pkg["trained_ts"]).sum()) if len(L) else 0
    return new >= every_n


def _active(pkg):
    return bool(pkg) and pkg.get("status") in ("ACTIVE_LIVE", "ACTIVE_HIST")


def predict_frame(pkg, df):
    """Final (calibrated, shrunk) P(win) for rows of a feature frame; is_live is forced to 1."""
    d = df.copy()
    d["is_live"] = 1.0
    p_raw, p_lr, p_gb, Xs = _predict_raw(pkg["model"], d)
    a, b = pkg["calib"]
    pc = _sig(a + b * _logit(p_raw))
    return pkg["prior"] + pkg["shrink"] * (pc - pkg["prior"]), p_lr, p_gb, Xs


def predict_one(pkg, feat, parts):
    """feat = live_features(...) dict, parts = conviction components dict. Returns None unless the gate is open."""
    if not _active(pkg) or not feat:
        return None
    row = {c: np.nan for c in ALL_COLS}
    row.update({k: v for k, v in feat.items() if v is not None})
    for k, v in (parts or {}).items():
        if k in PART_KEYS:
            row["p_" + k] = float(v)
    p, p_lr, p_gb, Xs = predict_frame(pkg, pd.DataFrame([row]))
    p = float(p[0])
    ev = p * pkg["avg_win"] - (1 - p) * pkg["avg_loss"]
    agree = None if p_gb is None else float(1 - min(1.0, 2 * abs(p_lr[0] - p_gb[0])))
    beta = pkg["model"]["beta"][1:]
    contrib = beta * Xs[0]
    top = np.argsort(-np.abs(contrib))[:4]
    drivers = [(NICE.get(pkg["model"]["cols"][j], pkg["model"]["cols"][j]), float(contrib[j])) for j in top]
    cfg = pkg.get("cfg", DEFAULT_CFG)
    if ev >= cfg.get("take_ev", 0.10) and p >= pkg["prior"] + 0.02 and (agree is None or agree >= 0.4):
        call = "✅ TAKE"
    elif ev < 0:
        call = "⛔ SKIP"
    else:
        call = "➖ NEUTRAL"
    return {"p": p, "ev": float(ev), "agree": agree, "drivers": drivers, "status": pkg["status"], "call": call}


def live_check(pkg, jdf):
    """Out-of-sample check on MATCH signals that closed AFTER the model was trained."""
    if not pkg or "model" not in pkg:
        return None
    L = journal_frame(jdf, pkg["cfg"]["cost_bps"], pkg["cfg"].get("include_watch", True))
    if L.empty:
        return None
    L = L[(L["t_close"] > pkg["trained_ts"]) & L["has_feat"]]
    if len(L) < 5:
        return {"n": len(L)}
    for c in ALL_COLS:
        if c not in L.columns:
            L[c] = np.nan
    p = predict_frame(pkg, L)[0]
    y = L["y"].values
    prior = pkg["prior"]
    return {"n": len(L), "pred": float(p.mean()), "actual": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2)), "brier_base": float(np.mean((prior - y) ** 2)),
            "auc": auc_score(y, p) if len(L) >= 20 else None}


# ══════════════════════════════════════════════════════════════════════════════
# 6. Streamlit tab
# ══════════════════════════════════════════════════════════════════════════════
BADGE = {"ACTIVE_LIVE": "🟢 ACTIVE · validated on your live MATCH signals",
         "ACTIVE_HIST": "🟡 ACTIVE · validated on historical backtest only (capped confidence)",
         "LEARNING": "⚪ LEARNING · gate closed, forecasts are hidden",
         "NO_DATA": "⚪ NO DATA"}


def render_tab(st, jdf, cfg, use_hist):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    st.subheader("🤖 AI model: win probability for MATCH / WATCH setups")
    st.caption("Trained only on rule-based signals (MATCH + WATCH from the journal, plus an optional historical backtest of your own rules). "
               "Walk-forward validated, calibrated, shrunk toward the base rate and switched on only when it "
               "beats the base rate out-of-sample. It is a probability estimate, not a guarantee or financial advice.")
    pkg = load_model()
    L = journal_frame(jdf, cfg["cost_bps"], cfg["include_watch"])
    hi = hist_info()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closed live signals", len(L),
              f"{int((L['status'] == 'MATCH').sum())} MATCH · {int((L['status'] == 'WATCH').sum())} WATCH" if len(L) else None,
              delta_color="off")
    c2.metric("Historical MATCH rows", hi["n"] if hi else 0)
    c3.metric("scikit-learn", "yes" if HAVE_SK else "no (logistic only)")
    c4.metric("Model", "trained" if pkg else "none")
    if len(L) and "has_feat" in L and (~L["has_feat"]).any():
        st.caption(f"ℹ️ {int((~L['has_feat']).sum())} older journal rows have no stored price features "
                   "(logged before this version): they train on conviction components only.")
    if not hi:
        st.info("Optional but recommended for a cold start: run the 🧪 Backtest tab (stocks work best, they have "
                "10y of daily bars) and press 'Use these trades to pre-train the AI model'.")

    b1, b2 = st.columns([1, 3])
    if b1.button("🔁 Retrain now"):
        with st.spinner("Walk-forward validation + final fit..."):
            new = train_and_save(jdf, load_hist() if use_hist else None, cfg)
        if new is None:
            st.warning("Another training is already running.")
        else:
            st.success(f"Done: {new['status']}")
            st.rerun()
    if pkg is None:
        st.warning("No model yet. It is created automatically once enough MATCH signals have closed "
                   "(or from a historical pre-training set), or press Retrain.")
        return
    b2.markdown(f"**{BADGE.get(pkg['status'], pkg['status'])}**  ·  trained "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(pkg['trained_ts']))} UTC  ·  "
                f"{pkg['n_live']} live + {pkg['n_hist']} historical rows"
                + ("  ·  ensemble: logistic + gradient boosting + ExtraTrees" if pkg.get("has_gb") and pkg.get("has_et") else
                   "  ·  ensemble: logistic + available tree model" if (pkg.get("has_gb") or pkg.get("has_et")) else "  ·  logistic only"))
    for r in pkg["gate"]["reasons"]:
        st.warning("Gate closed: " + r)
    m = pkg.get("metrics") or {}
    if m:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("OOF AUC", f"{m['auc']:.3f}", f"90% CI {m['auc_lo']:.2f}–{m['auc_hi']:.2f}", delta_color="off")
        k2.metric("Brier skill vs base rate", f"{m['skill'] * 100:+.1f}%")
        k3.metric("Base win rate", f"{m['win_all'] * 100:.0f}%")
        th = pkg.get("thresholds") or []
        if len(th) >= 2:
            k4.metric("Win % in AI top third", f"{th[1]['Win %']:.0f}%", f"avg {th[1]['Avg R (net)']:+.2f}R", delta_color="off")
        k5.metric("Shrink weight", f"{pkg['shrink'] * 100:.0f}%", help="Share of the model deviation from the base rate "
                                                                        "that is passed on to forecasts.")
        if pkg.get("by_status"):
            st.markdown("**By signal type** (out-of-fold: is WATCH worth trading, and does the AI separate it?)")
            st.dataframe(pd.DataFrame(pkg["by_status"]), hide_index=True, width="content")
        st.markdown("**Precision by AI rank** (out-of-fold: does a higher AI probability really win more?)")
        st.dataframe(pd.DataFrame(pkg["thresholds"]), hide_index=True, width="content")
        st.markdown("**Reliability** (predicted vs actual win rate in 5 buckets; they should track each other)")
        st.dataframe(pd.DataFrame(pkg["reliability"]), hide_index=True, width="content")
    if m.get("calibration_mode"):
        st.caption(f"Validation calibration: {m['calibration_mode']} · {m.get('n_oof', 0)} evaluated OOF predictions "
                   f"from {m.get('n_oof_total', m.get('n_oof', 0))} total walk-forward predictions.")
    st.markdown("**What the model uses** (standardised logistic coefficient; + = raises P(win); tree importance is averaged across available tree models)")
    st.dataframe(pd.DataFrame(pkg["importance"]).sort_values("LR coef (std)", key=abs, ascending=False),
                 hide_index=True, width="content")
    lc = live_check(pkg, jdf)
    if lc and lc.get("n", 0) >= 5:
        st.markdown("**Live check: signals that closed after the last training (true out-of-sample)**")
        q1, q2, q3 = st.columns(3)
        q1.metric("n", lc["n"])
        q2.metric("Predicted vs actual win %", f"{lc['pred'] * 100:.0f}% vs {lc['actual'] * 100:.0f}%")
        q3.metric("Brier (model vs base)", f"{lc['brier']:.3f} vs {lc['brier_base']:.3f}", delta_color="off")
        if lc["n"] >= 30 and lc["brier"] > lc["brier_base"]:
            st.error("The model is currently WORSE than the base rate on fresh signals: retrain or stop using it.")
    elif lc:
        st.caption(f"Live check needs ≥ 5 new closed signals after training (have {lc['n']}).")
    with st.expander("How to read this / limits"):
        st.markdown(
            "- **Rule-signal training** means the model answers one question: *given that the rules fired "
            "(MATCH, or WATCH = all but one timeframe), which signals tend to work?* It cannot find setups your "
            "rules miss. Turn off WATCH training in the sidebar to go back to MATCH only.\n"
            "- **AI call**: TAKE needs net EV ≥ 0.10R, a P(win) above the base rate and the two models roughly "
            "agreeing; SKIP means negative expected R; everything else is NEUTRAL. It is a filter, not a command.\n"
            "- A forecast is shown only if the model beats the base rate in a **purged walk-forward test**. "
            "Below ~100 closed signals expect LEARNING most of the time: that is the honest answer.\n"
            "- Markets change. Weights decay with a 2-year half-life and the live check watches for drift.\n"
            "- Historical rows come from the Backtest tab (today's survivors, price features only, down-weighted) "
            "and are never used to claim live accuracy while ≥ 30 live signals exist.\n"
            "- The AI score is **not** part of the conviction grade, so the model never trains on its own output.")
