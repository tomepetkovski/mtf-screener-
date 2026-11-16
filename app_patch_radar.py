# ═══════════════════════════════════════════════════════════════════════════
# PATCH for MTF EMA Screener Pro  -  adds the "🧲 Pre-breakout radar" tab
# Put zone_engine.py next to app.py. Three small edits:
# ═══════════════════════════════════════════════════════════════════════════

# ── EDIT 1: next to the other optional imports (top of app.py) ──────────────
try:
    import zone_engine as ze        # pre-breakout radar (optional)
except Exception:
    ze = None

# ── EDIT 2: replace the st.tabs(...) line with this (adds t8) ───────────────
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["✅ MATCH", "👀 WATCH", "All", "🎯 Trade planner", "📒 Journal",
                                           "🧪 Backtest", "🤖 AI model", "🧲 Pre-breakout radar"])

# ── EDIT 3: append at the very END of app.py ────────────────────────────────
def _radar_chart(dfc, z, bars=170):
    import plotly.graph_objects as go
    d = dfc.iloc[-bars:]
    fig = go.Figure(go.Candlestick(x=d["t"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                   name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
    x0, x1 = d["t"].iloc[0], d["t"].iloc[-1]
    if z.get("demand"):
        fig.add_hrect(y0=z["demand"]["lo"], y1=z["demand"]["hi"], fillcolor="rgba(59,130,246,.20)", line_width=0,
                      annotation_text="demand zone", annotation_position="bottom left")
    lines = [("PIVOT", z["pivot"], "#f59e0b", "dash"), ("ENTRY", z["entry"], "#3b82f6", "solid"),
             ("SL", z["sl"], "#ef4444", "solid"), ("POC", z["poc"], "#a855f7", "dot"),
             ("VAH", z["vah"], "#94a3b8", "dot"), ("VAL", z["val"], "#94a3b8", "dot"),
             ("AVWAP swing", z["avwap_swing"], "#38bdf8", "dash"), ("AVWAP climax", z["avwap_climax"], "#22d3ee", "dash")]
    lines += [(f"TP{i+1} ({z['tp_src'][i]})", t, "#22c55e", "dash") for i, t in enumerate(z["tps"])]
    lines += [("HVN", h, "#eab308", "dot") for h in z["hvn_ahead"]]
    for nm, y, col, dash in lines:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=col, width=1.2, dash=dash))
        fig.add_annotation(x=x1, y=y, text=f"{nm} {fmt(y)}", showarrow=False, xanchor="left", font=dict(color=col, size=10))
    fig.update_layout(height=620, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=150, t=30, b=10), title=f"{z['symbol']} · {z['side']} · {z['phase']} · radar {z['radar']}")
    return fig


with t8:
    if ze is None:
        st.error("zone_engine.py not found next to this app.")
    else:
        st.subheader("🧲 Pre-breakout radar")
        st.caption("Independent of the EMA rules: scans every symbol of the universe above on DAILY candles for coils "
                   "(squeeze, contraction, volume dry-up) that sit under a pivot, inside an uptrend (or a downtrend for shorts), "
                   "with accumulation, relative-strength lead and room before the next obstacle. "
                   "COILED = set-up, buy-stop above the pivot · TRIGGER = fresh break on volume · FORMING = not tight yet.")
        q1, q2, q3, q4 = st.columns(4)
        r_long = q1.selectbox("Side", ["Long", "Short"], index=0 if long_bias else 1, key="rd_side") == "Long"
        r_min = q2.slider("Min coil score", 30, 90, 50, 5, key="rd_min")
        r_dv = q3.number_input("Min avg daily $ volume", 0, 10**11, 1_000_000 if is_stocks else 100_000, 100_000, key="rd_dv")
        r_ph = q4.multiselect("Phases", ["COILED", "TRIGGER", "FORMING"], default=["COILED", "TRIGGER"], key="rd_ph")
        if st.button("🧲 Run radar on the whole universe", key="rd_run"):
            bdf = get_candles(data_src, "SPY" if is_stocks else "BTCUSDT", "1d")
            bench = None if bdf is None else bdf["close"].tolist()
            with st.spinner(f"Reading daily candles of {len(df)} symbols (first run is slow, later runs use the cache)..."):
                st.session_state["radar"] = ze.scan_universe(
                    list(df["Symbol"]), lambda s: get_candles(data_src, s, "1d"), r_long, bench, r_min, r_dv,
                    tuple(r_ph), workers=12 if is_stocks else 24)
        res = st.session_state.get("radar")
        if res is not None:
            if not res:
                st.info("Nothing coiled right now with these filters. That is normal: real coils are rare.")
            else:
                tbl = pd.DataFrame([{
                    "Symbol": z["symbol"], "Phase": z["phase"], "Grade": z["grade"], "Radar": z["radar"], "Coil": z["coil"],
                    "Accum": z["accum"], "RS 60d %": z["rs60"], "RS lead": "✅" if z["rs_lead"] else "",
                    "To pivot (ATR)": z["dist_atr"], "Rvol": z["rvol"], "Entry": z["entry"], "SL": z["sl"], "Risk %": z["risk_pct"],
                    "TP1": z["tps"][0], "TP2": z["tps"][1], "TP3": z["tps"][2], "Room (R)": z["room_r"],
                    "Notes": " · ".join(z["notes"])} for z in res])
                st.dataframe(tbl, hide_index=True, width="content", column_config={
                    "Radar": st.column_config.ProgressColumn("Radar", min_value=0, max_value=100, format="%d"),
                    "Coil": st.column_config.ProgressColumn("Coil", min_value=0, max_value=100, format="%d"),
                    "Accum": st.column_config.ProgressColumn("Accum", min_value=0, max_value=100, format="%d")})
                st.download_button("⬇️ CSV", tbl.to_csv(index=False), f"radar_{market}.csv", key="rd_dl")
                pick = st.selectbox("Inspect zones", [z["symbol"] for z in res], key="rd_pick")
                z = next(x for x in res if x["symbol"] == pick)
                cdf_ = get_candles(data_src, pick, "1d")
                if cdf_ is not None:
                    st.plotly_chart(_radar_chart(cdf_, z), width="stretch")
                    st.caption("Plan = buy-stop 0.1 ATR beyond the pivot, stop under the base (or the demand zone), "
                               "targets at the next HVN / swing level / measured move. Skip if Room < 1.5R.")

        st.markdown("---")
        st.markdown("**🧪 Does the coil score really add edge?** (same universe, daily candles, entry = buy-stop break of the pivot; "
                    "baseline = every bar of an uptrend with the same stop / targets)")
        b1, b2 = st.columns(2)
        bt_n = b1.slider("Symbols", 20, 400, 120, 20, key="rdbt_n")
        bt_cost = b2.number_input("Round-trip cost (bps)", 0.0, 200.0, 15.0, 1.0, key="rdbt_cost")
        if st.button("Run radar backtest", key="rdbt_run"):
            syms_ = list(df["Symbol"])
            random.Random(11).shuffle(syms_)
            with st.spinner("Backtesting..."):
                st.session_state["radar_bt"] = ze.backtest(syms_[:bt_n], lambda s: get_candles(data_src, s, "1d"), r_long, bt_cost)
        tbt = st.session_state.get("radar_bt")
        if tbt is not None:
            sm = ze.summarize(tbt)
            if not sm:
                st.info("No trades (crypto feeds return < 300 daily bars: use stocks for this test).")
            else:
                st.markdown("**COILED vs baseline** (COILED must beat baseline, otherwise the score is decoration)")
                st.dataframe(sm["by_kind"], width="content")
                st.markdown("**By coil score**")
                st.dataframe(sm["by_coil"], width="content")
                st.markdown("**By accumulation**")
                st.dataframe(sm["by_accum"], width="content")
                st.caption("Survivorship bias: the universe is today's liquid symbols, so absolute numbers are optimistic. "
                           "Compare COILED with baseline, not with zero. Below ~30 trades per row it is noise.")
