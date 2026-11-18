import os
from datetime import datetime, timezone
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="The5ers Trading Co-Pilot | Phase 2",
    page_icon="📈",
    layout="wide",
)


def api_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def api_post(path: str, json_body: dict | None = None, params: dict | None = None):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, params=params, timeout=120)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def api_delete(path: str):
    try:
        resp = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def render_drawdown_gauge(title: str, used_pct: float, soft_limit: float, hard_limit: float, remaining_usd: float):
    bar_color = "#22c55e"  # Green
    if used_pct >= soft_limit:
        bar_color = "#ef4444"  # Red
    elif used_pct >= (soft_limit * 0.65):
        bar_color = "#f59e0b"  # Amber

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=used_pct,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": f"{title}<br><span style='font-size:13px;color:gray'>Buffer Left: ${remaining_usd:,.2f}</span>"},
            gauge={
                "axis": {"range": [0, hard_limit], "tickwidth": 1},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, soft_limit], "color": "rgba(34, 197, 94, 0.12)"},
                    {"range": [soft_limit, hard_limit], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": soft_limit,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=55, b=10))
    return fig


def bias_badge(bias: str) -> str:
    mapping = {
        "BULLISH": "🟢 BULLISH",
        "BEARISH": "🔴 BEARISH",
        "NEUTRAL": "⚪ NEUTRAL",
        "VOLATILE_CHOP": "🟠 VOLATILE CHOP",
    }
    return mapping.get(bias, bias)


# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚡ Co-Pilot Controls")
    st.caption("Phase 2: Interactive Journal & Macro Radar")

    lookback_days = st.slider("History Lookback (Days)", min_value=7, max_value=180, value=30, step=1)

    if st.button("🔄 Sync MetaTrader 5 Now", width="stretch", type="primary"):
        with st.spinner("Connecting to MT5 terminal & syncing closed deals..."):
            sync_data, err = api_post("/sync/mt5", params={"lookback_days": lookback_days})
            if err:
                st.error(f"Sync failed: {err}")
            else:
                st.success(f"Synced {sync_data['trades_synced']} trades!")
                st.rerun()

    if st.button("🌍 Refresh Macro & Calendar", width="stretch"):
        with st.spinner("Scanning live DXY, yields, Fed/ECB & red-folder calendar via Google Search..."):
            macro_res, m_err = api_post("/macro/refresh")
            if m_err:
                st.error(f"Macro scan failed: {m_err}")
            else:
                st.success(f"Updated regime ({macro_res['events_upserted']} events)!")
                st.rerun()

    st.divider()
    symbol_filter = st.selectbox("Filter Instrument", ["ALL", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])
    st.caption(f"Current UTC: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`")
    st.info("Active Model: `gemini-4-hydrogen`\nFirm: **The5ers High Stakes**")


# --- Header, News Blackout Banner & Live Drawdown Gauges ---
st.title("The5ers High Stakes — Multi-Agent Trading Co-Pilot")

dd_status, dd_err = api_get("/account/drawdown")
if dd_err:
    st.error(f"FastAPI Backend Offline ({API_BASE_URL}): {dd_err}")
    st.stop()

if dd_status["soft_lock_triggered"]:
    st.error(f"🚨 **DRAWDOWN CIRCUIT BREAKER TRIGGERED:** {dd_status['lock_reason']} Stop trading for the session.")

blackout_status, _ = api_get("/macro/blackout", params={"symbol": symbol_filter})
if blackout_status:
    if blackout_status["is_blackout_active"]:
        st.error(f"🛑 **{blackout_status['message']}**")
    else:
        st.success(f"🟢 **{blackout_status['message']}**")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Current Equity", f"${dd_status['current_equity']:,.2f}", f"Ref: ${dd_status['day_reference_level']:,.2f}")
col_m2.metric("Starting Balance", f"${dd_status['starting_account_balance']:,.2f}", f"Date: {dd_status['trading_date']}")
col_m3.metric("Daily Buffer Left", f"${dd_status['daily_drawdown_remaining_usd']:,.2f}", f"-{dd_status['daily_drawdown_used_pct']}% / 5.0%")
col_m4.metric("Static Buffer Left", f"${dd_status['static_drawdown_remaining_usd']:,.2f}", f"-{dd_status['static_drawdown_used_pct']}% / 10.0%")

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(
        render_drawdown_gauge(
            "Daily Loss Limit (5.0% Max / 3.5% Soft Lock)",
            dd_status["daily_drawdown_used_pct"],
            soft_limit=3.5,
            hard_limit=5.0,
            remaining_usd=dd_status["daily_drawdown_remaining_usd"],
        ),
        width="stretch",
    )
with g2:
    st.plotly_chart(
        render_drawdown_gauge(
            "Max Static Drawdown (10.0% Max / 7.5% Soft Lock)",
            dd_status["static_drawdown_used_pct"],
            soft_limit=7.5,
            hard_limit=10.0,
            remaining_usd=dd_status["static_drawdown_remaining_usd"],
        ),
        width="stretch",
    )

st.divider()

# --- Main Tabs (Phase 1 + Phase 1.5 + Phase 2) ---
tab_macro, tab_journal, tab_chat, tab_analytics, tab_audit = st.tabs(
    [
        "🌍 Macro & News Radar",
        "📓 Trade Log & AI Debrief",
        "💬 Chat With Journal",
        "📊 Edge & Session Analytics",
        "🧠 Behavioral Leak Audit",
    ]
)

# ==============================================================================
# TAB 1: PHASE 2 MACRO & NEWS RADAR
# ==============================================================================
with tab_macro:
    macro_data, macro_err = api_get("/macro/latest")
    if macro_err:
        st.error(f"Could not load macro state: {macro_err}")
    else:
        snap = macro_data.get("snapshot")
        events = macro_data.get("events", [])

        top_m1, top_m2 = st.columns([3, 1])
        with top_m1:
            st.subheader("Live Macro Regime & Central Bank Bias")
            if snap:
                st.caption(f"Last Grounded Scan: `{snap['updated_at']}`")
            else:
                st.info("No macro scan saved yet. Click **🌍 Refresh Macro & Calendar** to run a live grounded search.")
        with top_m2:
            if st.button("🔄 Run Grounded Scan Now", type="primary", width="stretch"):
                with st.spinner("Searching DXY, 10Y yields, Fed/ECB & economic calendar..."):
                    res, err = api_post("/macro/refresh")
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

        if snap:
            b1, b2, b3 = st.columns(3)
            b1.metric("Macro Regime", snap["regime"])
            b2.metric("XAU/USD (Gold) Bias", bias_badge(snap["xauusd_bias"]))
            b3.metric("EUR/USD Bias", bias_badge(snap["eurusd_bias"]))

            c_left, c_right = st.columns(2)
            with c_left:
                st.info(f"**💵 DXY & Real Yields Context:**\n\n{snap['dxy_yield_context']}")
            with c_right:
                st.info(f"**🏦 Fed & ECB Policy Tone:**\n\n{snap['central_bank_summary']}")

            st.warning(f"**🎯 Intraday Execution Guideline:** {snap['trading_guideline']}")

            if snap.get("sources"):
                with st.expander("Verified Grounding Sources"):
                    for src in snap["sources"]:
                        st.markdown(f"- [{src['title']}]({src['uri']})")

        st.divider()
        cal_col, form_col = st.columns([1.6, 1.0], gap="large")

        with cal_col:
            st.subheader("Upcoming High-Impact Economic Calendar (UTC)")
            if not events:
                st.caption("No upcoming events cached. Run a grounded scan or add a manual catalyst on the right.")
            else:
                ev_df = pd.DataFrame(events)
                ev_cols = ["id", "scheduled_utc", "currency", "impact", "event_name", "forecast", "previous", "actual"]
                st.dataframe(
                    ev_df[ev_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "id": "ID",
                        "scheduled_utc": "Time (UTC)",
                        "currency": "CCY",
                        "impact": "Impact",
                        "event_name": "Release / Catalyst",
                    },
                )

                del_col1, del_col2 = st.columns([1, 2])
                del_id = del_col1.number_input("Event ID to Remove", min_value=1, step=1, label_visibility="collapsed")
                if del_col2.button("🗑️ Delete Event ID"):
                    _, d_err = api_delete(f"/macro/events/{int(del_id)}")
                    if d_err:
                        st.error(d_err)
                    else:
                        st.rerun()

        with form_col:
            st.subheader("Pin Manual Catalyst (UTC)")
            st.caption("Ensure exact +/-30m blackout enforcement around NFP, CPI, or FOMC.")
            with st.form("manual_macro_event_form", clear_on_submit=True):
                ev_name = st.text_input("Event Name", placeholder="E.g., US Non-Farm Payrolls (NFP)")
                ev_ccy = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
                ev_impact = st.selectbox("Impact", ["HIGH", "MEDIUM"])
                default_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT13:30:00Z")
                ev_time = st.text_input("Scheduled UTC (ISO-8601)", value=default_ts)
                ev_fcst = st.text_input("Forecast (Optional)", placeholder="E.g., 180K or 0.3%")
                ev_prev = st.text_input("Previous (Optional)", placeholder="E.g., 223K or 0.2%")
                add_btn = st.form_submit_button("Pin Event to Calendar", type="primary", width="stretch")

                if add_btn:
                    if not ev_name.strip():
                        st.warning("Enter an event name.")
                    else:
                        aff = ["XAUUSD", "EURUSD"] if ev_ccy == "USD" else ["EURUSD"]
                        payload = {
                            "event_name": ev_name.strip(),
                            "currency": ev_ccy,
                            "impact": ev_impact,
                            "scheduled_utc": ev_time.strip(),
                            "forecast": ev_fcst.strip() or None,
                            "previous": ev_prev.strip() or None,
                            "affected_symbols": aff,
                        }
                        _, p_err = api_post("/macro/events", json_body=payload)
                        if p_err:
                            st.error(p_err)
                        else:
                            st.success("Event pinned!")
                            st.rerun()

# ==============================================================================
# TAB 2: PHASE 1 & 1.5 TRADE LOG & INTERACTIVE DEBRIEF
# ==============================================================================
trades_data, trades_err = api_get("/trades", params={"limit": 200, "symbol": symbol_filter})
trades_data = trades_data or []

with tab_journal:
    left_col, right_col = st.columns([1.4, 1.0], gap="large")

    with left_col:
        st.subheader("Synced MT5 Closed Trades")
        if not trades_data:
            st.warning("No closed trades found. Click **Sync MetaTrader 5 Now** in the sidebar.")
        else:
            df = pd.DataFrame(trades_data)
            display_cols = [
                "mt5_position_id",
                "symbol",
                "direction",
                "volume",
                "net_profit",
                "r_multiple",
                "session",
                "setup_type",
                "emotional_state",
                "followed_plan",
            ]
            st.dataframe(
                df[display_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "mt5_position_id": "MT5 ID",
                    "net_profit": st.column_config.NumberColumn("Net PnL ($)", format="$%.2f"),
                    "r_multiple": st.column_config.NumberColumn("R-Mult", format="%.2fR"),
                    "followed_plan": st.column_config.CheckboxColumn("Plan Followed"),
                },
            )

            df_sorted = df.sort_values("close_time")
            df_sorted["cum_pnl"] = df_sorted["net_profit"].cumsum()
            fig_eq = px.line(
                df_sorted,
                x="close_time",
                y="cum_pnl",
                markers=True,
                title="Cumulative Realized PnL ($)",
                labels={"close_time": "Close Time (UTC)", "cum_pnl": "Net Profit ($)"},
            )
            fig_eq.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_eq, width="stretch")

    with right_col:
        st.subheader("Interactive Post-Trade Debrief")
        if not trades_data:
            st.info("Sync MT5 trades on the left to start an interactive coaching debrief.")
        else:
            trade_options = {
                f"#{t['mt5_position_id']} | {t['symbol']} {t['direction']} (${t['net_profit']:+.2f})": t
                for t in trades_data
            }
            selected_label = st.selectbox("Select Trade to Debrief", list(trade_options.keys()))
            selected_trade = trade_options[selected_label]
            pos_id = selected_trade["mt5_position_id"]

            if selected_trade.get("ai_feedback"):
                st.success(
                    f"**Saved Journal Tag:** `{selected_trade['setup_type']}` | "
                    f"Session: `{selected_trade['session']}` | Emotion: `{selected_trade['emotional_state']}`\n\n"
                    f"{selected_trade['ai_feedback']}"
                )

            debrief_key = f"debrief_history_{pos_id}"
            if debrief_key not in st.session_state:
                st.session_state[debrief_key] = []

            col_b1, col_b2 = st.columns(2)
            if col_b1.button("🎙️ Start / Reset Debrief", width="stretch"):
                with st.spinner("Coach is reviewing MT5 execution telemetry..."):
                    res, err = api_post(f"/trades/{pos_id}/debrief", json_body={"conversation": []})
                    if err:
                        st.error(err)
                    else:
                        st.session_state[debrief_key] = [{"role": "model", "content": res["coach_message"]}]
                        st.rerun()

            if st.session_state[debrief_key]:
                if col_b2.button("✅ Finalize & Save Now", width="stretch"):
                    with st.spinner("Saving structured journal entry..."):
                        res, err = api_post(
                            f"/trades/{pos_id}/debrief",
                            json_body={"conversation": st.session_state[debrief_key], "force_finalize": True},
                        )
                        if err:
                            st.error(err)
                        else:
                            st.session_state[debrief_key].append({"role": "model", "content": res["coach_message"]})
                            st.success("Journal entry saved to SQLite!")
                            st.rerun()

                for msg in st.session_state[debrief_key]:
                    with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                        st.write(msg["content"])

                with st.form(f"debrief_reply_form_{pos_id}", clear_on_submit=True):
                    trader_reply = st.text_input("Your answer to the coach...")
                    send_reply = st.form_submit_button("Send Reply", type="primary", width="stretch")
                    if send_reply and trader_reply.strip():
                        st.session_state[debrief_key].append({"role": "user", "content": trader_reply.strip()})
                        with st.spinner("Coach is analyzing your response..."):
                            res, err = api_post(
                                f"/trades/{pos_id}/debrief",
                                json_body={"conversation": st.session_state[debrief_key]},
                            )
                            if err:
                                st.error(err)
                            else:
                                st.session_state[debrief_key].append({"role": "model", "content": res["coach_message"]})
                                if res["status"] == "COMPLETED":
                                    st.toast("Debrief complete — JournalEntry saved!", icon="✅")
                                st.rerun()

# ==============================================================================
# TAB 3: PHASE 1.5 CHAT WITH JOURNAL
# ==============================================================================
with tab_chat:
    st.subheader("Interrogate Your Trading Journal (`gemini-4-hydrogen` + SQLite Tools)")
    st.caption("Ask questions about your win rates, emotional leaks, XAU/USD vs. EUR/USD edge, or The5ers drawdown buffer.")

    if "journal_chat_messages" not in st.session_state:
        st.session_state.journal_chat_messages = []

    quick_prompts = [
        "Compare my win rate and net PnL when I follow my plan vs. when I break it.",
        "Which session and instrument has my highest expectancy over the last 30 days?",
        "Show my last 5 losing trades and summarize the recurring mistake tags.",
    ]
    q_cols = st.columns(3)
    for idx, qp in enumerate(quick_prompts):
        if q_cols[idx].button(qp, key=f"qp_{idx}", width="stretch"):
            st.session_state.journal_chat_messages.append({"role": "user", "content": qp})
            with st.spinner("Querying SQLite journal..."):
                res, err = api_post(
                    "/journal/chat",
                    json_body={"messages": st.session_state.journal_chat_messages, "lookback_days": lookback_days},
                )
                if err:
                    st.error(err)
                else:
                    st.session_state.journal_chat_messages.append({"role": "model", "content": res["reply"]})
            st.rerun()

    for m in st.session_state.journal_chat_messages:
        with st.chat_message("assistant" if m["role"] == "model" else "user"):
            st.markdown(m["content"])

    user_prompt = st.chat_input("Ask anything about your trades, sessions, setups, or The5ers risk...")
    if user_prompt:
        st.session_state.journal_chat_messages.append({"role": "user", "content": user_prompt})
        with st.spinner("Querying your trade history..."):
            res, err = api_post(
                "/journal/chat",
                json_body={"messages": st.session_state.journal_chat_messages, "lookback_days": lookback_days},
            )
            if err:
                st.error(err)
            else:
                st.session_state.journal_chat_messages.append({"role": "model", "content": res["reply"]})
        st.rerun()

# ==============================================================================
# TAB 4: EDGE & SESSION ANALYTICS
# ==============================================================================
with tab_analytics:
    metrics, m_err = api_get("/analytics/metrics", params={"lookback_days": lookback_days})
    if m_err or not metrics or metrics.get("total_trades") == 0:
        st.info("Sync and journal trades to populate session, setup, and emotional expectancy charts.")
    else:
        ov = metrics["overall"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", ov["trades"])
        c2.metric("Win Rate", f"{ov['win_rate_pct']}%")
        c3.metric("Net PnL", f"${ov['net_pnl_usd']:,.2f}")
        c4.metric("Average R", f"{ov['avg_r']}R" if ov["avg_r"] is not None else "N/A")

        chart_a, chart_b = st.columns(2)
        with chart_a:
            if metrics.get("by_session"):
                sess_df = (
                    pd.DataFrame.from_dict(metrics["by_session"], orient="index")
                    .reset_index()
                    .rename(columns={"index": "Session"})
                )
                fig_sess = px.bar(
                    sess_df,
                    x="Session",
                    y="net_pnl_usd",
                    color="win_rate_pct",
                    title="Net PnL ($) & Win Rate by Session",
                    labels={"net_pnl_usd": "Net PnL ($)", "win_rate_pct": "Win Rate %"},
                )
                st.plotly_chart(fig_sess, width="stretch")
            else:
                st.info("No session breakdown data available yet.")

        with chart_b:
            if metrics.get("by_emotion"):
                emo_df = (
                    pd.DataFrame.from_dict(metrics["by_emotion"], orient="index")
                    .reset_index()
                    .rename(columns={"index": "Emotional State"})
                )
                fig_emo = px.bar(
                    emo_df,
                    x="Emotional State",
                    y="net_pnl_usd",
                    color="trades",
                    title="Net PnL ($) by Emotional State",
                    labels={"net_pnl_usd": "Net PnL ($)", "trades": "Trade Count"},
                )
                st.plotly_chart(fig_emo, width="stretch")
            else:
                st.info("Complete an AI debrief in the Journal tab to unlock Emotional State analytics.")

        if metrics.get("top_mistakes"):
            st.subheader("Top Execution Mistakes")
            st.bar_chart(pd.Series(metrics["top_mistakes"], name="Count"))

# ==============================================================================
# TAB 5: BEHAVIORAL LEAK AUDIT
# ==============================================================================
with tab_audit:
    st.subheader("AI Edge & Behavioral Leak Diagnosis")
    st.write("Uses `gemini-4-hydrogen` (`thinking_level='high'`) to cross-examine your win rates, emotional leaks, and The5ers risk rules.")
    if st.button("🔍 Run Deep Behavioral Leak Audit", type="primary"):
        with st.spinner("Auditing trade history, session expectancy, and rule violations..."):
            audit_res, err = api_post("/analytics/audit", params={"lookback_days": lookback_days})
            if err:
                st.error(f"Audit failed: {err}")
            else:
                st.markdown(audit_res["audit_report"])