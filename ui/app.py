import math
import os
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
UAE_TZ = timezone(timedelta(hours=4))

st.set_page_config(
    page_title="The5ers Trading Co-Pilot",
    page_icon="🛡️",
    layout="wide",
)


def fmt_utc_uae(iso_str: str | None) -> str:
    """Convert any ISO timestamp string into 'YYYY-MM-DD HH:MM UTC (HH:MM UAE)'."""
    if not iso_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        uae_dt = dt.astimezone(UAE_TZ)
        return f"{dt.strftime('%Y-%m-%d %H:%M')} UTC ({uae_dt.strftime('%H:%M')} UAE)"
    except Exception:
        return iso_str


def play_session_chime():
    """Play a gentle two-tone WebAudio chime when a 30-minute pre-session alert is active."""
    components.html(
        """
        <script>
        (function() {
            try {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                const ctx = new AudioCtx();
                const now = ctx.currentTime;
                [523.25, 659.25].forEach((freq, idx) => {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(freq, now + idx * 0.18);
                    gain.gain.setValueAtTime(0.08, now + idx * 0.18);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.18 + 0.35);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start(now + idx * 0.18);
                    osc.stop(now + idx * 0.18 + 0.36);
                });
            } catch (e) {}
        })();
        </script>
        """,
        height=0,
    )


def calculate_the5ers_position_size(
    symbol: str,
    account_balance: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    tp1_price: float = 0.0,
    tp2_price: float = 0.0,
    daily_buffer_remaining_usd: float | None = None,
    usdjpy_rate: float = 152.0,
) -> dict:
    """
    Calculate exact MT5 lot size (rounded down to 0.01 lots so risk never exceeds cap)
    and R:R projections for XAUUSD, EURUSD, GBPUSD, and USDJPY.
    """
    sym = symbol.upper().strip()
    stop_dist = abs(entry_price - stop_loss)
    if stop_dist <= 0 or account_balance <= 0:
        return {"valid": False, "error": "Entry price and Stop Loss must be different positive numbers."}

    # Standard MT5 contract specs:
    # - XAUUSD: 1.00 lot = 100 oz ($1.00 price move = $100/lot; 1 pip = $0.10 = $10/lot)
    # - EURUSD / GBPUSD: 1.00 lot = 100,000 units ($0.0001 pip = $10/lot)
    # - USDJPY: 1.00 lot = 100,000 USD (0.01 pip = 1,000 JPY / usdjpy_rate per lot)
    if sym == "XAUUSD":
        pip_size = 0.10
        usd_per_price_unit_per_lot = 100.0
    elif sym == "USDJPY":
        pip_size = 0.01
        rate = usdjpy_rate if usdjpy_rate > 0 else (entry_price if entry_price > 50 else 152.0)
        usd_per_price_unit_per_lot = 100_000.0 / rate
    else:
        pip_size = 0.0001
        usd_per_price_unit_per_lot = 100_000.0

    stop_pips = round(stop_dist / pip_size, 1)
    risk_per_lot_usd = stop_dist * usd_per_price_unit_per_lot

    target_risk_usd = round(account_balance * (risk_pct / 100.0), 2)
    capped_by_daily_buffer = False
    effective_risk_budget = target_risk_usd

    if daily_buffer_remaining_usd is not None and daily_buffer_remaining_usd > 0:
        # Leave at least a 10% safety cushion inside remaining daily loss buffer
        safe_daily_cap = round(daily_buffer_remaining_usd * 0.90, 2)
        if effective_risk_budget > safe_daily_cap:
            effective_risk_budget = safe_daily_cap
            capped_by_daily_buffer = True

    raw_lots = effective_risk_budget / risk_per_lot_usd if risk_per_lot_usd > 0 else 0.0
    # Floor to 0.01 micro-lots so actual risk never exceeds target
    lots = math.floor(raw_lots * 100.0) / 100.0
    actual_risk_usd = round(lots * risk_per_lot_usd, 2)
    actual_risk_pct = round((actual_risk_usd / account_balance) * 100.0, 2)

    is_buy = entry_price > stop_loss

    def _eval_tp(tp_val: float):
        if not tp_val or tp_val <= 0:
            return None, None
        valid_dir = (tp_val > entry_price) if is_buy else (tp_val < entry_price)
        if not valid_dir:
            return None, None
        reward_dist = abs(tp_val - entry_price)
        rr = round(reward_dist / stop_dist, 2)
        profit_usd = round(lots * reward_dist * usd_per_price_unit_per_lot, 2)
        return rr, profit_usd

    tp1_rr, tp1_usd = _eval_tp(tp1_price)
    tp2_rr, tp2_usd = _eval_tp(tp2_price)

    return {
        "valid": True,
        "direction": "BUY" if is_buy else "SELL",
        "lots": lots,
        "stop_pips": stop_pips,
        "target_risk_usd": target_risk_usd,
        "actual_risk_usd": actual_risk_usd,
        "actual_risk_pct": actual_risk_pct,
        "capped_by_daily_buffer": capped_by_daily_buffer,
        "tp1_rr": tp1_rr,
        "tp1_usd": tp1_usd,
        "tp2_rr": tp2_rr,
        "tp2_usd": tp2_usd,
    }


def _extract_http_error(exc: requests.HTTPError) -> str:
    if exc.response is None:
        return str(exc)
    try:
        data = exc.response.json()
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
        return str(data)
    except Exception:
        raw_text = (exc.response.text or "").strip()
        return f"HTTP {exc.response.status_code}: {raw_text or str(exc)}"


def api_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as e:
        return None, _extract_http_error(e)
    except Exception as e:
        return None, str(e)


def api_post(path: str, json_body: dict | None = None, params: dict | None = None):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, params=params, timeout=120)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as e:
        return None, _extract_http_error(e)
    except Exception as e:
        return None, str(e)


def api_upload_image(uploaded_file) -> tuple[str | None, str | None]:
    if uploaded_file is None:
        return None, None
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/png")}
        resp = requests.post(f"{API_BASE_URL}/charts/upload", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json().get("chart_image_path"), None
    except requests.HTTPError as e:
        return None, _extract_http_error(e)
    except Exception as e:
        return None, str(e)


def api_delete(path: str):
    try:
        resp = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as e:
        return None, _extract_http_error(e)
    except Exception as e:
        return None, str(e)


def render_drawdown_gauge(title: str, used_pct: float, soft_limit: float, hard_limit: float, remaining_usd: float):
    bar_color = "#22c55e"
    if used_pct >= soft_limit:
        bar_color = "#ef4444"
    elif used_pct >= (soft_limit * 0.65):
        bar_color = "#f59e0b"

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
        "BULLISH": "🟢 Uptrend (Buyers)",
        "BEARISH": "🔴 Downtrend (Sellers)",
        "NEUTRAL": "⚪ Sideways / Balanced",
        "VOLATILE_CHOP": "🟠 Messy / Up-Down Spikes",
    }
    return mapping.get(bias, bias)


def regime_badge(regime: str) -> str:
    mapping = {
        "EVENT_VOLATILITY": "⚡ Major News Ahead",
        "RATE_DRIVEN": "🏦 Interest-Rate Driven",
        "RISK_ON": "📈 Confident / Risk-On",
        "RISK_OFF": "🛡️ Cautious / Safe-Haven",
    }
    return mapping.get(regime, regime)


def market_speed_label(regime: str) -> str:
    mapping = {
        "DEAD_COMPRESSED": "Slow / Quiet",
        "NORMAL_HEALTHY": "Normal Pace",
        "HIGH_EXPANSION": "Fast & Volatile",
    }
    return mapping.get(regime, "Normal Pace")


def spread_status_label(symbol: str, spread_points: int) -> str:
    max_normal = 45 if "XAU" in symbol.upper() else 18
    if spread_points > max_normal:
        return f"⚠️ High Spread ({spread_points} pts)"
    return f"Normal Spread ({spread_points} pts)"


# --- Sidebar Controls ---
with st.sidebar:
    st.title("🛡️ Trading Co-Pilot")
    st.caption("The5ers High Stakes Risk & Setup Assistant")

    lookback_days = st.slider("History Lookback (Days)", min_value=7, max_value=180, value=30, step=1)
    live_poll_enabled = st.toggle("📡 Live Market & Trade Watch (15s)", value=True)
    sound_alert_enabled = st.toggle("🔔 Sound Chime Before Session Open", value=False)

    if st.button("🔄 Sync MetaTrader 5 Now", width="stretch", type="primary"):
        with st.spinner("Syncing trades & stop losses from MetaTrader 5..."):
            sync_data, err = api_post("/sync/mt5", params={"lookback_days": lookback_days})
            if err:
                st.error(f"Sync failed: {err}")
            else:
                msg = f"Synced {sync_data['trades_synced']} closed | {sync_data['open_positions_count']} live open"
                if sync_data.get("auto_verdicts_generated", 0) > 0:
                    msg += f" | {sync_data['auto_verdicts_generated']} graded!"
                st.success(msg)
                st.rerun()

    if st.button("🌍 Refresh News & Market Guide", width="stretch"):
        with st.spinner("Checking this week's economic calendar & market direction..."):
            macro_res, m_err = api_post("/macro/refresh")
            if m_err:
                st.error(f"Market scan failed: {m_err}")
            else:
                st.success(f"Updated market guide ({macro_res['events_upserted']} events)!")
                st.rerun()

    st.divider()
    symbol_filter = st.selectbox("Filter Instrument", ["ALL", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])

    now_utc = datetime.now(timezone.utc)
    now_uae = now_utc.astimezone(UAE_TZ)
    st.info(
        f"**Current Time:**\n"
        f"`{now_utc.strftime('%H:%M')} UTC` (`{now_uae.strftime('%H:%M')} UAE`)\n\n"
        f"Account: **The5ers High Stakes**"
    )


# --- Header, Drawdown Soft-Lock & News Blackout Banner ---
st.title("The5ers High Stakes — Trading Co-Pilot")

dd_status, dd_err = api_get("/account/drawdown")
if dd_err:
    st.error(f"Backend Offline ({API_BASE_URL}): {dd_err}")
    st.stop()

if dd_status["soft_lock_triggered"]:
    st.error(f"🚨 **DRAWDOWN STOP-LOSS TRIGGERED:** {dd_status['lock_reason']} Stop trading for today.")

blackout_status, _ = api_get("/macro/blackout", params={"symbol": symbol_filter})
if blackout_status:
    if blackout_status["is_blackout_active"]:
        st.error(f"🛑 **{blackout_status['message']}**")
    else:
        st.success(f"🟢 **{blackout_status['message']}**")


# ==============================================================================
# LIVE SESSION COUNTDOWN, CLEAN MARKET CARDS & OPEN-POSITION WATCHER (15s POLL)
# ==============================================================================
@st.fragment(run_every=15 if live_poll_enabled else None)
def render_live_session_and_positions():
    pulse, pulse_err = api_get("/market/pulse", params={"symbols": "XAUUSD,EURUSD"})
    if not pulse_err and pulse:
        sess = pulse["session"]
        sev = sess.get("alert_severity", "INFO")
        if sev == "DANGER":
            st.error(sess["banner_message"])
        elif sev == "WARNING":
            st.warning(sess["banner_message"])
            if sound_alert_enabled and sess.get("pre_session_alert_active"):
                alert_sig = f"{sess['next_event_name']}_{sess['next_event_time_utc_uae']}"
                if st.session_state.get("last_chimed_session") != alert_sig:
                    play_session_chime()
                    st.session_state["last_chimed_session"] = alert_sig
        else:
            st.info(sess["banner_message"])

        vols = pulse.get("volatility", [])
        if vols:
            v_cols = st.columns(len(vols))
            for idx, v in enumerate(vols):
                room_left_pct = max(0, int(round(100.0 - float(v["adr_used_pct"]))))
                speed_txt = market_speed_label(v["volatility_regime"])
                spread_txt = spread_status_label(v["symbol"], int(v["spread_points"]))
                pips_left = max(0.0, round(float(v["adr_14d_pips"]) - float(v["today_range_pips"]), 1))

                with v_cols[idx]:
                    with st.container(border=True):
                        vc1, vc2, vc3 = st.columns(3)
                        vc1.metric(
                            f"{v['symbol']} Price",
                            f"{v['bid_price']}",
                            spread_txt,
                            delta_color="off" if "Normal" in spread_txt else "inverse",
                        )
                        vc2.metric(
                            "Market Speed",
                            speed_txt,
                            f"{v['volume_surge_ratio']}x typical activity",
                            delta_color="off",
                        )
                        vc3.metric(
                            "Daily Move Left",
                            f"{room_left_pct}% Left",
                            f"~{pips_left} pips room",
                            delta_color="normal" if room_left_pct > 25 else "inverse",
                        )
                        st.caption(v["plain_english_summary"])

    open_positions, open_err = api_get("/positions/open", params={"poll_mt5": True})
    if open_err or not open_positions:
        return

    st.warning(f"⚡ **LIVE OPEN POSITIONS ({len(open_positions)}) — Log Your Entry Reason Before Closing!**")

    for pos in open_positions:
        pos_id = pos["mt5_position_id"]
        sl_text = f"{pos['stop_loss']}" if pos.get("stop_loss") else "⚠️ NO STOP LOSS SET"
        risk_text = f"${pos['risk_amount_usd']:.2f}" if pos.get("risk_amount_usd") else "Unknown"

        with st.container(border=True):
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric(f"Ticket #{pos_id}", f"{pos['symbol']} {pos['direction']}", f"{pos['volume']} lots")
            h2.metric("Entry Price", f"{pos['entry_price']}", f"SL: {sl_text}")
            h3.metric("Live Floating P&L", f"${pos['floating_pnl']:+.2f}", f"Risk: {risk_text}")
            h4.metric("Opened — UTC (UAE)", fmt_utc_uae(pos["open_time"]))
            h5.metric("Entry Plan", "✅ Saved" if pos["stage1_completed"] else "⏳ Action Needed")

            if pos["stage1_completed"]:
                st.success(
                    f"**Saved Entry Plan ({pos.get('setup_type')}):** {pos.get('entry_thesis')} | "
                    f"**Stop-Loss Plan:** {pos.get('invalidation_plan')} | **Mindset:** `{pos.get('pre_trade_emotion')}`"
                )
            else:
                with st.form(f"stage1_form_{pos_id}"):
                    st.markdown(f"**📋 Why did you enter this `{pos['symbol']} {pos['direction']}` trade?**")
                    c1, c2, c3 = st.columns(3)
                    setup_type = c1.selectbox(
                        "Setup Pattern",
                        [
                            "ICT Liquidity Sweep + FVG",
                            "Elliott Wave 2 / 4 Pullback",
                            "Fibonacci Golden Zone Overlap",
                            "Wave 5 Exhaustion & RSI Divergence",
                            "Trendline Break & Retest",
                            "Asian Range Killzone Setup",
                            "Impulse / Unplanned",
                        ],
                        key=f"s1_setup_{pos_id}",
                    )
                    timeframe = c2.selectbox("Timeframe", ["M5", "M15", "M30", "H1", "H4"], index=1, key=f"s1_tf_{pos_id}")
                    emotion = c3.selectbox(
                        "Mental State Right Now",
                        ["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "BORED"],
                        key=f"s1_emo_{pos_id}",
                    )

                    entry_thesis = st.text_input(
                        "Why was this trade taken? (Entry Trigger & Confluence)",
                        placeholder="E.g., Wave 2 pullback into 0.618 Fib + 15m bullish FVG retest.",
                        key=f"s1_thesis_{pos_id}",
                    )
                    invalidation_plan = st.text_input(
                        "Where is your hard Stop Loss and what proves this trade wrong?",
                        placeholder="E.g., Hard SL below Wave 1 origin at 3545.00 (0.5% risk).",
                        key=f"s1_inv_{pos_id}",
                    )

                    u1, u2 = st.columns([1, 2])
                    macro_aligned = u1.checkbox("Aligned with Market Guide Bias", value=True, key=f"s1_macro_{pos_id}")
                    chart_file = u2.file_uploader(
                        "Optional: Attach Entry Chart Screenshot (PNG/JPG)",
                        type=["png", "jpg", "jpeg"],
                        key=f"s1_chart_{pos_id}",
                    )

                    submit_s1 = st.form_submit_button("🔒 Save Entry Plan", type="primary", width="stretch")
                    if submit_s1:
                        if not entry_thesis.strip() or not invalidation_plan.strip():
                            st.warning("Fill in both your Entry Reason and Stop-Loss Plan before saving.")
                        else:
                            chart_path, up_err = api_upload_image(chart_file)
                            if up_err:
                                st.error(f"Chart upload failed: {up_err}")
                            else:
                                payload = {
                                    "setup_type": setup_type,
                                    "timeframe": timeframe,
                                    "entry_thesis": entry_thesis.strip(),
                                    "invalidation_plan": invalidation_plan.strip(),
                                    "pre_trade_emotion": emotion,
                                    "macro_aligned": macro_aligned,
                                    "entry_chart_path": chart_path,
                                }
                                _, s1_err = api_post(f"/trades/{pos_id}/stage1", json_body=payload)
                                if s1_err:
                                    st.error(s1_err)
                                else:
                                    st.toast(f"Entry plan saved for #{pos_id}!", icon="🔒")
                                    st.rerun()


render_live_session_and_positions()

# --- Drawdown Metrics & Gauges ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Current Equity", f"${dd_status['current_equity']:,.2f}", f"Ref: ${dd_status['day_reference_level']:,.2f}")
col_m2.metric("Starting Balance", f"${dd_status['starting_account_balance']:,.2f}", f"Date: {dd_status['trading_date']}")
col_m3.metric("Daily Buffer Left", f"${dd_status['daily_drawdown_remaining_usd']:,.2f}", f"-{dd_status['daily_drawdown_used_pct']}% / 5.0%")
col_m4.metric("Static Buffer Left", f"${dd_status['static_drawdown_remaining_usd']:,.2f}", f"-{dd_status['static_drawdown_used_pct']}% / 10.0%")

g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(
        render_drawdown_gauge(
            "Daily Loss Limit (5.0% Max / 3.5% Soft Stop)",
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
            "Max Static Drawdown (10.0% Max / 7.5% Soft Stop)",
            dd_status["static_drawdown_used_pct"],
            soft_limit=7.5,
            hard_limit=10.0,
            remaining_usd=dd_status["static_drawdown_remaining_usd"],
        ),
        width="stretch",
    )

st.divider()

# --- Main 6 Tabs ---
tab_gatekeeper, tab_journal, tab_macro, tab_chat, tab_analytics, tab_audit = st.tabs(
    [
        "🛡️ Chart & Setup Advisor",
        "📓 Trade Log & Grades",
        "🌍 News & Market Guide",
        "💬 Ask Your Journal",
        "📊 Win Rate & Discipline",
        "🧠 Habit & Leak Review",
    ]
)

# ==============================================================================
# TAB 1: CHART & SETUP ADVISOR + THE5ERS POSITION SIZE CALCULATOR
# ==============================================================================
with tab_gatekeeper:
    st.subheader("Chart & Setup Advisor — Strict Strategy Checklist")
    st.caption(
        "Upload a Higher-Timeframe (1H/4H) and/or Execution (15m/5m) chart—drawn or blank. "
        "Every strategy is audited against strict mandatory rules (Elliott Wave, Fibonacci Golden Zone, ICT, Trendlines & Asian Killzones)."
    )

    gk_left, gk_right = st.columns([1.0, 1.3], gap="large")

    with gk_left:
        with st.form("pre_trade_gatekeeper_form"):
            intent_choice = st.radio(
                "How should the Advisor check your chart?",
                [
                    "🧭 Analyze Chart / Unsure (Recommend Best Strategy)",
                    "🟢 Check My BUY Plan",
                    "🔴 Check My SELL Plan",
                ],
                horizontal=False,
            )

            r1_c1, r1_c2, r1_c3 = st.columns(3)
            gk_symbol = r1_c1.selectbox("Instrument", ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])
            gk_htf = r1_c2.selectbox("Higher Timeframe", ["H1", "H4", "D1"], index=0)
            gk_tf = r1_c3.selectbox("Execution TF", ["M5", "M15", "M30", "H1"], index=1)

            gk_risk = st.slider(
                "Planned Risk Per Trade (% of Account — Max 1.0%)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
            )

            u_col1, u_col2 = st.columns(2)
            htf_chart_file = u_col1.file_uploader(
                "1. Optional: Higher-Timeframe Chart (1H / 4H)",
                type=["png", "jpg", "jpeg", "webp"],
            )
            ltf_chart_file = u_col2.file_uploader(
                "2. Execution Chart (15m / 5m — Drawn or Blank)",
                type=["png", "jpg", "jpeg", "webp"],
            )

            gk_desc = st.text_area(
                "Notes, Wave Count, Fib Levels, or Questions (Optional)",
                placeholder="E.g., Check if Gold is in a Wave 2 pullback into the 0.618 Fib Golden Zone or if my trendline is a trap.",
                height=85,
            )

            run_gk = st.form_submit_button("🧠 Check Strategy Rules & Build Plan", type="primary", width="stretch")

            if run_gk:
                if not htf_chart_file and not ltf_chart_file and not gk_desc.strip():
                    st.warning("Upload at least one chart screenshot or describe your setup first.")
                else:
                    if "BUY" in intent_choice:
                        intent_mode = "BUY"
                        prop_dir = "BUY"
                    elif "SELL" in intent_choice:
                        intent_mode = "SELL"
                        prop_dir = "SELL"
                    else:
                        intent_mode = "RECOMMEND"
                        prop_dir = "UNSPECIFIED"

                    with st.spinner("Auditing Elliott Wave cycle, Fibonacci levels, ICT zones, and strict strategy rules..."):
                        htf_path, htf_err = api_upload_image(htf_chart_file)
                        ltf_path, ltf_err = api_upload_image(ltf_chart_file)
                        if htf_err or ltf_err:
                            st.error(f"Chart upload failed: {htf_err or ltf_err}")
                        else:
                            payload = {
                                "symbol": gk_symbol,
                                "intent_mode": intent_mode,
                                "proposed_direction": prop_dir,
                                "timeframe": gk_tf,
                                "htf_timeframe": gk_htf,
                                "proposed_risk_pct": gk_risk,
                                "setup_description": gk_desc.strip(),
                                "chart_image_path": ltf_path,
                                "htf_chart_image_path": htf_path,
                            }
                            gk_res, gk_err = api_post("/gatekeeper/audit", json_body=payload)
                            if gk_err:
                                st.error(gk_err)
                            else:
                                st.session_state["latest_gk_audit"] = gk_res

        # --- The5ers Position Size (Lot) Calculator Widget ---
        st.divider()
        with st.container(border=True):
            st.markdown("#### 🧮 The5ers Position Size (Lot) Calculator")
            st.caption(
                f"Uses your current equity (`${dd_status['current_equity']:,.2f}`) and clamps risk to "
                f"The5ers' **1.0% max cap** and remaining daily loss buffer (`${dd_status['daily_drawdown_remaining_usd']:,.2f}`)."
            )

            calc_c1, calc_c2 = st.columns(2)
            calc_symbol = calc_c1.selectbox(
                "Pair / Metal",
                ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
                index=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"].index(gk_symbol),
                key="lot_calc_symbol",
            )
            calc_risk_pct = calc_c2.number_input(
                "Risk % (Max 1.0%)",
                min_value=0.10,
                max_value=1.00,
                value=min(1.00, float(gk_risk)),
                step=0.05,
                format="%.2f",
                key="lot_calc_risk_pct",
            )

            default_prices = {
                "XAUUSD": (3620.00, 3645.00, 3590.00, 3530.00),
                "EURUSD": (1.10000, 1.10250, 1.09500, 1.09000),
                "GBPUSD": (1.29000, 1.29300, 1.28400, 1.27800),
                "USDJPY": (152.000, 152.350, 151.300, 150.600),
            }
            d_entry, d_sl, d_tp1, d_tp2 = default_prices[calc_symbol]
            step_val = 0.10 if calc_symbol == "XAUUSD" else (0.01 if calc_symbol == "USDJPY" else 0.0001)
            fmt_str = "%.2f" if calc_symbol == "XAUUSD" else ("%.3f" if calc_symbol == "USDJPY" else "%.5f")

            p1, p2 = st.columns(2)
            calc_entry = p1.number_input(
                "Entry Price",
                min_value=0.0001,
                value=d_entry,
                step=step_val,
                format=fmt_str,
                key=f"calc_entry_{calc_symbol}",
            )
            calc_sl = p2.number_input(
                "Stop Loss (SL)",
                min_value=0.0001,
                value=d_sl,
                step=step_val,
                format=fmt_str,
                key=f"calc_sl_{calc_symbol}",
            )

            t1, t2 = st.columns(2)
            calc_tp1 = t1.number_input(
                "Take-Profit 1 (Optional)",
                min_value=0.0,
                value=d_tp1,
                step=step_val,
                format=fmt_str,
                key=f"calc_tp1_{calc_symbol}",
            )
            calc_tp2 = t2.number_input(
                "Take-Profit 2 (Optional)",
                min_value=0.0,
                value=d_tp2,
                step=step_val,
                format=fmt_str,
                key=f"calc_tp2_{calc_symbol}",
            )

            sizing = calculate_the5ers_position_size(
                symbol=calc_symbol,
                account_balance=float(dd_status["current_equity"]),
                risk_pct=float(calc_risk_pct),
                entry_price=float(calc_entry),
                stop_loss=float(calc_sl),
                tp1_price=float(calc_tp1),
                tp2_price=float(calc_tp2),
                daily_buffer_remaining_usd=float(dd_status["daily_drawdown_remaining_usd"]),
            )

            if not sizing["valid"]:
                st.warning(sizing["error"])
            else:
                if sizing["capped_by_daily_buffer"]:
                    st.warning("⚠️ Risk was automatically reduced so a loss cannot breach your remaining daily drawdown limit.")

                m_lot, m_risk, m_stop = st.columns(3)
                m_lot.metric("MT5 Lot Size", f"{sizing['lots']:.2f} lots", sizing["direction"])
                m_risk.metric("Exact Dollar Risk", f"${sizing['actual_risk_usd']:.2f}", f"{sizing['actual_risk_pct']}% of equity")
                m_stop.metric("Stop Distance", f"{sizing['stop_pips']} pips", f"Target: ${sizing['target_risk_usd']:.2f}")

                if sizing["lots"] < 0.01:
                    st.error("🛑 Stop loss is too wide for 0.01 micro-lots at this risk %. Tighten your SL or skip this setup.")
                else:
                    rr_parts = []
                    if sizing["tp1_rr"] is not None:
                        rr_parts.append(f"**TP1:** `{sizing['tp1_rr']}R` (`+${sizing['tp1_usd']:.2f}`)")
                    if sizing["tp2_rr"] is not None:
                        rr_parts.append(f"**TP2:** `{sizing['tp2_rr']}R` (`+${sizing['tp2_usd']:.2f}`)")
                    if rr_parts:
                        st.success(" | ".join(rr_parts))

    with gk_right:
        st.subheader("Strategy Rule Audit & Trade Blueprint")
        latest_gk = st.session_state.get("latest_gk_audit")
        if latest_gk:
            rep = latest_gk["report"]
            verdict = rep["verdict"]
            score = rep["discipline_score"]
            rec_action = rep["recommended_action"]
            rec_strat = rep["recommended_strategy"]
            conf_count = rep["confluences_count"]
            mode_tag = rep["analysis_mode"].replace("_", " ").title()

            if verdict == "PASS":
                st.success(f"### ✅ VERDICT: PASS ({conf_count} Signals Aligned | Score: {score}/100)")
            elif verdict == "CAUTION":
                st.warning(f"### ⚠️ VERDICT: CAUTION — WAIT FOR TRIGGER ({conf_count} Signal(s) | Score: {score}/100)")
            else:
                st.error(f"### 🛑 VERDICT: BLOCK — DO NOT ENTER (Score: {score}/100)")

            bp1, bp2, bp3 = st.columns(3)
            bp1.metric("Recommended Play", rec_action)
            bp2.metric("Matched Strategy", rec_strat)
            bp3.metric("Chart Check Mode", mode_tag)

            checklist = rep.get("strategy_checklist", [])
            if checklist:
                with st.container(border=True):
                    st.markdown(f"**📋 Mandatory Rule Checklist — {rec_strat}:**")
                    for item in checklist:
                        st_code = item.get("status", "MISSING")
                        icon = "✅" if st_code == "PASS" else ("⏳" if st_code == "MISSING" else "❌")
                        st.markdown(f"- {icon} **`[{st_code}]` {item.get('rule_name')}:** {item.get('detail')}")

            if rep.get("confluences_found"):
                st.success(
                    f"**🧩 Aligning Confluences ({conf_count}):**\n"
                    + "\n".join(f"- {c}" for c in rep["confluences_found"])
                )

            with st.container(border=True):
                st.markdown(f"**🗺️ Higher-Timeframe Map:** {rep['htf_bias_read']}")
                st.markdown(f"**🌊 Elliott Wave Cycle Position:** {rep['wave_cycle_read']}")
                st.markdown(f"**📐 Fibonacci Levels & Golden Zone:** {rep['fib_levels_summary']}")
                st.markdown(f"**🔍 Trendline & ICT Check:** {rep['trendline_ict_critique']}")
                st.markdown(f"**📊 Execution Chart Read:** {rep['chart_structure_read']}")

            st.info(f"**🎯 Exact Trigger to Wait For:** {rep['trigger_to_wait_for']}")
            lvl1, lvl2, lvl3 = st.columns(3)
            with lvl1:
                st.error(f"**🛑 Stop-Loss Zone:**\n\n{rep['invalidation_sl_zone']}")
            with lvl2:
                st.success(f"**🎯 Take-Profit 1 (TP1):**\n\n{rep['tp1_zone']}")
            with lvl3:
                st.success(f"**🚀 Take-Profit 2 (TP2):**\n\n{rep['tp2_zone']}")

            if rep.get("rule_violations"):
                st.warning("**⚠️ Rule Violations / Missing Conditions:**\n" + "\n".join(f"- {v}" for v in rep["rule_violations"]))

            st.markdown(f"**💬 Advisor Summary:** {rep['coaching_verdict']}")
        else:
            st.info(
                "Upload your 1H/4H and/or 15m chart on the left to run a **strict rule-by-rule audit** across:\n"
                "- **Elliott Wave Cycle & Invalidation** (`Wave 2`, `Wave 4`, or `Wave 5` + RSI divergence)\n"
                "- **Fibonacci Golden Zone (`0.50–0.618–0.786`)** + **`TP1 / TP2` (`-0.272` / `-0.618`)**\n"
                "- **ICT Liquidity Sweep + FVG/OB**, **Trendline Geometry**, & **Asian Range Killzones**"
            )

        history_data, _ = api_get("/gatekeeper/history", params={"limit": 5})
        if history_data:
            with st.expander("Recent Chart Checks"):
                for item in history_data:
                    st.markdown(
                        f"- **{item['verdict']}** → Play: `{item['recommended_action']}` ({item['recommended_strategy']}, "
                        f"{item['confluences_count']} signals) on `{item['symbol']}` at `{item['created_at']}`"
                    )

# ==============================================================================
# TAB 2: AUTO-GRADED CLOSED TRADES & CLICK-TO-INSPECT REASON
# ==============================================================================
trades_data, trades_err = api_get("/trades", params={"limit": 200, "symbol": symbol_filter})
trades_data = trades_data or []

ungraded_count = sum(1 for t in trades_data if not t.get("process_verdict"))
if ungraded_count > 0 and not st.session_state.get("auto_grade_triggered"):
    api_post("/trades/auto-grade-pending")
    st.session_state["auto_grade_triggered"] = True

with tab_journal:
    left_col, right_col = st.columns([1.45, 1.0], gap="large")

    with left_col:
        st.subheader("Closed Trades — Click Any Row to See Why It Was Good or Bad")
        if ungraded_count > 0:
            st.caption(f"⏳ Grading {ungraded_count} trade(s) in the background—refresh in a moment if any show 'Grading...'.")

        if not trades_data:
            st.warning("No closed trades found. Click **Sync MetaTrader 5 Now** in the sidebar.")
        else:
            df = pd.DataFrame(trades_data)
            df["closed_utc_uae"] = df["close_time"].apply(fmt_utc_uae)
            df["verdict_badge"] = df["process_verdict"].apply(
                lambda v: "🟢 GOOD TRADE" if v == "GOOD_TRADE" else ("🔴 BAD TRADE" if v == "BAD_TRADE" else "⏳ Grading...")
            )
            display_cols = [
                "mt5_position_id",
                "closed_utc_uae",
                "symbol",
                "direction",
                "net_profit",
                "r_multiple",
                "verdict_badge",
                "setup_type",
                "session",
            ]
            selection_event = st.dataframe(
                df[display_cols],
                width="stretch",
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "mt5_position_id": "Ticket",
                    "closed_utc_uae": "Closed — UTC (UAE)",
                    "net_profit": st.column_config.NumberColumn("Net PnL ($)", format="$%.2f"),
                    "r_multiple": st.column_config.NumberColumn("R-Mult", format="%.2fR"),
                    "verdict_badge": "Process Grade",
                },
            )

            df_sorted = df.sort_values("close_time")
            df_sorted["cum_pnl"] = df_sorted["net_profit"].cumsum()
            fig_eq = px.line(
                df_sorted,
                x="closed_utc_uae",
                y="cum_pnl",
                markers=True,
                title="Cumulative Realized PnL ($)",
                labels={"closed_utc_uae": "Close Time — UTC (UAE)", "cum_pnl": "Net Profit ($)"},
            )
            fig_eq.update_layout(height=270, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_eq, width="stretch")

    with right_col:
        st.subheader("Why This Trade Was Graded Good or Bad")
        if not trades_data:
            st.info("Sync MT5 trades on the left to view automatic Good/Bad Trade explanations.")
        else:
            selected_idx = 0
            if selection_event and selection_event.selection and selection_event.selection.rows:
                selected_idx = selection_event.selection.rows[0]

            selected_trade = trades_data[selected_idx]
            pos_id = selected_trade["mt5_position_id"]

            sl_disp = f"{selected_trade['stop_loss']}" if selected_trade.get("stop_loss") else "⚠️ None Set"
            r_disp = f"{selected_trade['r_multiple']:+.2f}R" if selected_trade.get("r_multiple") is not None else "N/A"
            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Ticket #{pos_id}", f"{selected_trade['symbol']} {selected_trade['direction']}", f"{selected_trade['volume']} lots")
                m2.metric("Net P&L / R", f"${selected_trade['net_profit']:+.2f}", f"R: {r_disp}")
                m3.metric("Entry / SL", f"{selected_trade['entry_price']}", f"SL: {sl_disp}")
                st.caption(f"Closed: `{fmt_utc_uae(selected_trade['close_time'])}` | Session: `{selected_trade.get('session') or 'N/A'}`")

            if selected_trade.get("entry_thesis"):
                st.info(f"**📋 Saved Entry Plan:** {selected_trade['entry_thesis']}")

            pv = selected_trade.get("process_verdict")
            quad = selected_trade.get("outcome_quadrant") or ""
            quad_human = {
                "GOOD_PROCESS_GOOD_OUTCOME": "Good Process + Profitable Outcome",
                "GOOD_PROCESS_BAD_OUTCOME": "Good Process + Normal Stop-Out Loss",
                "BAD_PROCESS_GOOD_OUTCOME": "Bad Process + Lucky Profit (High Risk)",
                "BAD_PROCESS_BAD_OUTCOME": "Bad Process + Loss",
            }.get(quad, quad)

            if pv == "GOOD_TRADE":
                st.success(
                    f"### 🟢 GOOD TRADE\n"
                    f"**Classification:** `{quad_human}` | **Setup:** `{selected_trade.get('setup_type') or 'N/A'}`\n\n"
                    f"**Why:** {selected_trade.get('ai_feedback', 'Followed your trading plan and risk limits.')}"
                )
            elif pv == "BAD_TRADE":
                st.error(
                    f"### 🔴 BAD TRADE\n"
                    f"**Classification:** `{quad_human}` | **Setup:** `{selected_trade.get('setup_type') or 'N/A'}`\n\n"
                    f"**Why:** {selected_trade.get('ai_feedback', 'Broke one or more execution or risk rules.')}"
                )
            else:
                st.warning("⏳ **Automatic grading is running in the background...** Click **Sync MetaTrader 5 Now** or refresh in a moment.")

            if selected_trade.get("mistake_tags"):
                st.caption("**Detected Flags:** " + ", ".join(f"`{tag}`" for tag in selected_trade["mistake_tags"]))

            with st.expander("🎙️ Optional: Coaching Debrief or Re-Check"):
                debrief_key = f"debrief_history_{pos_id}"
                if debrief_key not in st.session_state:
                    st.session_state[debrief_key] = []

                col_b1, col_b2 = st.columns(2)
                if col_b1.button("Start / Reset Interview", key=f"start_db_{pos_id}", width="stretch"):
                    with st.spinner("Reviewing execution details..."):
                        res, err = api_post(f"/trades/{pos_id}/debrief", json_body={"conversation": []})
                        if err:
                            st.error(err)
                        else:
                            st.session_state[debrief_key] = [{"role": "model", "content": res["coach_message"]}]
                            st.rerun()

                if col_b2.button("Re-Run Trade Grade", key=f"regrade_{pos_id}", width="stretch"):
                    with st.spinner("Re-checking trade execution..."):
                        _, v_err = api_post(f"/trades/{pos_id}/verdict")
                        if v_err:
                            st.error(v_err)
                        else:
                            st.rerun()

                if st.session_state[debrief_key]:
                    if st.button("✅ Finalize & Save Interview", key=f"fin_db_{pos_id}", width="stretch"):
                        with st.spinner("Saving journal entry..."):
                            res, err = api_post(
                                f"/trades/{pos_id}/debrief",
                                json_body={"conversation": st.session_state[debrief_key], "force_finalize": True},
                            )
                            if err:
                                st.error(err)
                            else:
                                st.session_state[debrief_key].append({"role": "model", "content": res["coach_message"]})
                                st.rerun()

                    for msg in st.session_state[debrief_key]:
                        with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                            st.write(msg["content"])

                    with st.form(f"debrief_reply_form_{pos_id}", clear_on_submit=True):
                        trader_reply = st.text_input("Your answer...")
                        send_reply = st.form_submit_button("Send Reply", width="stretch")
                        if send_reply and trader_reply.strip():
                            st.session_state[debrief_key].append({"role": "user", "content": trader_reply.strip()})
                            with st.spinner("Reviewing your answer..."):
                                res, err = api_post(
                                    f"/trades/{pos_id}/debrief",
                                    json_body={"conversation": st.session_state[debrief_key]},
                                )
                                if err:
                                    st.error(err)
                                else:
                                    st.session_state[debrief_key].append({"role": "model", "content": res["coach_message"]})
                                    st.rerun()

# ==============================================================================
# TAB 3: NEWS & MARKET GUIDE
# ==============================================================================
with tab_macro:
    macro_data, macro_err = api_get("/macro/latest")
    if macro_err:
        st.error(f"Could not load market guide: {macro_err}")
    else:
        snap = macro_data.get("snapshot")
        events = macro_data.get("events", [])

        top_m1, top_m2 = st.columns([3, 1])
        with top_m1:
            st.subheader("Market Condition & Trade Direction Guide")
            if snap:
                st.caption(f"Last Updated: `{fmt_utc_uae(snap['updated_at'])}`")
            else:
                st.info("No market guide saved yet. Click **🌍 Refresh News & Market Guide** in the sidebar.")
        with top_m2:
            if st.button("🔄 Refresh Market Guide", type="primary", width="stretch"):
                with st.spinner("Checking calendar & analyzing Gold and EUR/USD..."):
                    res, err = api_post("/macro/refresh")
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

        if snap:
            b1, b2, b3 = st.columns(3)
            b1.metric("Market Condition", regime_badge(snap["regime"]))
            b2.metric("XAU/USD (Gold) Bias", bias_badge(snap["xauusd_bias"]))
            b3.metric("EUR/USD Bias", bias_badge(snap["eurusd_bias"]))

            with st.expander("ℹ️ What do these labels mean?"):
                st.markdown(
                    "- **🟢 Uptrend (Buyers):** News favors higher prices—focus on buy setups.\n"
                    "- **🔴 Downtrend (Sellers):** News favors lower prices—focus on sell setups.\n"
                    "- **⚪ Sideways / Balanced:** No strong push either way—range highs/lows tend to hold.\n"
                    "- **🟠 Messy / Up-Down Spikes (Volatile Chop):** Price jumps sharply both ways around big news, hitting stop-losses on both sides. Avoid chasing breakouts."
                )

            c_left, c_right = st.columns(2)
            with c_left:
                st.info(f"**💵 US Dollar & Yields Summary:**\n\n{snap['dxy_yield_context']}")
            with c_right:
                st.info(f"**🏦 Fed & ECB Interest-Rate Outlook:**\n\n{snap['central_bank_summary']}")

            st.warning(f"**🎯 Today's Trading Action Plan:** {snap['trading_guideline']}")

        st.divider()
        cal_col, form_col = st.columns([1.7, 1.0], gap="large")

        with cal_col:
            st.subheader("Upcoming High-Impact News — UTC (UAE Time)")
            if not events:
                st.caption("No upcoming events cached. Click Refresh Market Guide or pin a custom news time on the right.")
            else:
                ev_df = pd.DataFrame(events)
                ev_df["time_utc_uae"] = ev_df["scheduled_utc"].apply(fmt_utc_uae)
                ev_cols = ["id", "time_utc_uae", "currency", "impact", "event_name", "forecast", "previous", "actual"]
                st.dataframe(
                    ev_df[ev_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "id": "ID",
                        "time_utc_uae": "Time — UTC (UAE UTC+4)",
                        "currency": "CCY",
                        "impact": "Impact",
                        "event_name": "News Release",
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
            st.subheader("Add Custom News Alert")
            st.caption("Pin an exact UTC time (UAE is UTC+4) to block entries +/-30 mins around it.")
            with st.form("manual_macro_event_form", clear_on_submit=True):
                ev_name = st.text_input("News Name", placeholder="E.g., US Non-Farm Payrolls (NFP)")
                ev_ccy = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
                ev_impact = st.selectbox("Impact", ["HIGH", "MEDIUM"])
                default_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT13:30:00Z")
                ev_time = st.text_input("Scheduled UTC (13:30 UTC = 17:30 UAE)", value=default_ts)
                ev_fcst = st.text_input("Forecast (Optional)", placeholder="E.g., 180K or 0.3%")
                ev_prev = st.text_input("Previous (Optional)", placeholder="E.g., 223K or 0.2%")
                add_btn = st.form_submit_button("Save News Alert", type="primary", width="stretch")

                if add_btn:
                    if not ev_name.strip():
                        st.warning("Enter a news name.")
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
                            st.success("News alert saved!")
                            st.rerun()

# ==============================================================================
# TAB 4: ASK YOUR JOURNAL
# ==============================================================================
with tab_chat:
    st.subheader("Ask Questions About Your Trading History")
    st.caption("Ask about your win rates, Good vs. Bad Trade grades, XAU/USD vs. EUR/USD results, or The5ers drawdown buffer.")

    if "journal_chat_messages" not in st.session_state:
        st.session_state.journal_chat_messages = []

    quick_prompts = [
        "Compare my win rate and net PnL on GOOD TRADES vs. BAD TRADES.",
        "Which session and instrument makes me the most money over the last 30 days?",
        "Show any lucky wins (Bad Trade — Good Outcome) or rule violations I made recently.",
    ]
    q_cols = st.columns(3)
    for idx, qp in enumerate(quick_prompts):
        if q_cols[idx].button(qp, key=f"qp_{idx}", width="stretch"):
            st.session_state.journal_chat_messages.append({"role": "user", "content": qp})
            with st.spinner("Checking your trade history..."):
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

    user_prompt = st.chat_input("Ask anything about your Good/Bad Trades, sessions, setups, or The5ers risk...")
    if user_prompt:
        st.session_state.journal_chat_messages.append({"role": "user", "content": user_prompt})
        with st.spinner("Checking your trade history..."):
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
# TAB 5: WIN RATE & DISCIPLINE ANALYTICS
# ==============================================================================
with tab_analytics:
    metrics, m_err = api_get("/analytics/metrics", params={"lookback_days": lookback_days})
    if m_err or not metrics or metrics.get("total_trades") == 0:
        st.info("Sync and journal trades to populate session, setup, and Good vs. Bad Trade performance charts.")
    else:
        ov = metrics["overall"]
        disc_pct = metrics.get("process_discipline_pct")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", ov["trades"])
        c2.metric("Win Rate", f"{ov['win_rate_pct']}%")
        c3.metric("Net PnL", f"${ov['net_pnl_usd']:,.2f}")
        c4.metric("Average R", f"{ov['avg_r']}R" if ov["avg_r"] is not None else "N/A")
        c5.metric("Process Discipline", f"{disc_pct}%" if disc_pct is not None else "N/A")

        st.divider()
        st.subheader("⚖️ Process-Over-Outcome Breakdown (Good Trade vs. Bad Trade)")

        by_pv = metrics.get("by_process_verdict", {})
        good_stats = by_pv.get("GOOD_TRADE", {"trades": 0, "win_rate_pct": 0.0, "net_pnl_usd": 0.0, "avg_r": None})
        bad_stats = by_pv.get("BAD_TRADE", {"trades": 0, "win_rate_pct": 0.0, "net_pnl_usd": 0.0, "avg_r": None})

        pv_c1, pv_c2 = st.columns(2)
        with pv_c1:
            st.success(
                f"**🟢 GOOD TRADES (Rules & Plan Followed):** `{good_stats['trades']}` trades | "
                f"Win Rate: **`{good_stats['win_rate_pct']}%`** | Net PnL: **`${good_stats['net_pnl_usd']:+,.2f}`**"
            )
        with pv_c2:
            st.error(
                f"**🔴 BAD TRADES (Rules Broken / Impulse):** `{bad_stats['trades']}` trades | "
                f"Win Rate: **`{bad_stats['win_rate_pct']}%`** | Net PnL: **`${bad_stats['net_pnl_usd']:+,.2f}`**"
            )

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            if by_pv:
                pv_rows = []
                label_map = {
                    "GOOD_TRADE": "🟢 Good Trade (Clean Process)",
                    "BAD_TRADE": "🔴 Bad Trade (Rule Violation)",
                    "UNGRADED": "⏳ Grading...",
                }
                for key, stats in by_pv.items():
                    pv_rows.append(
                        {
                            "Process Grade": label_map.get(key, key),
                            "Trades": stats["trades"],
                            "Win Rate %": stats["win_rate_pct"],
                            "Net PnL ($)": stats["net_pnl_usd"],
                        }
                    )
                pv_df = pd.DataFrame(pv_rows)
                fig_pv = px.bar(
                    pv_df,
                    x="Process Grade",
                    y="Net PnL ($)",
                    color="Win Rate %",
                    text="Trades",
                    title="Net Profit ($) & Win Rate by Process Grade",
                )
                fig_pv.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pv, width="stretch")

        with q_col2:
            by_quad = metrics.get("by_outcome_quadrant", {})
            if by_quad:
                quad_labels = {
                    "GOOD_PROCESS_GOOD_OUTCOME": "🟢 Good Trade → Profit",
                    "GOOD_PROCESS_BAD_OUTCOME": "🔵 Good Trade → Loss (Normal)",
                    "BAD_PROCESS_GOOD_OUTCOME": "🟠 Bad Trade → Lucky Profit (Danger)",
                    "BAD_PROCESS_BAD_OUTCOME": "🔴 Bad Trade → Loss",
                }
                q_rows = [
                    {
                        "Quadrant": quad_labels.get(k, k),
                        "Trades": v["trades"],
                        "Net PnL ($)": v["net_pnl_usd"],
                    }
                    for k, v in by_quad.items()
                ]
                q_df = pd.DataFrame(q_rows)
                fig_quad = px.bar(
                    q_df,
                    x="Quadrant",
                    y="Trades",
                    color="Net PnL ($)",
                    title="4-Quadrant Process vs. Outcome Distribution",
                )
                fig_quad.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_quad, width="stretch")
            else:
                st.info("Closed trades will automatically populate the 4-Quadrant Process Matrix once graded.")

        st.divider()
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
                    title="Net Profit ($) & Win Rate by Session",
                    labels={"net_pnl_usd": "Net Profit ($)", "win_rate_pct": "Win Rate %"},
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
                    title="Net Profit ($) by Emotional State",
                    labels={"net_pnl_usd": "Net Profit ($)", "trades": "Trade Count"},
                )
                st.plotly_chart(fig_emo, width="stretch")
            else:
                st.info("Complete an entry plan or debrief to unlock Emotional State analytics.")

        if metrics.get("top_mistakes"):
            st.subheader("Top Execution Mistakes")
            st.bar_chart(pd.Series(metrics["top_mistakes"], name="Count"))

# ==============================================================================
# TAB 6: HABIT & LEAK REVIEW
# ==============================================================================
with tab_audit:
    st.subheader("Performance & Habit Review")
    st.write("Reviews your Good vs. Bad Trade grades, win rates, emotional mistakes, and The5ers risk rules.")
    if st.button("🔍 Run Full Habit & Profit Review", type="primary"):
        with st.spinner("Reviewing trade history, grades, session profits, and rule violations..."):
            audit_res, err = api_post("/analytics/audit", params={"lookback_days": lookback_days})
            if err:
                st.error(f"Review failed: {err}")
            else:
                st.markdown(audit_res["audit_report"])