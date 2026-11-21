import io
import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
from sqlalchemy.orm import Session

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import (
    Trade,
    JournalEntry,
    DailyAccountSnapshot,
    MacroRegimeSnapshot,
    DisciplineAuditLog,
)
from core.schemas import (
    Stage1EntryRequest,
    ProcessOutcomeVerdict,
    PreTradeAuditReport,
)
from connectors.mt5_sync import MT5SyncConnector
from agents.journal_agent import TradingJournalAgent
from agents.macro_agent import MacroCatalystAgent, format_utc_and_uae

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("discipline_agent")

PRIMARY_VISION_MODEL = "gemini-4-argon"
FAST_TEXT_MODEL = "gemini-3.8-flash"
SECONDARY_MODEL = "gemini-4-hydrogen"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

UAE_TZ = timezone(timedelta(hours=4))


def is_fx_weekend_closed(now_utc: Optional[datetime] = None) -> bool:
    """Return True when spot FX & XAUUSD markets are closed (Fri 22:00 UTC to Sun 22:00 UTC)."""
    dt = now_utc or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    wd = dt.weekday()  # Mon=0 ... Fri=4, Sat=5, Sun=6
    if wd == 5:
        return True
    if wd == 4 and dt.hour >= 22:
        return True
    if wd == 6 and dt.hour < 22:
        return True
    return False


class DisciplineGatekeeperAgent:
    """
    Phase 3.5 Agent (Fast Multimodal Chart Strategist + Automatic Process Trade Grader):
    1. Stage 1 In-Flight Capture: Records entry thesis, invalidation SL, emotion & chart screenshot while OPEN.
    2. Pre-Trade Chart Strategist (HTF + LTF): Uses gemini-4-argon with 20s per-attempt timeout & JPEG compression.
    3. Stage 2 Close Verdict (Automatic Trade Grading): Uses fast gemini-3.8-flash with 20s timeout so Good/Bad Trade
       verdicts complete in ~1.5 seconds and never stall.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20_000),
        )
        self.rules = load_risk_rules()

    def _infer_session(self, dt: datetime) -> str:
        utc_dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        hour = utc_dt.hour
        if 0 <= hour < 7:
            return "ASIA"
        elif 7 <= hour < 12:
            return "LONDON"
        elif 12 <= hour < 16:
            return "LONDON_NY_OVERLAP"
        elif 16 <= hour < 21:
            return "NEW_YORK"
        return "UNKNOWN"

    def _build_image_part(self, image_path: Optional[str], max_width: int = 1280) -> Optional[types.Part]:
        if not image_path:
            return None
        path_obj = Path(image_path)
        if not path_obj.exists() or not path_obj.is_file():
            logger.warning(f"Chart image path not found: {image_path}")
            return None

        try:
            with Image.open(path_obj) as img:
                rgb_img = img.convert("RGB")
                if rgb_img.width > max_width:
                    ratio = max_width / float(rgb_img.width)
                    new_height = int(float(rgb_img.height) * ratio)
                    rgb_img = rgb_img.resize((max_width, new_height), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                rgb_img.save(buf, format="JPEG", quality=85, optimize=True)
                return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
        except Exception as exc:
            logger.warning(f"PIL resize fallback to raw bytes for {image_path}: {exc}")
            ext = path_obj.suffix.lower()
            mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            return types.Part.from_bytes(data=path_obj.read_bytes(), mime_type=mime_type)

    def _call_with_fallback(
        self,
        contents,
        response_schema,
        system_instruction: Optional[str] = None,
        prefer_vision_model: bool = True,
    ):
        """
        Fast, resilient Gemini call with strict per-call timeouts and automatic fallback:
        - Vision calls: gemini-4-argon -> gemini-3.8-flash -> gemini-4-hydrogen -> gemini-3.5-flash-lite
        - Fast text grading calls: gemini-3.8-flash -> gemini-3.5-flash-lite -> gemini-4-hydrogen
        """
        if prefer_vision_model:
            attempts = [
                (PRIMARY_VISION_MODEL, 0.5),
                (FAST_TEXT_MODEL, 0.5),
                (SECONDARY_MODEL, 0.5),
                (FALLBACK_MODEL, 0.0),
            ]
        else:
            attempts = [
                (FAST_TEXT_MODEL, 0.4),
                (FALLBACK_MODEL, 0.4),
                (SECONDARY_MODEL, 0.0),
            ]

        last_err = None
        for idx, (model_name, backoff_sec) in enumerate(attempts):
            try:
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": 0.15,
                }
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                logger.info(f"Discipline Agent response generated via {model_name}.")
                return response.text
            except APIError as exc:
                last_err = exc
                if idx < len(attempts) - 1:
                    logger.warning(
                        f"APIError ({exc.code}) on {model_name} (attempt {idx + 1}/{len(attempts)}); "
                        f"falling back to {attempts[idx + 1][0]}..."
                    )
                    if backoff_sec > 0:
                        time.sleep(backoff_sec)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if idx < len(attempts) - 1:
                    logger.warning(
                        f"Timeout/network error on {model_name}: {exc}; "
                        f"failing over to {attempts[idx + 1][0]}..."
                    )
                    if backoff_sec > 0:
                        time.sleep(backoff_sec)
                    continue
                raise
        raise last_err

    def submit_stage1_entry(
        self,
        db: Session,
        mt5_position_id: int,
        payload: Stage1EntryRequest,
    ) -> dict:
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found.")

        now_utc = datetime.now(timezone.utc)
        inferred_session = self._infer_session(trade.open_time)

        entry = db.query(JournalEntry).filter(JournalEntry.trade_id == trade.id).first()
        if not entry:
            entry = JournalEntry(trade_id=trade.id)
            db.add(entry)

        entry.session = inferred_session
        entry.setup_type = payload.setup_type.strip()
        entry.timeframe = payload.timeframe.strip()
        entry.entry_thesis = payload.entry_thesis.strip()
        entry.invalidation_plan = payload.invalidation_plan.strip()
        entry.pre_trade_emotion = payload.pre_trade_emotion
        entry.macro_aligned = payload.macro_aligned
        if payload.entry_chart_path:
            entry.entry_chart_path = payload.entry_chart_path
        entry.stage1_completed_at = now_utc

        db.commit()
        logger.info(f"Stage 1 Entry Questionnaire saved for MT5 #{mt5_position_id} ({trade.symbol} {trade.direction}).")

        return {
            "mt5_position_id": mt5_position_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "stage1_completed_at": format_utc_and_uae(now_utc),
            "message": "Stage 1 Entry Thesis locked in. Process verdict will auto-generate when MT5 closes this position.",
        }

    def evaluate_closed_trade_process(self, db: Session, mt5_position_id: int) -> ProcessOutcomeVerdict:
        """
        Fast Stage 2 Process-Over-Outcome Trade Grader (~1.5s response):
        Compares MT5 execution telemetry & Stage 1 entry thesis against The5ers rules,
        grading the trade as GOOD_TRADE or BAD_TRADE regardless of whether it made or lost money.
        """
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found.")

        je = trade.journal_entry
        inferred_session = (je.session if (je and je.session) else self._infer_session(trade.open_time))

        starting_bal = float(self.rules["account"]["starting_balance"])
        max_risk_pct = float(self.rules["position_sizing"]["max_risk_per_trade_pct"])
        max_risk_usd = round(starting_bal * (max_risk_pct / 100.0), 2)

        telemetry_flags = []
        if not trade.stop_loss or trade.stop_loss <= 0:
            telemetry_flags.append("NO_STOP_LOSS_ON_MT5")
        if trade.risk_amount_usd and trade.risk_amount_usd > max_risk_usd * 1.05:
            telemetry_flags.append(
                f"EXCEEDED_1PCT_RISK (${trade.risk_amount_usd:.2f} vs ${max_risk_usd:.2f} max)"
            )
        if je and je.pre_trade_emotion in ("FOMO", "REVENGE", "BORED"):
            telemetry_flags.append(f"IMPULSE_EMOTION_AT_ENTRY ({je.pre_trade_emotion})")
        if je and je.macro_aligned is False:
            telemetry_flags.append("TRADED_AGAINST_MACRO_BIAS")

        stage1_context = (
            f"- Setup Claimed: {je.setup_type} ({je.timeframe or 'M15'})\n"
            f"- Entry Thesis: {je.entry_thesis}\n"
            f"- Invalidation Plan: {je.invalidation_plan}\n"
            f"- Pre-Trade Emotion: {je.pre_trade_emotion}\n"
            f"- Macro Aligned Claimed: {je.macro_aligned}"
            if (je and je.stage1_completed_at)
            else f"No Stage 1 pre-close questionnaire submitted. Existing notes: {je.raw_notes if (je and je.raw_notes) else 'None'}"
        )

        prompt = f"""
You are the Head Trading Discipline Coach for a The5ers High Stakes prop account.
Grade this closed trade using PROCESS OVER OUTCOME in plain, everyday English:
- **GOOD_TRADE**: Valid setup, hard Stop Loss <= 1% risk (${max_risk_usd:.2f}), calm/planned execution, and respected entry thesis — EVEN IF it lost money ('GOOD_PROCESS_BAD_OUTCOME').
- **BAD_TRADE**: Missing Stop Loss, >1% risk, FOMO/Revenge entry, or broke plan — EVEN IF it got lucky and made money ('BAD_PROCESS_GOOD_OUTCOME').

Format any mentioned times as UTC (UAE UTC+4).

CLOSED TRADE TELEMETRY:
- Ticket: #{trade.mt5_position_id} | {trade.symbol} {trade.direction} | {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Opened: {format_utc_and_uae(trade.open_time)} | Closed: {format_utc_and_uae(trade.close_time)}
- Session: {inferred_session}
- Dollar Risk: {f'${trade.risk_amount_usd:.2f}' if trade.risk_amount_usd else 'Unknown (No SL)'} (Max Allowed: ${max_risk_usd:.2f})
- Net PnL: ${trade.net_profit:+.2f} | R-Multiple: {trade.r_multiple if trade.r_multiple is not None else 'N/A'}
- Automated Rule Flags: {telemetry_flags if telemetry_flags else 'None — Clean Telemetry'}

ENTRY THESIS & NOTES:
{stage1_context}
"""

        contents: list = [prompt]
        chart_part = self._build_image_part(je.entry_chart_path if je else None)
        if chart_part is not None:
            contents.append(chart_part)

        raw_json = self._call_with_fallback(
            contents=contents,
            response_schema=ProcessOutcomeVerdict,
            prefer_vision_model=bool(chart_part is not None),
        )
        verdict = ProcessOutcomeVerdict.model_validate_json(raw_json)

        if not je:
            je = JournalEntry(trade_id=trade.id)
            db.add(je)

        je.session = verdict.session
        je.setup_type = verdict.setup_type
        je.emotional_state = verdict.emotional_state
        je.followed_plan = verdict.followed_plan
        je.mistake_tags = verdict.mistake_tags
        je.process_verdict = verdict.process_verdict
        je.outcome_quadrant = verdict.outcome_quadrant
        je.ai_feedback = verdict.ai_feedback

        db.commit()
        logger.info(
            f"Stage 2 Process Verdict saved for MT5 #{mt5_position_id}: "
            f"{verdict.process_verdict} ({verdict.outcome_quadrant}) | PnL: ${trade.net_profit:+.2f}"
        )
        return verdict

    def run_pre_trade_chart_audit(
        self,
        db: Session,
        symbol: str,
        intent_mode: str = "RECOMMEND",
        proposed_direction: str = "UNSPECIFIED",
        timeframe: str = "M15",
        htf_timeframe: Optional[str] = "H1",
        proposed_risk_pct: float = 0.5,
        setup_description: str = "",
        chart_image_path: Optional[str] = None,
        htf_chart_image_path: Optional[str] = None,
    ) -> dict:
        sym = symbol.upper().strip()
        mode = intent_mode.upper().strip()
        direction = proposed_direction.upper().strip()
        now_utc = datetime.now(timezone.utc)
        weekend_closed = is_fx_weekend_closed(now_utc)
        current_session = "WEEKEND_CLOSED" if weekend_closed else self._infer_session(now_utc)

        max_risk_pct = float(self.rules["position_sizing"]["max_risk_per_trade_pct"])
        allowed_symbols = self.rules["instruments"].get("allowed_symbols", [])
        dd_snap = (
            db.query(DailyAccountSnapshot)
            .order_by(DailyAccountSnapshot.trading_date.desc())
            .first()
        )
        dd_context = (
            {
                "equity": dd_snap.current_equity,
                "daily_dd_used_pct": dd_snap.daily_drawdown_used_pct,
                "static_dd_used_pct": dd_snap.static_drawdown_used_pct,
                "soft_lock_triggered": dd_snap.is_locked_out,
            }
            if dd_snap
            else {"status": "Not synced yet — assume starting balance"}
        )

        macro_agent = MacroCatalystAgent()
        blackout = macro_agent.check_news_blackout(db, symbol=sym)
        macro_snap = (
            db.query(MacroRegimeSnapshot)
            .order_by(MacroRegimeSnapshot.created_at.desc())
            .first()
        )
        macro_context = (
            {
                "regime": macro_snap.regime,
                "xauusd_bias": macro_snap.xauusd_bias,
                "eurusd_bias": macro_snap.eurusd_bias,
                "guideline": macro_snap.trading_guideline,
            }
            if macro_snap
            else {"status": "No macro snapshot saved yet"}
        )

        mt5_conn = MT5SyncConnector()
        session_status = mt5_conn.get_session_countdown_status(now_utc)
        if weekend_closed:
            vol_context = "Weekend Market Closed (Sat/Sun) — live tick volatility skipped; focus on HTF/LTF weekend prep."
        else:
            vol_context = "MT5 live volatility unavailable (terminal offline)."
            if mt5_conn.connect():
                try:
                    sym_vol = mt5_conn.get_symbol_volatility(sym)
                    if sym_vol:
                        vol_context = sym_vol.model_dump_json()
                finally:
                    mt5_conn.disconnect()

        journal_agent = TradingJournalAgent()
        perf_metrics = journal_agent.compute_performance_metrics(db, lookback_days=60)
        symbol_edge = perf_metrics.get("by_symbol", {}).get(sym, "No historical trades on this symbol yet.")
        session_edge = perf_metrics.get("by_session", {}).get(current_session, "No historical trades in this session yet.")

        htf_part = self._build_image_part(htf_chart_image_path)
        ltf_part = self._build_image_part(chart_image_path)
        has_any_chart = bool(htf_part is not None or ltf_part is not None)

        intent_instruction = (
            "MODE = 'RECOMMEND' (The trader is UNSURE whether to Buy or Sell). "
            "Analyze the chart(s) objectively, pick the highest-probability directional play ('BUY', 'SELL', or 'WAIT'), "
            "and recommend the exact strategy from their playbook."
            if mode == "RECOMMEND"
            else f"MODE = 'AUDIT_{mode}' (The trader wants to {mode}). Stress-test whether {mode} is valid or a trap."
        )

        prompt = f"""
You are the Senior ICT, Trendline & Price-Action Chart Strategist for a The5ers High Stakes prop trader.
Speak in clear, plain, everyday English. Always format any mentioned times as UTC (UAE UTC+4).

{intent_instruction}

TRADER REQUEST:
- Instrument: {sym} (Allowed List: {allowed_symbols})
- Intent Mode: {mode} | Proposed Direction: {direction}
- Execution Timeframe: {timeframe} | Higher Timeframe Provided: {htf_timeframe if htf_part else 'None'}
- Proposed Risk: {proposed_risk_pct}% (Hard Max Cap: {max_risk_pct}%)
- Market Open Status: {'WEEKEND CLOSED (Saturday/Sunday Prep Mode)' if weekend_closed else 'LIVE MARKET SESSION'}
- Trader's Notes / Question: \"\"\"{setup_description or 'Analyze my chart and recommend the best strategy.'}\"\"\"

LIVE SESSION, VOLATILITY & GUARDRAIL TELEMETRY:
1. Session Radar: {session_status.banner_message}
2. Live MT5 Volatility & Volume ({sym}): {vol_context}
3. The5ers Drawdown Buffer: {json.dumps(dd_context)}
4. News Blackout Status (+/-30m rule): {blackout.model_dump_json()}
5. Phase 2 Macro Radar Bias: {json.dumps(macro_context)}
6. Trader's 60-Day Edge on {sym}: {json.dumps(symbol_edge)}
7. Trader's 60-Day Edge in {current_session}: {json.dumps(session_edge)}

TRADER'S STRATEGY PLAYBOOK (Match the best fit):
- Strategy A: **ICT Liquidity Sweep + MSS + FVG/OB Reversal**
- Strategy B: **Trendline Break & Retest**
- Strategy C: **15M Impulse Continuation (RTB / RTS)**
- Strategy D: **Sit Out / Wait (`WAIT`)**

MANDATORY ANALYSIS RULES:
1. **Top-Down HTF Check (`htf_bias_read`):** Identify Premium vs. Discount dealing range and HTF draw on liquidity.
2. **Drawn-Chart Critique vs. Blank-Chart Mapping (`analysis_mode` & `trendline_ict_critique`):**
   - If user drew trendlines/boxes: `analysis_mode='DRAWN_CHART_CRITIQUE'`. Judge VALID vs. FORCED anchor and warn if a multi-touch trendline is a retail liquidity pool likely to be swept.
   - If blank chart: `analysis_mode='BLANK_CHART_MAPPING'` and map Asian/London highs/lows, FVGs, Order Blocks, and trendlines.
   - If no image: `analysis_mode='TEXT_ONLY'`.
3. **Strict 2-Confluence Minimum Rule (`confluences_count` & `verdict`):**
   - `verdict='BLOCK'` if soft-lock is True, news blackout active, risk > {max_risk_pct}%, symbol disallowed, or direction fights clear HTF/ICT reversal.
   - `verdict='PASS'` ONLY when `confluences_count >= 2` AND zero hard blocks exist AND `recommended_action` is `'BUY'` or `'SELL'`.
   - Otherwise `verdict='CAUTION'`.
"""

        contents: list = [prompt]
        if htf_part is not None:
            contents.append(f"IMAGE 1 — HIGHER TIMEFRAME ({htf_timeframe or '1H/4H'}) MAP:")
            contents.append(htf_part)
        if ltf_part is not None:
            contents.append(f"IMAGE 2 — EXECUTION TIMEFRAME ({timeframe}) CHART:")
            contents.append(ltf_part)

        raw_json = self._call_with_fallback(
            contents=contents,
            response_schema=PreTradeAuditReport,
            prefer_vision_model=has_any_chart,
        )
        audit = PreTradeAuditReport.model_validate_json(raw_json)

        log_row = DisciplineAuditLog(
            symbol=sym,
            intent_mode=mode,
            proposed_direction=direction if mode != "RECOMMEND" else audit.recommended_action,
            timeframe=timeframe,
            htf_timeframe=htf_timeframe if htf_part else None,
            proposed_risk_pct=proposed_risk_pct,
            setup_description=setup_description or "Chart Strategist Recommendation Request",
            chart_image_path=chart_image_path,
            htf_chart_image_path=htf_chart_image_path,
            verdict=audit.verdict,
            recommended_action=audit.recommended_action,
            recommended_strategy=audit.recommended_strategy,
            confluences_count=audit.confluences_count,
            confluences_found=audit.confluences_found,
            discipline_score=audit.discipline_score,
            htf_bias_read=audit.htf_bias_read,
            chart_structure_read=audit.chart_structure_read,
            trendline_ict_critique=audit.trendline_ict_critique,
            trigger_to_wait_for=audit.trigger_to_wait_for,
            invalidation_sl_zone=audit.invalidation_sl_zone,
            rule_violations=audit.rule_violations,
            coaching_verdict=audit.coaching_verdict,
        )
        db.add(log_row)
        db.commit()
        db.refresh(log_row)

        return {
            "audit_id": log_row.id,
            "created_at": format_utc_and_uae(log_row.created_at),
            "report": audit.model_dump(),
        }