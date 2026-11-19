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
from agents.journal_agent import TradingJournalAgent
from agents.macro_agent import MacroCatalystAgent, format_utc_and_uae

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("discipline_agent")

PRIMARY_VISION_MODEL = "gemini-4-argon"
SECONDARY_MODEL = "gemini-4-hydrogen"
TERTIARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

UAE_TZ = timezone(timedelta(hours=4))


class DisciplineGatekeeperAgent:
    """
    Phase 3 Agent (Multimodal Vision + Process Discipline):
    1. Stage 1 In-Flight Capture: Records entry thesis, invalidation SL, emotion & chart screenshot while OPEN.
    2. Pre-Trade / Live Chart Audit: Audits TradingView chart + risk + macro + journal edge (PASS / CAUTION / BLOCK).
    3. Stage 2 Close Verdict: Grades closed trades as GOOD_TRADE or BAD_TRADE regardless of profit or loss.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(api_key=api_key)
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

    def _build_image_part(self, image_path: Optional[str]) -> Optional[types.Part]:
        """Load a local PNG/JPG TradingView chart screenshot into a Gemini Part."""
        if not image_path:
            return None
        path_obj = Path(image_path)
        if not path_obj.exists() or not path_obj.is_file():
            logger.warning(f"Chart image path not found: {image_path}")
            return None

        ext = path_obj.suffix.lower()
        mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        image_bytes = path_obj.read_bytes()
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def _call_with_fallback(
        self,
        contents,
        response_schema,
        system_instruction: Optional[str] = None,
        prefer_vision_model: bool = True,
    ):
        """
        Resilient Gemini call with automatic 503/429 retry and model fallback:
        gemini-4-argon -> gemini-4-hydrogen -> gemini-3.8-flash -> gemini-3.5-flash-lite
        """
        if prefer_vision_model:
            attempts = [
                (PRIMARY_VISION_MODEL, 1.5),
                (SECONDARY_MODEL, 1.5),
                (TERTIARY_MODEL, 1.0),
                (FALLBACK_MODEL, 0.0),
            ]
        else:
            attempts = [
                (SECONDARY_MODEL, 1.5),
                (TERTIARY_MODEL, 1.0),
                (FALLBACK_MODEL, 0.0),
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
                if model_name in (PRIMARY_VISION_MODEL, SECONDARY_MODEL):
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                logger.info(f"Discipline Gatekeeper response generated via {model_name}.")
                return response.text
            except APIError as exc:
                last_err = exc
                if exc.code in (429, 500, 503) and idx < len(attempts) - 1:
                    logger.warning(
                        f"HTTP {exc.code} on {model_name} (attempt {idx + 1}/{len(attempts)}); "
                        f"falling back to {attempts[idx + 1][0]} in {backoff_sec}s..."
                    )
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
        """
        Save the trader's in-flight Entry Thesis Questionnaire while the MT5 position is still OPEN.
        """
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
        Stage 2 Process-Over-Outcome Verdict:
        Compares the promised Stage 1 entry thesis & risk rules against actual MT5 execution upon close,
        grading the trade as GOOD_TRADE or BAD_TRADE regardless of whether it ended in profit or loss.
        """
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found.")

        je = trade.journal_entry
        inferred_session = (je.session if (je and je.session) else self._infer_session(trade.open_time))

        starting_bal = float(self.rules["account"]["starting_balance"])
        max_risk_pct = float(self.rules["position_sizing"]["max_risk_per_trade_pct"])
        max_risk_usd = round(starting_bal * (max_risk_pct / 100.0), 2)

        # Objective telemetry checks
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
            else "No Stage 1 pre-close questionnaire was submitted before this trade closed."
        )

        prompt = f"""
You are the Head Trading Discipline Coach for a The5ers High Stakes prop account.
Your single most important principle is PROCESS OVER OUTCOME:
- A trade that followed a valid setup, respected a hard Stop Loss <= 1% risk (${max_risk_usd:.2f}), avoided news blackouts, and honored its entry thesis is a **GOOD_TRADE** — even if it hit Stop Loss and lost money ('GOOD_PROCESS_BAD_OUTCOME').
- A trade that had no Stop Loss, exceeded 1% risk, chased FOMO/Revenge, or ignored the entry plan is a **BAD_TRADE** — even if it got lucky and closed in profit ('BAD_PROCESS_GOOD_OUTCOME').

Evaluate this closed MT5 execution in plain, everyday English. Always format any mentioned times as UTC (UAE UTC+4).

CLOSED TRADE TELEMETRY:
- MT5 Ticket: #{trade.mt5_position_id} | Symbol: {trade.symbol} | Direction: {trade.direction} | Volume: {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Opened: {format_utc_and_uae(trade.open_time)} | Closed: {format_utc_and_uae(trade.close_time)}
- Inferred Session: {inferred_session}
- Initial Dollar Risk: {f'${trade.risk_amount_usd:.2f}' if trade.risk_amount_usd else 'Unknown (No SL)'} (Max Allowed: ${max_risk_usd:.2f})
- Final Net PnL: ${trade.net_profit:+.2f} | Realized R-Multiple: {trade.r_multiple if trade.r_multiple is not None else 'N/A'}
- Automated Rule Flags: {telemetry_flags if telemetry_flags else 'None — Clean Telemetry'}

STAGE 1 IN-FLIGHT ENTRY QUESTIONNAIRE:
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
        proposed_direction: str,
        timeframe: str,
        proposed_risk_pct: float,
        setup_description: str,
        chart_image_path: Optional[str] = None,
    ) -> dict:
        """
        Pre-Trade Multimodal Gatekeeper (gemini-4-argon):
        Cross-examines TradingView chart screenshot + The5ers drawdown buffer + Phase 2 Macro Radar
        + +/-30m news blackout + Phase 1.5 historical session/symbol win rate before entry.
        """
        sym = symbol.upper().strip()
        direction = proposed_direction.upper().strip()
        now_utc = datetime.now(timezone.utc)
        current_session = self._infer_session(now_utc)

        # 1. Gather The5ers Risk Rules & Latest Drawdown Snapshot
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

        # 2. Gather Phase 2 Macro Bias & Active News Blackout
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

        # 3. Gather Historical Edge for this Symbol & Session
        journal_agent = TradingJournalAgent()
        perf_metrics = journal_agent.compute_performance_metrics(db, lookback_days=60)
        symbol_edge = perf_metrics.get("by_symbol", {}).get(sym, "No historical trades on this symbol yet.")
        session_edge = perf_metrics.get("by_session", {}).get(current_session, "No historical trades in this session yet.")

        prompt = f"""
You are the Pre-Trade Discipline Gatekeeper for a The5ers High Stakes prop trading account.
Audit this proposed trade setup BEFORE the trader clicks Buy/Sell on MetaTrader 5.
Speak in clear, plain, everyday English. Format any mentioned times as UTC (UAE UTC+4).

PROPOSED TRADE:
- Instrument: {sym} (Allowed List: {allowed_symbols})
- Direction: {direction} | Timeframe: {timeframe}
- Proposed Risk: {proposed_risk_pct}% (Hard Max Cap: {max_risk_pct}%)
- Current Time & Session: {format_utc_and_uae(now_utc)} ({current_session})
- Trader's Setup Description: \"\"\"{setup_description}\"\"\"

LIVE GUARDRAIL TELEMETRY:
1. The5ers Drawdown Buffer: {json.dumps(dd_context)}
2. News Blackout Status (+/-30m rule): {blackout.model_dump_json()}
3. Phase 2 Macro Radar Bias: {json.dumps(macro_context)}
4. Trader's 60-Day Edge on {sym}: {json.dumps(symbol_edge)}
5. Trader's 60-Day Edge in {current_session}: {json.dumps(session_edge)}

STRICT DECISION RULES:
- Set verdict='BLOCK' if ANY of these are true:
  (a) Drawdown soft_lock_triggered is True,
  (b) News blackout is_blackout_active is True,
  (c) proposed_risk_pct > {max_risk_pct},
  (d) Instrument is not in {allowed_symbols},
  (e) A TradingView chart screenshot is attached and clearly shows no Stop Loss structure, chasing an extended candle into HTF resistance/support, or invalid setup geometry.
- Set verdict='CAUTION' if hard rules pass, but the trade goes counter to the Macro Radar directional bias, sits in VOLATILE_CHOP, or is in a historically losing session for this trader.
- Set verdict='PASS' only when risk, news buffer, macro context, and chart structure align cleanly.
"""

        contents: list = [prompt]
        chart_part = self._build_image_part(chart_image_path)
        if chart_part is not None:
            contents.append(chart_part)
        else:
            contents.append("Note: No TradingView chart screenshot was uploaded for this check; evaluate based on setup description and live guardrails.")

        raw_json = self._call_with_fallback(
            contents=contents,
            response_schema=PreTradeAuditReport,
            prefer_vision_model=True,
        )
        audit = PreTradeAuditReport.model_validate_json(raw_json)

        log_row = DisciplineAuditLog(
            symbol=sym,
            proposed_direction=direction,
            timeframe=timeframe,
            proposed_risk_pct=proposed_risk_pct,
            setup_description=setup_description,
            chart_image_path=chart_image_path,
            verdict=audit.verdict,
            discipline_score=audit.discipline_score,
            chart_structure_read=audit.chart_structure_read,
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


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        agent = DisciplineGatekeeperAgent()
        res = agent.run_pre_trade_chart_audit(
            db=db,
            symbol="XAUUSD",
            proposed_direction="BUY",
            timeframe="M15",
            proposed_risk_pct=0.5,
            setup_description="London session liquidity sweep below Asian low followed by 15m bullish engulfing and FVG retest.",
        )
        print("\n=== Phase 3 Pre-Trade Gatekeeper Test ===")
        print(json.dumps(res, indent=2))
    finally:
        db.close()