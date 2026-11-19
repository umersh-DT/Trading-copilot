import os
import json
import time
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sqlalchemy.orm import Session, joinedload

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import Trade, JournalEntry, DailyAccountSnapshot
from core.schemas import ParsedJournalNote, DebriefTurnResponse, ChatMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("journal_agent")

PRIMARY_MODEL_ID = "gemini-4-hydrogen"
SECONDARY_MODEL_ID = "gemini-3.8-flash"
FALLBACK_MODEL_ID = "gemini-3.5-flash-lite"
UAE_TZ = timezone(timedelta(hours=4))


def format_utc_and_uae(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    uae_dt = utc_dt.astimezone(UAE_TZ)
    return f"{utc_dt.strftime('%Y-%m-%d %H:%M')} UTC ({uae_dt.strftime('%H:%M')} UAE)"


class TradingJournalAgent:
    """Phase 1.5 + Phase 3 Analytics Agent: Debrief coach, process-verdict tracker & fast journal analyst."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(api_key=api_key)
        self.rules = load_risk_rules()

    def _infer_session_hint(self, open_time: datetime) -> str:
        utc_dt = open_time.astimezone(timezone.utc) if open_time.tzinfo else open_time.replace(tzinfo=timezone.utc)
        uae_dt = utc_dt.astimezone(UAE_TZ)
        hour = utc_dt.hour
        time_label = f"{utc_dt.strftime('%H:%M')} UTC ({uae_dt.strftime('%H:%M')} UAE)"
        if 0 <= hour < 7:
            return f"{time_label} (likely ASIA)"
        elif 7 <= hour < 12:
            return f"{time_label} (likely LONDON)"
        elif 12 <= hour < 16:
            return f"{time_label} (likely LONDON_NY_OVERLAP)"
        elif 16 <= hour < 21:
            return f"{time_label} (likely NEW_YORK)"
        return f"{time_label} (off-hours / late NY)"

    def _call_with_fallback(
        self,
        contents,
        system_instruction: Optional[str] = None,
        response_schema=None,
        thinking_level: str = "low",
    ):
        """Resilient Gemini call with automatic 503/429 retry and model fallback."""
        attempts = [
            (PRIMARY_MODEL_ID, 1.0),
            (SECONDARY_MODEL_ID, 1.0),
            (FALLBACK_MODEL_ID, 0.0),
        ]
        last_err = None

        for idx, (model_name, backoff_sec) in enumerate(attempts):
            try:
                config_kwargs = {}
                if system_instruction:
                    config_kwargs["system_instruction"] = system_instruction
                if response_schema is not None:
                    config_kwargs["response_mime_type"] = "application/json"
                    config_kwargs["response_schema"] = response_schema
                if model_name == PRIMARY_MODEL_ID:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

                return self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except APIError as exc:
                last_err = exc
                if exc.code in (429, 500, 503) and idx < len(attempts) - 1:
                    logger.warning(f"HTTP {exc.code} on {model_name}; switching to {attempts[idx + 1][0]}...")
                    time.sleep(backoff_sec)
                    continue
                raise
        raise last_err

    def _save_parsed_entry(self, db: Session, trade: Trade, raw_notes: str, parsed: ParsedJournalNote) -> None:
        """Persist or update a finalized JournalEntry in SQLite."""
        entry = db.query(JournalEntry).filter(JournalEntry.trade_id == trade.id).first()
        if entry:
            entry.session = parsed.session
            entry.setup_type = parsed.setup_type
            entry.timeframe = parsed.timeframe
            entry.raw_notes = raw_notes
            entry.emotional_state = parsed.emotional_state
            entry.followed_plan = parsed.followed_plan
            entry.mistake_tags = parsed.mistake_tags
            entry.ai_feedback = parsed.ai_feedback
        else:
            entry = JournalEntry(
                trade_id=trade.id,
                session=parsed.session,
                setup_type=parsed.setup_type,
                timeframe=parsed.timeframe,
                raw_notes=raw_notes,
                emotional_state=parsed.emotional_state,
                followed_plan=parsed.followed_plan,
                mistake_tags=parsed.mistake_tags,
                ai_feedback=parsed.ai_feedback,
            )
            db.add(entry)
        db.commit()

    def log_trade_note(self, db: Session, mt5_position_id: int, raw_notes: str) -> ParsedJournalNote:
        """One-shot structured extraction."""
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found.")

        session_hint = self._infer_session_hint(trade.open_time)
        prompt = f"""
You are a strict trading coach auditing a trade on a The5ers High Stakes prop account.
Extract the structured journal classification from the telemetry and trader note below.

TRADE TELEMETRY:
- Symbol: {trade.symbol} | Direction: {trade.direction} | Volume: {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Open Time: {session_hint}
- Net PnL: ${trade.net_profit:.2f} | Realized R-Multiple: {trade.r_multiple}

TRADER NOTE:
\"\"\"{raw_notes}\"\"\"
"""
        response = self._call_with_fallback(
            contents=prompt,
            response_schema=ParsedJournalNote,
            thinking_level="low",
        )
        parsed = ParsedJournalNote.model_validate_json(response.text)
        self._save_parsed_entry(db, trade, raw_notes, parsed)
        return parsed

    def run_debrief_turn(
        self,
        db: Session,
        mt5_position_id: int,
        conversation: list[ChatMessage],
        force_finalize: bool = False,
    ) -> DebriefTurnResponse:
        """Interactive multi-turn post-trade interview."""
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found.")

        session_hint = self._infer_session_hint(trade.open_time)
        max_risk_pct = self.rules["position_sizing"]["max_risk_per_trade_pct"]
        starting_bal = self.rules["account"]["starting_balance"]
        max_risk_usd = starting_bal * (max_risk_pct / 100.0)

        risk_flag = "Within 1% max risk limit."
        if trade.risk_amount_usd and trade.risk_amount_usd > max_risk_usd * 1.05:
            risk_flag = f"VIOLATION: Initial risk ${trade.risk_amount_usd:.2f} exceeded {max_risk_pct}% (${max_risk_usd:.2f}) cap."
        elif not trade.stop_loss:
            risk_flag = "WARNING: No Stop Loss recorded on MT5 deal comment."

        user_turns = [m for m in conversation if m.role == "user"]
        must_complete = force_finalize or len(user_turns) >= 2

        system_instruction = f"""
You are a sharp, plain-spoken trading coach conducting a post-trade debrief for a The5ers High Stakes trader.
Uncover the exact setup trigger, execution discipline, and emotional state behind this trade in 1-2 short turns.

TRADE TELEMETRY:
- MT5 Position ID: #{trade.mt5_position_id}
- Symbol: {trade.symbol} | Direction: {trade.direction} | Volume: {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Open Time: {session_hint}
- Net PnL: ${trade.net_profit:+.2f} | Realized R-Multiple: {trade.r_multiple if trade.r_multiple is not None else 'N/A'}
- Risk Audit: {risk_flag}

RULES FOR THIS TURN:
1. If the conversation has 0 user messages, set status='ASKING_FOLLOWUP' and ask 2 concise, plain-English questions based on the telemetry.
2. If the trader has answered clearly OR must_complete={must_complete}, set status='COMPLETED', populate 'final_entry' completely, and write a crisp 2-sentence coaching takeaway in 'coach_message'.
3. Only set status='ASKING_FOLLOWUP' after 1 user turn if the trader gave a one-word/vague answer.
"""

        contents = []
        if not conversation:
            contents.append("Start the post-trade debrief for this MT5 execution.")
        else:
            for msg in conversation:
                contents.append(
                    types.Content(
                        role=msg.role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        response = self._call_with_fallback(
            contents=contents,
            system_instruction=system_instruction,
            response_schema=DebriefTurnResponse,
            thinking_level="low",
        )

        result = DebriefTurnResponse.model_validate_json(response.text)

        if result.status == "COMPLETED" and result.final_entry is not None:
            lines = []
            for m in conversation:
                speaker = "Coach" if m.role == "model" else "Trader"
                lines.append(f"{speaker}: {m.content}")
            transcript = "\n".join(lines)
            self._save_parsed_entry(db, trade, raw_notes=transcript, parsed=result.final_entry)
            logger.info(f"Completed interactive debrief for MT5 #{mt5_position_id}")

        return result

    def compute_performance_metrics(self, db: Session, lookback_days: int = 30) -> dict:
        """
        Compute deterministic win rates, expectancy, R-multiples, mistake frequencies,
        and GOOD_TRADE vs BAD_TRADE process-over-outcome splits.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        trades = (
            db.query(Trade)
            .options(joinedload(Trade.journal_entry))
            .filter(Trade.status == "CLOSED", Trade.close_time >= cutoff)
            .order_by(Trade.close_time.asc())
            .all()
        )

        if not trades:
            return {"total_trades": 0, "message": "No closed trades found in the selected period."}

        def _summarize_bucket(trade_list: list[Trade]) -> dict:
            count = len(trade_list)
            wins = [t for t in trade_list if t.net_profit > 0]
            losses = [t for t in trade_list if t.net_profit < 0]
            net_pnl = round(sum(t.net_profit for t in trade_list), 2)
            win_rate = round((len(wins) / count) * 100.0, 1) if count else 0.0
            r_vals = [t.r_multiple for t in trade_list if t.r_multiple is not None]
            avg_r = round(sum(r_vals) / len(r_vals), 2) if r_vals else None
            avg_win = round(sum(t.net_profit for t in wins) / len(wins), 2) if wins else 0.0
            avg_loss = round(sum(t.net_profit for t in losses) / len(losses), 2) if losses else 0.0
            return {
                "trades": count,
                "win_rate_pct": win_rate,
                "net_pnl_usd": net_pnl,
                "avg_r": avg_r,
                "avg_win_usd": avg_win,
                "avg_loss_usd": avg_loss,
            }

        by_symbol = defaultdict(list)
        by_session = defaultdict(list)
        by_setup = defaultdict(list)
        by_emotion = defaultdict(list)
        by_process_verdict = defaultdict(list)
        by_outcome_quadrant = defaultdict(list)
        mistake_counter = Counter()
        plan_followed_trades = []
        plan_broken_trades = []

        for t in trades:
            by_symbol[t.symbol].append(t)
            if t.journal_entry:
                je = t.journal_entry
                by_session[je.session or "UNKNOWN"].append(t)
                by_setup[je.setup_type or "Untagged"].append(t)
                by_emotion[je.emotional_state or je.pre_trade_emotion or "NEUTRAL"].append(t)

                verdict_key = je.process_verdict if je.process_verdict in ("GOOD_TRADE", "BAD_TRADE") else "UNGRADED"
                by_process_verdict[verdict_key].append(t)

                if je.outcome_quadrant:
                    by_outcome_quadrant[je.outcome_quadrant].append(t)

                if je.mistake_tags:
                    mistake_counter.update(je.mistake_tags)
                if je.followed_plan:
                    plan_followed_trades.append(t)
                else:
                    plan_broken_trades.append(t)
            else:
                by_session["UNJOURNALED"].append(t)
                by_process_verdict["UNGRADED"].append(t)

        # Compute overall Process Discipline Score (% of graded trades that were GOOD_TRADE)
        good_count = len(by_process_verdict.get("GOOD_TRADE", []))
        bad_count = len(by_process_verdict.get("BAD_TRADE", []))
        graded_total = good_count + bad_count
        process_discipline_pct = round((good_count / graded_total) * 100.0, 1) if graded_total > 0 else None

        return {
            "lookback_days": lookback_days,
            "total_trades": len(trades),
            "process_discipline_pct": process_discipline_pct,
            "overall": _summarize_bucket(trades),
            "by_process_verdict": {k: _summarize_bucket(v) for k, v in by_process_verdict.items()},
            "by_outcome_quadrant": {k: _summarize_bucket(v) for k, v in by_outcome_quadrant.items()},
            "discipline_split": {
                "followed_plan": _summarize_bucket(plan_followed_trades),
                "broke_plan": _summarize_bucket(plan_broken_trades),
            },
            "by_symbol": {k: _summarize_bucket(v) for k, v in by_symbol.items()},
            "by_session": {k: _summarize_bucket(v) for k, v in by_session.items()},
            "by_setup": {k: _summarize_bucket(v) for k, v in by_setup.items()},
            "by_emotion": {k: _summarize_bucket(v) for k, v in by_emotion.items()},
            "top_mistakes": dict(mistake_counter.most_common(10)),
        }

    def chat_with_journal(self, db: Session, messages: list[ChatMessage], lookback_days: int = 60) -> str:
        """
        Fast single-pass Q&A over the trader's SQLite journal including GOOD_TRADE vs BAD_TRADE
        process verdicts, entry theses, and UTC + UAE (UTC+4) timestamps.
        """
        metrics = self.compute_performance_metrics(db, lookback_days=lookback_days)

        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        recent_rows = (
            db.query(Trade)
            .options(joinedload(Trade.journal_entry))
            .filter(Trade.status == "CLOSED", Trade.close_time >= cutoff)
            .order_by(Trade.close_time.desc())
            .limit(75)
            .all()
        )

        compact_trades = []
        for t in recent_rows:
            je = t.journal_entry
            compact_trades.append(
                {
                    "id": t.mt5_position_id,
                    "sym": t.symbol,
                    "dir": t.direction,
                    "lots": t.volume,
                    "pnl": t.net_profit,
                    "r": t.r_multiple,
                    "closed": format_utc_and_uae(t.close_time),
                    "sess": je.session if je else "UNJOURNALED",
                    "setup": je.setup_type if je else None,
                    "thesis": je.entry_thesis if je else None,
                    "emo": (je.emotional_state or je.pre_trade_emotion) if je else None,
                    "plan_ok": je.followed_plan if je else None,
                    "process_verdict": je.process_verdict if je else "UNGRADED",
                    "quadrant": je.outcome_quadrant if je else None,
                    "mistakes": je.mistake_tags if je else [],
                }
            )

        snap = (
            db.query(DailyAccountSnapshot)
            .order_by(DailyAccountSnapshot.trading_date.desc())
            .first()
        )
        risk_context = (
            {
                "date": snap.trading_date,
                "equity": snap.current_equity,
                "daily_dd_used_pct": snap.daily_drawdown_used_pct,
                "static_dd_used_pct": snap.static_drawdown_used_pct,
                "locked_out": snap.is_locked_out,
            }
            if snap
            else "No MT5 snapshot synced yet."
        )

        system_instruction = f"""
You are the trader's performance coach for a The5ers High Stakes prop account.
Answer clearly in simple, everyday English using ONLY the verified database snapshot below.
Pay special attention to **Process Over Outcome** ('GOOD_TRADE' vs 'BAD_TRADE' verdicts and the 4 quadrants:
GOOD_PROCESS_GOOD_OUTCOME, GOOD_PROCESS_BAD_OUTCOME, BAD_PROCESS_GOOD_OUTCOME [lucky/dangerous], BAD_PROCESS_BAD_OUTCOME).
Cite exact MT5 trade IDs, dollar amounts, win rates, and R-multiples. Whenever you mention a time, format it as UTC (UAE UTC+4).

THE5ERS RISK STATUS:
{json.dumps(risk_context)}

AGGREGATED METRICS & PROCESS VERDICT BREAKDOWN (LAST {lookback_days} DAYS):
{json.dumps(metrics)}

RECENT CLOSED TRADES (UP TO 75):
{json.dumps(compact_trades)}
"""

        contents = [
            types.Content(
                role=m.role,
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in messages
        ]

        response = self._call_with_fallback(
            contents=contents,
            system_instruction=system_instruction,
            thinking_level="low",
        )
        return response.text or "No response generated."

    def run_leak_audit(self, db: Session, lookback_days: int = 30) -> str:
        """Deep diagnostic report across all recent trades including GOOD_TRADE vs BAD_TRADE process discipline."""
        metrics = self.compute_performance_metrics(db, lookback_days=lookback_days)
        if metrics.get("total_trades") == 0:
            return "No closed trades available yet. Sync MT5 trades and log notes before running an audit."

        prompt = f"""
You are the Head Risk & Performance Coach overseeing a trader on a The5ers High Stakes prop account.
Review the trading performance metrics below—including the GOOD_TRADE vs BAD_TRADE process splits—and produce a clear, plain-English diagnostic report.

ACCOUNT RISK RULES:
{json.dumps(self.rules, indent=2)}

PERFORMANCE METRICS (LAST {lookback_days} DAYS):
{json.dumps(metrics, indent=2)}

Deliver your audit in three structured sections:
1. **What Is Working (Verified Edge & Good Process):** Identify the highest-profit symbol, session, and setup combinations, plus win rate on GOOD_TRADE executions.
2. **Where Money & Discipline Are Leaking:** Pinpoint BAD_TRADE habits (especially lucky wins on BAD_PROCESS_GOOD_OUTCOME), losing sessions, emotional triggers (FOMO/Revenge), or rule violations draining profit.
3. **3 Simple Rules for Next Week:** Give 3 concrete, easy-to-follow rules to enforce before entering any trade.
"""
        response = self._call_with_fallback(
            contents=prompt,
            thinking_level="medium",
        )
        return response.text