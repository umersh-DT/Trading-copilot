import os
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import Trade, JournalEntry
from core.schemas import ParsedJournalNote

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("journal_agent")

MODEL_ID = "gemini-4-hydrogen"


class TradingJournalAgent:
    """Phase 1 Agent: Parses post-trade notes, computes edge metrics, and diagnoses behavioral leaks."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(api_key=api_key)
        self.rules = load_risk_rules()

    def _infer_session_hint(self, open_time: datetime) -> str:
        """Provide UTC timestamp context to help Gemini classify the session accurately."""
        utc_dt = open_time.astimezone(timezone.utc) if open_time.tzinfo else open_time.replace(tzinfo=timezone.utc)
        hour = utc_dt.hour
        if 0 <= hour < 7:
            return f"{utc_dt.strftime('%H:%M')} UTC (likely ASIA)"
        elif 7 <= hour < 12:
            return f"{utc_dt.strftime('%H:%M')} UTC (likely LONDON)"
        elif 12 <= hour < 16:
            return f"{utc_dt.strftime('%H:%M')} UTC (likely LONDON_NY_OVERLAP)"
        elif 16 <= hour < 21:
            return f"{utc_dt.strftime('%H:%M')} UTC (likely NEW_YORK)"
        return f"{utc_dt.strftime('%H:%M')} UTC (off-hours / late NY)"

    def log_trade_note(self, db: Session, mt5_position_id: int, raw_notes: str) -> ParsedJournalNote:
        """
        Parse raw trader notes into a structured ParsedJournalNote using gemini-4-hydrogen
        and upsert the resulting JournalEntry into SQLite.
        """
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found in database.")

        session_hint = self._infer_session_hint(trade.open_time)
        max_risk_pct = self.rules["position_sizing"]["max_risk_per_trade_pct"]
        starting_bal = self.rules["account"]["starting_balance"]
        max_risk_usd = starting_bal * (max_risk_pct / 100.0)

        risk_violation_context = ""
        if trade.risk_amount_usd and trade.risk_amount_usd > max_risk_usd * 1.05:
            risk_violation_context = (
                f"ALERT: Initial trade risk was ${trade.risk_amount_usd:.2f}, exceeding the "
                f"max {max_risk_pct}% (${max_risk_usd:.2f}) per-trade limit."
            )

        prompt = f"""
You are a strict institutional trading performance coach auditing a trade on a The5ers High Stakes prop account.
Analyze the trade execution telemetry and the trader's post-trade note below, then extract the structured journal classification.

TRADE TELEMETRY:
- Symbol: {trade.symbol}
- Direction: {trade.direction}
- Volume: {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Open Time: {session_hint}
- Net PnL: ${trade.net_profit:.2f}
- Realized R-Multiple: {trade.r_multiple if trade.r_multiple is not None else 'Unknown (No SL recorded)'}
- {risk_violation_context}

TRADER'S RAW NOTE:
\"\"\"{raw_notes}\"\"\"
"""

        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ParsedJournalNote,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                temperature=0.1,
            ),
        )

        parsed = ParsedJournalNote.model_validate_json(response.text)

        # Upsert JournalEntry in SQLite
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
        logger.info(f"Logged journal entry for MT5 Position #{mt5_position_id} | Setup: {parsed.setup_type}")
        return parsed

    def compute_performance_metrics(self, db: Session, lookback_days: int = 30) -> dict:
        """Compute deterministic win rates, expectancy, R-multiples, and mistake frequencies."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        trades = (
            db.query(Trade)
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
        mistake_counter = Counter()
        plan_followed_trades = []
        plan_broken_trades = []

        for t in trades:
            by_symbol[t.symbol].append(t)
            if t.journal_entry:
                je = t.journal_entry
                by_session[je.session or "UNKNOWN"].append(t)
                by_setup[je.setup_type or "Untagged"].append(t)
                by_emotion[je.emotional_state or "NEUTRAL"].append(t)
                if je.mistake_tags:
                    mistake_counter.update(je.mistake_tags)
                if je.followed_plan:
                    plan_followed_trades.append(t)
                else:
                    plan_broken_trades.append(t)
            else:
                by_session["UNJOURNALED"].append(t)

        return {
            "lookback_days": lookback_days,
            "overall": _summarize_bucket(trades),
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

    def run_leak_audit(self, db: Session, lookback_days: int = 30) -> str:
        """
        Use gemini-4-hydrogen with high thinking depth to diagnose edge decay,
        session leaks, and psychological triggers hurting The5ers account progression.
        """
        metrics = self.compute_performance_metrics(db, lookback_days=lookback_days)
        if metrics.get("total_trades") == 0:
            return "No closed trades available yet. Sync MT5 trades and log notes before running an audit."

        prompt = f"""
You are the Head Risk & Performance Analyst overseeing a trader on a The5ers High Stakes prop account.
Review the deterministic trading performance metrics below and produce a concise, high-impact diagnostic report.

ACCOUNT RISK RULES:
{json.dumps(self.rules, indent=2)}

PERFORMANCE METRICS (LAST {lookback_days} DAYS):
{json.dumps(metrics, indent=2)}

Deliver your audit in three structured sections:
1. **Verified Edge (What is Working):** Identify the highest-expectancy symbol, session, and setup combinations.
2. **Primary Capital Leaks (Where Money is Lost):** Pinpoint negative-expectancy sessions, emotional triggers (FOMO/Revenge), or rule violations draining PnL.
3. **Actionable Rule Adjustments for Next Week:** Give 3 concrete, enforceable rules to feed into the Phase 3 Discipline Gatekeeper.
"""

        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                temperature=0.2,
            ),
        )
        return response.text


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        agent = TradingJournalAgent()
        latest_trade = db.query(Trade).order_by(Trade.close_time.desc()).first()
        if latest_trade:
            sample_note = (
                "Entered short on London session liquidity sweep above Asian high on M15. "
                "Waited for displacement candle, kept risk tight at 0.5%, and held cleanly to target."
            )
            result = agent.log_trade_note(db, latest_trade.mt5_position_id, sample_note)
            print("\n=== Parsed Journal Entry ===")
            print(result.model_dump_json(indent=2))
            print("\n=== Performance Metrics ===")
            print(json.dumps(agent.compute_performance_metrics(db), indent=2))
        else:
            print("Database has 0 trades. Run `python -m connectors.mt5_sync` first.")
    finally:
        db.close()