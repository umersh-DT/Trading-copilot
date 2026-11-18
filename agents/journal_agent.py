import os
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy.orm import Session, joinedload

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import Trade, JournalEntry, DailyAccountSnapshot
from core.schemas import ParsedJournalNote, DebriefTurnResponse, ChatMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("journal_agent")

MODEL_ID = "gemini-4-hydrogen"


class TradingJournalAgent:
    """Phase 1.5 Agent: Interactive post-trade debrief coach + tool-calling journal analyst."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(api_key=api_key)
        self.rules = load_risk_rules()

    def _infer_session_hint(self, open_time: datetime) -> str:
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
        """One-shot structured extraction (kept for quick logging or batch imports)."""
        trade = db.query(Trade).filter(Trade.mt5_position_id == mt5_position_id).first()
        if not trade:
            raise ValueError(f"Trade with mt5_position_id={mt5_position_id} not found.")

        session_hint = self._infer_session_hint(trade.open_time)
        prompt = f"""
You are a strict institutional trading coach auditing a trade on a The5ers High Stakes prop account.
Extract the structured journal classification from the telemetry and trader note below.

TRADE TELEMETRY:
- Symbol: {trade.symbol} | Direction: {trade.direction} | Volume: {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Open Time: {session_hint}
- Net PnL: ${trade.net_profit:.2f} | Realized R-Multiple: {trade.r_multiple}

TRADER NOTE:
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
        self._save_parsed_entry(db, trade, raw_notes, parsed)
        return parsed

    def run_debrief_turn(
        self,
        db: Session,
        mt5_position_id: int,
        conversation: list[ChatMessage],
        force_finalize: bool = False,
    ) -> DebriefTurnResponse:
        """
        Interactive multi-turn post-trade interview.
        - Turn 0 (empty conversation): Coach inspects MT5 telemetry and asks 1-2 targeted opening questions.
        - Subsequent turns: Evaluates trader answers; asks 1 follow-up if vague, or finalizes & saves JournalEntry.
        """
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
You are a sharp, no-nonsense trading performance coach conducting a post-trade debrief for a The5ers High Stakes trader.
Your goal is to uncover the exact setup trigger, execution discipline, and emotional state behind this trade in 1-2 short turns—never interrogate endlessly.

TRADE TELEMETRY:
- MT5 Position ID: #{trade.mt5_position_id}
- Symbol: {trade.symbol} | Direction: {trade.direction} | Volume: {trade.volume} lots
- Entry: {trade.entry_price} | Exit: {trade.exit_price} | Stop Loss: {trade.stop_loss}
- Open Time: {session_hint}
- Net PnL: ${trade.net_profit:+.2f} | Realized R-Multiple: {trade.r_multiple if trade.r_multiple is not None else 'N/A'}
- Risk Audit: {risk_flag}

RULES FOR THIS TURN:
1. If the conversation has 0 user messages, set status='ASKING_FOLLOWUP' and ask 2 concise, specific questions based on the telemetry (e.g., if it was a loss or had no SL, ask what invalidated the setup or why SL wasn't set; if a win, ask what confirmation triggered entry and whether exit followed plan).
2. If the trader has answered clearly OR must_complete={must_complete}, set status='COMPLETED', populate 'final_entry' completely, and write a crisp 2-sentence coaching takeaway in 'coach_message'.
3. Only set status='ASKING_FOLLOWUP' after 1 user turn if the trader gave a one-word/vague answer (e.g., "good setup" or "got stopped").
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

        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=DebriefTurnResponse,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                temperature=0.2,
            ),
        )

        result = DebriefTurnResponse.model_validate_json(response.text)

        # Compile full transcript safely and persist to SQLite once completed
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
        """Compute deterministic win rates, expectancy, R-multiples, and mistake frequencies."""
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

    def chat_with_journal(self, db: Session, messages: list[ChatMessage], lookback_days: int = 60) -> str:
        """
        Natural-language Q&A over the trader's SQLite journal using automatic Python tool calling.
        gemini-4-hydrogen queries exact trade rows, metrics, and drawdown snapshots before answering.
        """

        def get_performance_summary(days: int = lookback_days) -> dict:
            """Return win rates, net PnL, average R, and breakdowns by symbol, session, setup, emotion, and plan adherence."""
            return self.compute_performance_metrics(db, lookback_days=days)

        def search_trades(
            symbol: Optional[str] = None,
            session: Optional[str] = None,
            outcome: Optional[str] = None,
            followed_plan: Optional[bool] = None,
            limit: int = 15,
        ) -> list[dict]:
            """
            Query individual closed trades from the database with optional filters.
            Args:
                symbol: Filter by instrument such as 'XAUUSD' or 'EURUSD'.
                session: Filter by session ('ASIA', 'LONDON', 'NEW_YORK', 'LONDON_NY_OVERLAP').
                outcome: Filter by 'WIN' (net_profit > 0) or 'LOSS' (net_profit < 0).
                followed_plan: Filter by True or False.
                limit: Max number of recent trades to return (default 15).
            """
            q = db.query(Trade).options(joinedload(Trade.journal_entry)).filter(Trade.status == "CLOSED")
            if symbol:
                q = q.filter(Trade.symbol == symbol.upper())
            if outcome and outcome.upper() == "WIN":
                q = q.filter(Trade.net_profit > 0)
            elif outcome and outcome.upper() == "LOSS":
                q = q.filter(Trade.net_profit < 0)

            rows = q.order_by(Trade.close_time.desc()).limit(100).all()
            filtered = []
            for t in rows:
                je = t.journal_entry
                if session and (not je or (je.session or "").upper() != session.upper()):
                    continue
                if followed_plan is not None and (not je or je.followed_plan != followed_plan):
                    continue
                filtered.append(
                    {
                        "mt5_id": t.mt5_position_id,
                        "symbol": t.symbol,
                        "direction": t.direction,
                        "volume": t.volume,
                        "net_profit": t.net_profit,
                        "r_multiple": t.r_multiple,
                        "close_time": t.close_time.isoformat() if t.close_time else None,
                        "session": je.session if je else "UNJOURNALED",
                        "setup_type": je.setup_type if je else None,
                        "emotional_state": je.emotional_state if je else None,
                        "followed_plan": je.followed_plan if je else None,
                        "mistake_tags": je.mistake_tags if je else [],
                        "ai_feedback": je.ai_feedback if je else None,
                    }
                )
                if len(filtered) >= limit:
                    break
            return filtered

        def get_the5ers_risk_status() -> dict:
            """Return current The5ers High Stakes account rules and latest daily/static drawdown buffer snapshot."""
            snap = (
                db.query(DailyAccountSnapshot)
                .order_by(DailyAccountSnapshot.trading_date.desc())
                .first()
            )
            return {
                "firm_rules": self.rules,
                "latest_snapshot": {
                    "trading_date": snap.trading_date,
                    "current_equity": snap.current_equity,
                    "daily_drawdown_used_pct": snap.daily_drawdown_used_pct,
                    "static_drawdown_used_pct": snap.static_drawdown_used_pct,
                    "is_locked_out": snap.is_locked_out,
                }
                if snap
                else "No MT5 snapshot synced yet.",
            }

        system_instruction = (
            "You are the trader's quantitative performance analyst and discipline coach for a The5ers High Stakes account. "
            "Always call the provided database tools to retrieve exact trade numbers before answering questions about performance, "
            "win rates, sessions, symbols, or drawdown buffers. Cite specific MT5 position IDs, dollar amounts, and R-multiples. "
            "Be direct, candid, and actionable."
        )

        contents = [
            types.Content(
                role=m.role,
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in messages
        ]

        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[get_performance_summary, search_trades, get_the5ers_risk_status],
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                temperature=0.2,
            ),
        )
        return response.text or "No response generated."

    def run_leak_audit(self, db: Session, lookback_days: int = 30) -> str:
        """Deep diagnostic report across all recent trades."""
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