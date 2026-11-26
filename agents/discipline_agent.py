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

from core.database import SessionLocal, init_db, load_risk_rules, load_strategy_rules
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
    Chart & Setup Advisor + Automatic Process Trade Grader:
    1. Stage 1 In-Flight Capture: Records entry thesis, invalidation SL, emotion & chart screenshot while OPEN.
    2. Chart & Setup Advisor (6-Strategy Rule Contracts):
       - Evaluates Elliott Wave cycle position, Fibonacci Golden Zone (0.50-0.786) & -0.272/-0.618 targets,
         ICT Liquidity/FVG/OB, Trendline geometry, and Asian Killzone rules from config/strategy_rules.yaml.
       - Applies deterministic Python post-checks so any [MISSING] checklist rule forces CAUTION
         and any [FAIL] or hard risk/news/ADR violation forces BLOCK.
    3. Stage 2 Close Verdict: Automatically grades closed trades as GOOD_TRADE or BAD_TRADE.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=25_000),
        )
        self.rules = load_risk_rules()
        self.strategy_rules = load_strategy_rules()

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
        return {
            "mt5_position_id": mt5_position_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "stage1_completed_at": format_utc_and_uae(now_utc),
            "message": "Entry plan saved. Process grade will generate automatically when MT5 closes this position.",
        }

    def evaluate_closed_trade_process(self, db: Session, mt5_position_id: int) -> ProcessOutcomeVerdict:
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
        return verdict

    def _format_strategy_playbook_for_prompt(self) -> str:
        strategies = self.strategy_rules.get("strategies", {})
        lines = []
        for key, cfg in strategies.items():
            lines.append(f"### {cfg.get('display_name', key)} (`{key}`)")
            lines.append(f"- Ideal Condition: {cfg.get('ideal_condition', '')}")
            lines.append("- Mandatory Checklist (ALL must pass for green PASS):")
            for item in cfg.get("mandatory_checklist", []):
                lines.append(f"  * {item}")
            if cfg.get("disqualifiers"):
                lines.append("- Instant Disqualifiers (Force BLOCK if triggered):")
                for dq in cfg.get("disqualifiers", []):
                    lines.append(f"  * {dq}")
            lines.append(
                f"- Levels Formula -> SL: {cfg.get('sl_formula')} | TP1: {cfg.get('tp1_formula')} | TP2: {cfg.get('tp2_formula')}"
            )
        return "\n".join(lines)

    def _enforce_deterministic_verdict(
        self,
        audit: PreTradeAuditReport,
        sym: str,
        allowed_symbols: list[str],
        proposed_risk_pct: float,
        max_risk_pct: float,
        dd_snap: Optional[DailyAccountSnapshot],
        blackout_active: bool,
        blackout_msg: str,
        adr_used_pct: Optional[float],
    ) -> PreTradeAuditReport:
        """
        Deterministic Python Rule Gate:
        Guarantees that no optimistic AI verdict can bypass hard risk rules, strategy disqualifiers,
        or incomplete checklist items.
        """
        min_confluences = int(self.strategy_rules.get("global_gates", {}).get("min_confluences_for_pass", 2))
        max_adr_breakout = float(self.strategy_rules.get("global_gates", {}).get("max_adr_used_pct_for_breakouts", 80.0))

        hard_blocks: list[str] = list(audit.rule_violations)

        if sym not in allowed_symbols:
            hard_blocks.append(f"Instrument {sym} is outside The5ers allowed list ({allowed_symbols}).")
        if proposed_risk_pct > max_risk_pct:
            hard_blocks.append(f"Planned risk ({proposed_risk_pct}%) exceeds the {max_risk_pct}% max cap.")
        if dd_snap and dd_snap.is_locked_out:
            hard_blocks.append("Account drawdown soft-lock is active — no new trades allowed today.")
        if blackout_active:
            hard_blocks.append(f"High-impact news blackout active: {blackout_msg}")

        failed_items = [c for c in audit.strategy_checklist if c.status == "FAIL"]
        missing_items = [c for c in audit.strategy_checklist if c.status == "MISSING"]

        for f_item in failed_items:
            msg = f"Disqualifier triggered ({f_item.rule_name}): {f_item.detail}"
            if msg not in hard_blocks:
                hard_blocks.append(msg)

        if hard_blocks:
            audit.verdict = "BLOCK"
            audit.rule_violations = hard_blocks
            return audit

        caution_reasons: list[str] = []
        if missing_items:
            for m_item in missing_items:
                caution_reasons.append(f"Waiting for ({m_item.rule_name}): {m_item.detail}")
        if audit.confluences_count < min_confluences:
            caution_reasons.append(
                f"Only {audit.confluences_count} confluence(s) aligned (minimum {min_confluences} required for PASS)."
            )
        if audit.recommended_action == "WAIT":
            caution_reasons.append("No high-probability entry trigger active right now — wait for confirmation.")
        if adr_used_pct is not None and adr_used_pct > max_adr_breakout:
            caution_reasons.append(
                f"{adr_used_pct}% of today's typical daily move is already used — avoid chasing late breakouts."
            )

        if caution_reasons:
            audit.verdict = "CAUTION"
            for r in caution_reasons:
                if r not in audit.rule_violations:
                    audit.rule_violations.append(r)
        else:
            audit.verdict = "PASS"

        return audit

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
        adr_used_pct: Optional[float] = None
        if weekend_closed:
            vol_context = "Weekend Market Closed — focus on HTF/LTF structure and next week's key levels."
        else:
            vol_context = "Live MT5 data unavailable (terminal offline)."
            if mt5_conn.connect():
                try:
                    sym_vol = mt5_conn.get_symbol_volatility(sym)
                    if sym_vol:
                        adr_used_pct = sym_vol.adr_used_pct
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
            "and match the single best strategy from the strict playbook below."
            if mode == "RECOMMEND"
            else f"MODE = 'AUDIT_{mode}' (The trader wants to {mode}). Stress-test whether {mode} satisfies every mandatory rule or fails."
        )

        playbook_rules_text = self._format_strategy_playbook_for_prompt()

        prompt = f"""
You are the Senior Technical, Elliott Wave, Fibonacci, ICT & Trendline Chart Strategist for a The5ers High Stakes prop trader.
Speak in clear, plain, everyday English. Never mention AI model names or technical phase numbers.
Always format any mentioned times as UTC (UAE UTC+4).

{intent_instruction}

TRADER REQUEST:
- Instrument: {sym} (Allowed List: {allowed_symbols})
- Intent Mode: {mode} | Proposed Direction: {direction}
- Execution Timeframe: {timeframe} | Higher Timeframe Provided: {htf_timeframe if htf_part else 'None'}
- Proposed Risk: {proposed_risk_pct}% (Hard Max Cap: {max_risk_pct}%)
- Market Open Status: {'WEEKEND CLOSED (Chart Study Mode)' if weekend_closed else 'LIVE MARKET SESSION'}
- Trader's Notes / Question: \"\"\"{setup_description or 'Analyze my chart, map Elliott Wave & Fibonacci levels, and check all strategy rules.'}\"\"\"

LIVE SESSION, VOLATILITY & GUARDRAIL TELEMETRY:
1. Session Radar: {session_status.banner_message}
2. Live Market Activity ({sym}): {vol_context}
3. The5ers Drawdown Buffer: {json.dumps(dd_context)}
4. News Blackout Status (+/-30m rule): {blackout.model_dump_json()}
5. Market Guide Bias: {json.dumps(macro_context)}
6. Trader's 60-Day Edge on {sym}: {json.dumps(symbol_edge)}
7. Trader's 60-Day Edge in {current_session}: {json.dumps(session_edge)}

STRICT 6-STRATEGY RULE CONTRACTS (Select the best fit and audit EVERY mandatory rule):
{playbook_rules_text}

MANDATORY OUTPUT REQUIREMENTS:
1. **Top-Down HTF Check (`htf_bias_read`):** Identify Premium vs. Discount dealing range, 20-EMA trend, and HTF draw on liquidity.
2. **Elliott Wave Cycle Read (`wave_cycle_read`):** Identify the current wave count (e.g., Wave 1-2-3-4-5 impulse or A-B-C correction), whether price is in a Wave 2 (`0.50-0.786`) pullback, Wave 4 (`0.236-0.382`) consolidation, or Wave 5 exhaustion with 14-RSI divergence, and state the exact price level that invalidates the wave count.
3. **Fibonacci Retracement & Extension Levels (`fib_levels_summary`):** Anchor on the dominant swing High-to-Low (or Low-to-High) — or critique the user's drawn Fib if already plotted on the chart — and list exact price levels for the **0.50–0.618–0.786 Golden Zone**, **-0.272**, and **-0.618** extensions.
4. **Drawn-Chart Critique vs. Blank-Chart Mapping (`analysis_mode` & `trendline_ict_critique`):**
   - If user drew trendlines/Fibs/boxes: `analysis_mode='DRAWN_CHART_CRITIQUE'`. Judge VALID vs. FORCED anchor and warn if an unbroken multi-touch trendline is a retail stop-hunt trap.
   - If blank chart: `analysis_mode='BLANK_CHART_MAPPING'` and map the key highs/lows, FVGs, Order Blocks, and trendlines.
   - If no image: `analysis_mode='TEXT_ONLY'`.
5. **Rule-by-Rule Strategy Checklist (`strategy_checklist`):**
   - For the selected `recommended_strategy`, output one `StrategyChecklistItem` per mandatory rule (plus any triggered disqualifier):
     * `status='PASS'` if clearly confirmed on the chart right now.
     * `status='MISSING'` if the setup is forming but waiting for that trigger (e.g., price has not yet reached the `0.618` Golden Zone or `15m` MSS has not printed yet).
     * `status='FAIL'` if a hard disqualifier is broken (e.g., Wave 4 overlaps Wave 1, Forced Trendline, or News Blackout).
6. **Exact Levels (`invalidation_sl_zone`, `tp1_zone`, `tp2_zone`):** Provide exact price zones for Stop Loss, Take-Profit 1 (`TP1`), and Take-Profit 2 (`TP2`) following the matched strategy's formula.
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

        audit = self._enforce_deterministic_verdict(
            audit=audit,
            sym=sym,
            allowed_symbols=allowed_symbols,
            proposed_risk_pct=proposed_risk_pct,
            max_risk_pct=max_risk_pct,
            dd_snap=dd_snap,
            blackout_active=blackout.is_blackout_active,
            blackout_msg=blackout.message,
            adr_used_pct=adr_used_pct,
        )

        log_row = DisciplineAuditLog(
            symbol=sym,
            intent_mode=mode,
            proposed_direction=direction if mode != "RECOMMEND" else audit.recommended_action,
            timeframe=timeframe,
            htf_timeframe=htf_timeframe if htf_part else None,
            proposed_risk_pct=proposed_risk_pct,
            setup_description=setup_description or "Chart & Setup Advisor Request",
            chart_image_path=chart_image_path,
            htf_chart_image_path=htf_chart_image_path,
            verdict=audit.verdict,
            recommended_action=audit.recommended_action,
            recommended_strategy=audit.recommended_strategy,
            confluences_count=audit.confluences_count,
            confluences_found=audit.confluences_found,
            strategy_checklist=[c.model_dump() for c in audit.strategy_checklist],
            discipline_score=audit.discipline_score,
            htf_bias_read=audit.htf_bias_read,
            wave_cycle_read=audit.wave_cycle_read,
            fib_levels_summary=audit.fib_levels_summary,
            chart_structure_read=audit.chart_structure_read,
            trendline_ict_critique=audit.trendline_ict_critique,
            trigger_to_wait_for=audit.trigger_to_wait_for,
            invalidation_sl_zone=audit.invalidation_sl_zone,
            tp1_zone=audit.tp1_zone,
            tp2_zone=audit.tp2_zone,
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