import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sqlalchemy.orm import Session

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import MacroEvent, MacroRegimeSnapshot
from core.schemas import EconomicCatalyst, MacroRegimeReport, NewsBlackoutStatus

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("macro_agent")

PRIMARY_MODEL_ID = "gemini-4-hydrogen"
SECONDARY_MODEL_ID = "gemini-3.8-flash"
FALLBACK_MODEL_ID = "gemini-3.5-flash-lite"

FF_WEEKLY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
UAE_TZ = timezone(timedelta(hours=4))


def format_utc_and_uae(dt: datetime) -> str:
    """Format a datetime as 'YYYY-MM-DD HH:MM UTC (HH:MM UAE)'."""
    utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    uae_dt = utc_dt.astimezone(UAE_TZ)
    return f"{utc_dt.strftime('%Y-%m-%d %H:%M')} UTC ({uae_dt.strftime('%H:%M')} UAE)"


class MacroCatalystAgent:
    """
    Phase 2 Agent:
    - Fetches high-impact USD/EUR/GBP/JPY calendar events directly from ForexFactory's free JSON feed.
    - Uses a resilient Gemini call with automatic 503/429 retry & multi-model fallback
      (gemini-4-hydrogen -> gemini-3.8-flash -> gemini-3.5-flash-lite) to classify macro regime
      and directional bias for XAU/USD & EUR/USD in plain, easy-to-understand English.
    - Enforces The5ers 30-minute high-impact news blackout buffer.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables (.env).")
        self.client = genai.Client(api_key=api_key)
        self.rules = load_risk_rules()

    @staticmethod
    def _parse_iso_utc(ts_str: str) -> Optional[datetime]:
        """Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
        try:
            cleaned = ts_str.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _fetch_forexfactory_calendar(self) -> list[dict]:
        """Fetch this week's economic calendar directly without consuming Gemini Search quota."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TradingCoPilot/2.0"}
        resp = requests.get(FF_WEEKLY_CALENDAR_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        raw_events = resp.json()

        target_currencies = {"USD", "EUR", "GBP", "JPY"}
        filtered = []
        for item in raw_events:
            ccy = (item.get("country") or "").upper()
            impact = (item.get("impact") or "").upper()
            if ccy not in target_currencies or impact not in {"HIGH", "MEDIUM"}:
                continue

            dt = self._parse_iso_utc(item.get("date", ""))
            if not dt:
                continue

            affected = ["XAUUSD", "EURUSD"] if ccy == "USD" else [f"{ccy}USD" if ccy != "JPY" else "USDJPY"]
            filtered.append(
                {
                    "event_name": item.get("title", "Economic Release"),
                    "currency": ccy,
                    "impact": impact,
                    "scheduled_utc": dt.isoformat(),
                    "scheduled_display": format_utc_and_uae(dt),
                    "forecast": item.get("forecast") or None,
                    "previous": item.get("previous") or None,
                    "actual": item.get("actual") or None,
                    "affected_symbols": affected,
                }
            )
        return filtered

    def _generate_with_fallback(self, prompt: str) -> MacroRegimeReport:
        """
        Resilient generation chain:
        1. Try gemini-4-hydrogen (with 1 quick retry on 503 temporary demand spikes)
        2. Fallback to gemini-3.8-flash on 503/429
        3. Fallback to gemini-3.5-flash-lite on 503/429
        """
        attempts = [
            (PRIMARY_MODEL_ID, 1.5),
            (PRIMARY_MODEL_ID, 2.0),
            (SECONDARY_MODEL_ID, 1.5),
            (FALLBACK_MODEL_ID, 0.0),
        ]
        last_err = None

        for idx, (model_name, backoff_sec) in enumerate(attempts):
            try:
                config_kwargs = {
                    "response_mime_type": "application/json",
                    "response_schema": MacroRegimeReport,
                    "temperature": 0.1,
                }
                if model_name == PRIMARY_MODEL_ID:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                logger.info(f"Macro report generated successfully using {model_name}.")
                return MacroRegimeReport.model_validate_json(response.text)
            except APIError as exc:
                last_err = exc
                if exc.code in (429, 500, 503) and idx < len(attempts) - 1:
                    logger.warning(
                        f"HTTP {exc.code} on {model_name} (attempt {idx + 1}/{len(attempts)}); "
                        f"retrying in {backoff_sec}s..."
                    )
                    time.sleep(backoff_sec)
                    continue
                raise
        raise last_err

    def refresh_macro_brief(self, db: Session) -> dict:
        """
        1. Fetch verified weekly economic calendar directly from ForexFactory JSON feed.
        2. Run a resilient structured Gemini call to classify macro regime & XAU/USD + EUR/USD bias in plain English.
        3. Persist MacroRegimeSnapshot and upsert MacroEvent rows into SQLite.
        """
        now_utc = datetime.now(timezone.utc)
        today_str = format_utc_and_uae(now_utc)

        calendar_events = self._fetch_forexfactory_calendar()
        high_impact_events = [e for e in calendar_events if e["impact"] == "HIGH"]

        prompt = f"""
You are a helpful trading coach advising a prop trader on XAU/USD (Gold) and EUR/USD.
Current Time: {today_str}.

Review the High-Impact Economic Calendar below and explain what is driving Gold and EUR/USD right now in SIMPLE, PLAIN, EVERYDAY ENGLISH (avoid complex Wall Street jargon).
Whenever you mention a time in your explanation, ALWAYS write it as: HH:MM UTC (HH:MM UAE), where UAE is UTC+4.

VERIFIED WEEKLY ECONOMIC CALENDAR:
{json.dumps(high_impact_events[:25], indent=2)}

RULES:
- regime: Choose RISK_ON, RISK_OFF, RATE_DRIVEN, or EVENT_VOLATILITY (use EVENT_VOLATILITY when major USD/EUR news is within 24 hours).
- xauusd_bias & eurusd_bias: Choose BULLISH (buyers in control), BEARISH (sellers in control), NEUTRAL (sideways/range), or VOLATILE_CHOP (messy up-and-down spikes—avoid chasing breakouts).
- dxy_yield_context: 2 short, plain-English sentences explaining whether the US Dollar is strong or weak and what that means for Gold and EUR/USD.
- central_bank_summary: 2 short, plain-English sentences explaining what the US Federal Reserve and European Central Bank are expected to do with interest rates.
- trading_guideline: 1-2 clear, practical sentences telling the trader how to manage risk and entries today (including exact UTC and UAE times for any major news).
- Leave upcoming_catalysts as an empty list [] because the calendar events are already parsed deterministically.
"""

        report = self._generate_with_fallback(prompt)

        sources = [
            {
                "title": "ForexFactory Weekly Economic Calendar Feed",
                "uri": "https://www.forexfactory.com/calendar",
            }
        ]

        # Persist MacroRegimeSnapshot
        snapshot = MacroRegimeSnapshot(
            regime=report.regime,
            xauusd_bias=report.xauusd_bias,
            eurusd_bias=report.eurusd_bias,
            dxy_yield_context=report.dxy_yield_context,
            central_bank_summary=report.central_bank_summary,
            trading_guideline=report.trading_guideline,
            sources=sources,
        )
        db.add(snapshot)

        # Upsert all HIGH-impact calendar events from the feed into SQLite
        upserted_events = 0
        for cat in high_impact_events:
            sched_dt = self._parse_iso_utc(cat["scheduled_utc"])
            if not sched_dt:
                continue

            window_start = sched_dt - timedelta(hours=2)
            window_end = sched_dt + timedelta(hours=2)
            existing = (
                db.query(MacroEvent)
                .filter(
                    MacroEvent.currency == cat["currency"],
                    MacroEvent.event_name == cat["event_name"],
                    MacroEvent.scheduled_utc >= window_start,
                    MacroEvent.scheduled_utc <= window_end,
                )
                .first()
            )
            if existing:
                existing.scheduled_utc = sched_dt
                existing.impact = cat["impact"]
                existing.forecast = cat["forecast"] or existing.forecast
                existing.previous = cat["previous"] or existing.previous
                existing.actual = cat["actual"] or existing.actual
                existing.affected_symbols = cat["affected_symbols"]
            else:
                db.add(
                    MacroEvent(
                        event_name=cat["event_name"],
                        currency=cat["currency"],
                        impact=cat["impact"],
                        scheduled_utc=sched_dt,
                        forecast=cat["forecast"],
                        previous=cat["previous"],
                        actual=cat["actual"],
                        affected_symbols=cat["affected_symbols"],
                    )
                )
            upserted_events += 1

        db.commit()
        logger.info(
            f"Macro refresh complete | Regime: {report.regime} | XAUUSD: {report.xauusd_bias} | "
            f"EURUSD: {report.eurusd_bias} | High-impact events upserted: {upserted_events}"
        )

        return {
            "report": report.model_dump(),
            "sources": sources,
            "events_upserted": upserted_events,
            "updated_at": format_utc_and_uae(snapshot.created_at),
        }

    def add_manual_event(self, db: Session, catalyst: EconomicCatalyst) -> MacroEvent:
        """Allow the trader to manually pin a critical release (e.g., NFP, CPI, FOMC) into SQLite."""
        sched_dt = self._parse_iso_utc(catalyst.scheduled_utc)
        if not sched_dt:
            raise ValueError(f"Invalid ISO-8601 UTC timestamp: '{catalyst.scheduled_utc}'. Use format YYYY-MM-DDTHH:MM:SSZ.")

        ev = MacroEvent(
            event_name=catalyst.event_name,
            currency=catalyst.currency,
            impact=catalyst.impact,
            scheduled_utc=sched_dt,
            forecast=catalyst.forecast,
            previous=catalyst.previous,
            actual=catalyst.actual,
            affected_symbols=catalyst.affected_symbols,
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return ev

    def check_news_blackout(self, db: Session, symbol: Optional[str] = None) -> NewsBlackoutStatus:
        """
        Enforce config/risk_rules.yaml `high_impact_news_buffer_minutes` (default: 30 mins before/after).
        Displays both UTC and UAE (UTC+4) times in the banner message.
        """
        buffer_mins = int(self.rules["instruments"].get("high_impact_news_buffer_minutes", 30))
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=buffer_mins)
        window_end = now_utc + timedelta(minutes=buffer_mins)

        events = (
            db.query(MacroEvent)
            .filter(
                MacroEvent.impact == "HIGH",
                MacroEvent.scheduled_utc >= window_start,
                MacroEvent.scheduled_utc <= window_end,
            )
            .order_by(MacroEvent.scheduled_utc.asc())
            .all()
        )

        relevant_events = []
        for ev in events:
            if not symbol or symbol.upper() == "ALL":
                relevant_events.append(ev)
            else:
                sym = symbol.upper()
                if ev.currency in sym or (sym == "XAUUSD" and ev.currency == "USD"):
                    relevant_events.append(ev)

        if relevant_events:
            target = relevant_events[0]
            ev_dt = target.scheduled_utc if target.scheduled_utc.tzinfo else target.scheduled_utc.replace(tzinfo=timezone.utc)
            delta_mins = int(round((ev_dt - now_utc).total_seconds() / 60.0))
            timing_label = f"in {delta_mins} min" if delta_mins >= 0 else f"{abs(delta_mins)} min ago"
            time_str = format_utc_and_uae(ev_dt)
            return NewsBlackoutStatus(
                is_blackout_active=True,
                buffer_minutes=buffer_mins,
                active_or_imminent_event=f"[{target.currency}] {target.event_name}",
                minutes_to_event=delta_mins,
                message=(
                    f"DO NOT TRADE (News Blackout): [{target.currency}] {target.event_name} at {time_str} ({timing_label}). "
                    f"Stay out within +/-{buffer_mins} mins to avoid spread spikes."
                ),
            )

        next_ev = (
            db.query(MacroEvent)
            .filter(
                MacroEvent.impact == "HIGH",
                MacroEvent.scheduled_utc > window_end,
            )
            .order_by(MacroEvent.scheduled_utc.asc())
            .first()
        )

        if next_ev:
            ev_dt = next_ev.scheduled_utc if next_ev.scheduled_utc.tzinfo else next_ev.scheduled_utc.replace(tzinfo=timezone.utc)
            mins_away = int(round((ev_dt - now_utc).total_seconds() / 60.0))
            time_str = format_utc_and_uae(ev_dt)
            return NewsBlackoutStatus(
                is_blackout_active=False,
                buffer_minutes=buffer_mins,
                active_or_imminent_event=f"[{next_ev.currency}] {next_ev.event_name}",
                minutes_to_event=mins_away,
                message=f"SAFE TO TRADE. Next major news: [{next_ev.currency}] {next_ev.event_name} at {time_str} (in {mins_away} min).",
            )

        return NewsBlackoutStatus(
            is_blackout_active=False,
            buffer_minutes=buffer_mins,
            message=f"SAFE TO TRADE. No major news events within +/-{buffer_mins} min buffer.",
        )


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        agent = MacroCatalystAgent()
        result = agent.refresh_macro_brief(db)
        print("\n=== Phase 2 Macro Regime Report ===")
        print(f"Regime       : {result['report']['regime']}")
        print(f"XAUUSD Bias  : {result['report']['xauusd_bias']}")
        print(f"EURUSD Bias  : {result['report']['eurusd_bias']}")
        print(f"USD & Yields : {result['report']['dxy_yield_context']}")
        print(f"Central Banks: {result['report']['central_bank_summary']}")
        print(f"Action Plan  : {result['report']['trading_guideline']}")
        print(f"Events Saved : {result['events_upserted']}")
        blackout = agent.check_news_blackout(db)
        print(f"News Status  : {blackout.message}")
    finally:
        db.close()