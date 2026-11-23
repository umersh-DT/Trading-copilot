import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sqlalchemy.orm import Session

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import MacroEvent, MacroRegimeSnapshot
from core.schemas import (
    EconomicCatalyst,
    MacroRegimeReport,
    NewsBlackoutStatus,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("macro_agent")

PRIMARY_MODEL = "gemini-4-hydrogen"
SECONDARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

UAE_TZ = timezone(timedelta(hours=4))


def format_utc_and_uae(dt: datetime) -> str:
    """Format a datetime cleanly in both UTC and UAE (UTC+4)."""
    utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    uae_dt = utc_dt.astimezone(UAE_TZ)
    return f"{utc_dt.strftime('%Y-%m-%d %H:%M')} UTC ({uae_dt.strftime('%H:%M')} UAE)"


def format_duration_readable(total_minutes: int) -> str:
    """Convert raw minutes into 'Xd Xh Ym', 'Xh Ym', or 'Ym' (never raw thousands of minutes)."""
    mins = max(1, int(total_minutes))
    days, rem = divmod(mins, 1440)
    hours, rem_mins = divmod(rem, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if rem_mins > 0 and days == 0:
        parts.append(f"{rem_mins}m")
    return " ".join(parts) if parts else "1m"


class MacroCatalystAgent:
    """
    Phase 2 Macro & Economic Calendar Agent:
    - Fetches upcoming high-impact USD/EUR/GBP/JPY catalysts using Google Search grounding.
    - Classifies plain-English market regime and directional biases for XAUUSD and EURUSD.
    - Enforces +/-30 minute news blackout windows with human-readable (Xd Xh Ym) countdowns.
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
        self.buffer_minutes = int(self.rules.get("news_blackout", {}).get("buffer_minutes", 30))

    def _call_with_fallback(self, contents: str, use_search: bool = True, response_schema=None):
        attempts = [
            (PRIMARY_MODEL, 0.8),
            (SECONDARY_MODEL, 0.6),
            (FALLBACK_MODEL, 0.0),
        ]
        last_err = None
        for idx, (model_name, backoff) in enumerate(attempts):
            try:
                config_kwargs = {"temperature": 0.15}
                if use_search:
                    config_kwargs["tools"] = [{"google_search": {}}]
                if response_schema is not None:
                    config_kwargs["response_mime_type"] = "application/json"
                    config_kwargs["response_schema"] = response_schema

                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return resp
            except APIError as exc:
                last_err = exc
                if idx < len(attempts) - 1:
                    logger.warning(f"APIError ({exc.code}) on {model_name}; switching to {attempts[idx + 1][0]}...")
                    if backoff > 0:
                        time.sleep(backoff)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if idx < len(attempts) - 1:
                    logger.warning(f"Network/timeout error on {model_name}: {exc}; switching to {attempts[idx + 1][0]}...")
                    if backoff > 0:
                        time.sleep(backoff)
                    continue
                raise
        raise last_err

    def refresh_macro_brief(self, db: Session) -> dict:
        """Fetch current week's high-impact USD/EUR/GBP/JPY catalysts and save regime classification."""
        now_utc = datetime.now(timezone.utc)
        week_end = now_utc + timedelta(days=7)

        search_prompt = f"""
Current UTC Date/Time: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}.
Search for the official Forex economic calendar high-impact (red-folder) events between {now_utc.strftime('%Y-%m-%d')} and {week_end.strftime('%Y-%m-%d')} for USD, EUR, GBP, and JPY (such as CPI, Core PCE, NFP, FOMC, ECB, BOE, BOJ, GDP, PMI, Unemployment Claims).
Also summarize the current US Dollar (DXY), US 10-Year Treasury yield trend, Federal Reserve / ECB rate outlook, and plain-English directional bias for XAU/USD (Gold) and EUR/USD.
Provide exact ISO-8601 UTC timestamps for every scheduled news event.
"""
        grounded_resp = self._call_with_fallback(contents=search_prompt, use_search=True, response_schema=None)
        raw_research = grounded_resp.text or ""

        sources = []
        try:
            cand = grounded_resp.candidates[0]
            if cand.grounding_metadata and cand.grounding_metadata.grounding_chunks:
                for chunk in cand.grounding_metadata.grounding_chunks:
                    if chunk.web and chunk.web.uri:
                        sources.append(chunk.web.uri)
        except Exception:
            pass

        struct_prompt = f"""
Convert the macro research below into a clean, plain-English trading guide for a prop trader trading XAUUSD and EURUSD.
Avoid technical jargon. Always format any mentioned times in both UTC and UAE (UTC+4).

RESEARCH SUMMARY:
{raw_research}
"""
        struct_resp = self._call_with_fallback(
            contents=struct_prompt,
            use_search=False,
            response_schema=MacroRegimeReport,
        )
        report = MacroRegimeReport.model_validate_json(struct_resp.text)

        snap = MacroRegimeSnapshot(
            regime=report.regime,
            xauusd_bias=report.xauusd_bias,
            eurusd_bias=report.eurusd_bias,
            dxy_yield_context=report.dxy_yield_context,
            central_bank_summary=report.central_bank_summary,
            trading_guideline=report.trading_guideline,
            sources=list(dict.fromkeys(sources))[:8],
        )
        db.add(snap)

        upserted = 0
        for cat in report.upcoming_catalysts:
            try:
                ev_dt = datetime.fromisoformat(cat.scheduled_utc.replace("Z", "+00:00"))
                if ev_dt.tzinfo is None:
                    ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                else:
                    ev_dt = ev_dt.astimezone(timezone.utc)
            except Exception:
                continue

            window_start = ev_dt - timedelta(hours=2)
            window_end = ev_dt + timedelta(hours=2)
            existing = (
                db.query(MacroEvent)
                .filter(
                    MacroEvent.event_name == cat.event_name,
                    MacroEvent.currency == cat.currency,
                    MacroEvent.scheduled_utc >= window_start,
                    MacroEvent.scheduled_utc <= window_end,
                )
                .first()
            )
            if existing:
                existing.scheduled_utc = ev_dt
                existing.impact = cat.impact
                existing.forecast = cat.forecast
                existing.previous = cat.previous
                existing.actual = cat.actual
                existing.affected_symbols = cat.affected_symbols
            else:
                db.add(
                    MacroEvent(
                        event_name=cat.event_name,
                        currency=cat.currency,
                        impact=cat.impact,
                        scheduled_utc=ev_dt,
                        forecast=cat.forecast,
                        previous=cat.previous,
                        actual=cat.actual,
                        affected_symbols=cat.affected_symbols,
                    )
                )
            upserted += 1

        db.commit()
        return {
            "regime": report.regime,
            "xauusd_bias": report.xauusd_bias,
            "eurusd_bias": report.eurusd_bias,
            "events_upserted": upserted,
        }

    def add_manual_event(self, db: Session, catalyst: EconomicCatalyst) -> MacroEvent:
        ev_dt = datetime.fromisoformat(catalyst.scheduled_utc.replace("Z", "+00:00"))
        if ev_dt.tzinfo is None:
            ev_dt = ev_dt.replace(tzinfo=timezone.utc)
        else:
            ev_dt = ev_dt.astimezone(timezone.utc)

        ev = MacroEvent(
            event_name=catalyst.event_name,
            currency=catalyst.currency,
            impact=catalyst.impact,
            scheduled_utc=ev_dt,
            forecast=catalyst.forecast,
            previous=catalyst.previous,
            actual=catalyst.actual,
            affected_symbols=catalyst.affected_symbols,
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return ev

    def check_news_blackout(self, db: Session, symbol: Optional[str] = "ALL") -> NewsBlackoutStatus:
        """Check +/-30m red-folder news blackout window and format countdown in readable Xd Xh Ym."""
        now_utc = datetime.now(timezone.utc)
        lookahead = now_utc + timedelta(days=7)
        lookbehind = now_utc - timedelta(minutes=self.buffer_minutes)

        events = (
            db.query(MacroEvent)
            .filter(
                MacroEvent.impact == "HIGH",
                MacroEvent.scheduled_utc >= lookbehind,
                MacroEvent.scheduled_utc <= lookahead,
            )
            .order_by(MacroEvent.scheduled_utc.asc())
            .all()
        )

        sym_upper = (symbol or "ALL").upper().strip()
        relevant_events = []
        for ev in events:
            if sym_upper == "ALL":
                relevant_events.append(ev)
            else:
                aff = [s.upper() for s in (ev.affected_symbols or [])]
                if sym_upper in aff or (ev.currency == "USD" and sym_upper in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY")):
                    relevant_events.append(ev)

        if not relevant_events:
            return NewsBlackoutStatus(
                is_blackout_active=False,
                buffer_minutes=self.buffer_minutes,
                message="SAFE TO TRADE — No major high-impact news scheduled nearby.",
            )

        # Check if any event is currently inside the +/- buffer_minutes blackout window
        for ev in relevant_events:
            ev_utc = ev.scheduled_utc.replace(tzinfo=timezone.utc) if ev.scheduled_utc.tzinfo is None else ev.scheduled_utc
            diff_mins = int(round((ev_utc - now_utc).total_seconds() / 60.0))
            if abs(diff_mins) <= self.buffer_minutes:
                timing_txt = f"in {format_duration_readable(diff_mins)}" if diff_mins > 0 else f"{format_duration_readable(abs(diff_mins))} ago"
                return NewsBlackoutStatus(
                    is_blackout_active=True,
                    buffer_minutes=self.buffer_minutes,
                    active_or_imminent_event=f"[{ev.currency}] {ev.event_name}",
                    minutes_to_event=diff_mins,
                    message=(
                        f"NEWS BLACKOUT ACTIVE (+/-{self.buffer_minutes}m): [{ev.currency}] {ev.event_name} "
                        f"({timing_txt} at {format_utc_and_uae(ev_utc)}) — Hold off on new entries."
                    ),
                )

        # Otherwise report the next upcoming high-impact event in clean Xd Xh Ym format
        upcoming = [
            ev for ev in relevant_events
            if (ev.scheduled_utc.replace(tzinfo=timezone.utc) if ev.scheduled_utc.tzinfo is None else ev.scheduled_utc) > now_utc
        ]
        if not upcoming:
            return NewsBlackoutStatus(
                is_blackout_active=False,
                buffer_minutes=self.buffer_minutes,
                message="SAFE TO TRADE — No upcoming high-impact news in the next 7 days.",
            )

        next_ev = upcoming[0]
        next_utc = next_ev.scheduled_utc.replace(tzinfo=timezone.utc) if next_ev.scheduled_utc.tzinfo is None else next_ev.scheduled_utc
        mins_away = max(1, int(round((next_utc - now_utc).total_seconds() / 60.0)))
        readable_wait = format_duration_readable(mins_away)

        return NewsBlackoutStatus(
            is_blackout_active=False,
            buffer_minutes=self.buffer_minutes,
            active_or_imminent_event=f"[{next_ev.currency}] {next_ev.event_name}",
            minutes_to_event=mins_away,
            message=(
                f"SAFE TO TRADE. Next major news: [{next_ev.currency}] {next_ev.event_name} "
                f"at {format_utc_and_uae(next_utc)} (in {readable_wait})."
            ),
        )


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        agent = MacroCatalystAgent()
        print(agent.check_news_blackout(db).message)
    finally:
        db.close()