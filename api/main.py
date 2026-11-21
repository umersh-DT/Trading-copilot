import os
import shutil
import threading
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from core.database import SessionLocal, init_db, get_db, load_risk_rules
from core.models import (
    Trade,
    DailyAccountSnapshot,
    MacroEvent,
    MacroRegimeSnapshot,
    DisciplineAuditLog,
)
from core.schemas import (
    DrawdownStatus,
    SessionCountdownStatus,
    SymbolVolatilityMetrics,
    ParsedJournalNote,
    TradeJournalSummary,
    OpenPositionSummary,
    Stage1EntryRequest,
    ProcessOutcomeVerdict,
    ChatMessage,
    DebriefTurnResponse,
    JournalChatRequest,
    EconomicCatalyst,
    NewsBlackoutStatus,
)
from connectors.mt5_sync import MT5SyncConnector
from agents.journal_agent import TradingJournalAgent
from agents.macro_agent import MacroCatalystAgent, format_utc_and_uae
from agents.discipline_agent import DisciplineGatekeeperAgent, is_fx_weekend_closed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MT5_LOCK = threading.Lock()
AUTO_GRADE_LOCK = threading.Lock()


def _grade_position_ids_background(position_ids: list[int]) -> int:
    """Safely grade a list of closed MT5 position IDs in its own DB session."""
    if not position_ids:
        return 0
    if not AUTO_GRADE_LOCK.acquire(blocking=False):
        return 0
    graded = 0
    db = SessionLocal()
    try:
        gatekeeper = DisciplineGatekeeperAgent()
        for pos_id in position_ids:
            try:
                trade = db.query(Trade).filter(Trade.mt5_position_id == pos_id).first()
                if trade and trade.status == "CLOSED" and (
                    not trade.journal_entry or not trade.journal_entry.process_verdict
                ):
                    gatekeeper.evaluate_closed_trade_process(db, mt5_position_id=pos_id)
                    graded += 1
            except Exception as exc:
                logger.warning(f"Auto-grade skipped for MT5 #{pos_id}: {exc}")
    finally:
        db.close()
        AUTO_GRADE_LOCK.release()
    return graded


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized (Phase 3.5 Auto-Grading & Chart Strategist ready).")
    yield


app = FastAPI(
    title="The5ers Multi-Agent Trading Co-Pilot API",
    version="3.5.3",
    lifespan=lifespan,
)


class JournalNoteRequest(BaseModel):
    raw_notes: str = Field(..., min_length=3)


class DebriefTurnRequest(BaseModel):
    conversation: list[ChatMessage] = Field(default_factory=list)
    force_finalize: bool = False


class JournalChatResponse(BaseModel):
    reply: str


class SyncResponse(BaseModel):
    open_positions_count: int
    trades_synced: int
    auto_verdicts_generated: int
    drawdown_status: DrawdownStatus


class AuditResponse(BaseModel):
    lookback_days: int
    audit_report: str


class PreTradeAuditRequest(BaseModel):
    symbol: str = "XAUUSD"
    intent_mode: str = "RECOMMEND"
    proposed_direction: str = "UNSPECIFIED"
    timeframe: str = "M15"
    htf_timeframe: Optional[str] = "H1"
    proposed_risk_pct: float = Field(default=0.5, ge=0.05, le=5.0)
    setup_description: str = ""
    chart_image_path: Optional[str] = None
    htf_chart_image_path: Optional[str] = None


class MarketPulseResponse(BaseModel):
    session: SessionCountdownStatus
    volatility: list[SymbolVolatilityMetrics] = Field(default_factory=list)
    mt5_connected: bool = False


@app.get("/health")
def health_check():
    return {"status": "ok", "phase": "Phase 3.5 - Auto-Graded Process Verdicts & Chart Strategist"}


@app.get("/market/pulse", response_model=MarketPulseResponse)
def get_market_pulse(symbols: str = Query(default="XAUUSD,EURUSD")):
    connector = MT5SyncConnector()
    session_info = connector.get_session_countdown_status()
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    vol_metrics = []
    connected = False
    if not is_fx_weekend_closed() and MT5_LOCK.acquire(timeout=1.5):
        try:
            connected = connector.connect()
            if connected:
                try:
                    vol_metrics = connector.get_market_volatility(sym_list)
                finally:
                    connector.disconnect()
        finally:
            MT5_LOCK.release()

    return MarketPulseResponse(
        session=session_info,
        volatility=vol_metrics,
        mt5_connected=connected,
    )


@app.post("/sync/mt5", response_model=SyncResponse)
def trigger_mt5_sync(lookback_days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    with MT5_LOCK:
        connector = MT5SyncConnector()
        if not connector.connect():
            raise HTTPException(
                status_code=503,
                detail="Could not connect to MetaTrader 5 terminal. Verify MT5 is open and credentials in .env are valid.",
            )
        try:
            open_positions = connector.sync_open_positions(db)
            synced_count, needs_verdict_ids = connector.sync_closed_trades(db, lookback_days=lookback_days)
            status = connector.update_drawdown_status(db)
        except Exception as exc:
            logger.exception("MT5 sync error")
            raise HTTPException(status_code=500, detail=f"MT5 sync error: {exc}")
        finally:
            connector.disconnect()

    verdicts_generated = _grade_position_ids_background(needs_verdict_ids[:15])

    return SyncResponse(
        open_positions_count=len(open_positions),
        trades_synced=synced_count,
        auto_verdicts_generated=verdicts_generated,
        drawdown_status=status,
    )


@app.post("/trades/auto-grade-pending")
def trigger_auto_grade_pending(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Queue automatic GOOD_TRADE / BAD_TRADE grading for any ungraded closed trades in SQLite."""
    pending_ids = MT5SyncConnector.get_ungraded_closed_position_ids(db, limit=15)
    if pending_ids:
        background_tasks.add_task(_grade_position_ids_background, pending_ids)
    return {"queued_count": len(pending_ids), "position_ids": pending_ids}


@app.get("/positions/open", response_model=list[OpenPositionSummary])
def get_live_open_positions(
    background_tasks: BackgroundTasks,
    poll_mt5: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    if poll_mt5 and not is_fx_weekend_closed() and MT5_LOCK.acquire(timeout=1.5):
        try:
            connector = MT5SyncConnector()
            if connector.connect():
                try:
                    open_list = connector.sync_open_positions(db)
                    _, needs_verdict_ids = connector.sync_closed_trades(db, lookback_days=7)
                    connector.update_drawdown_status(db)
                finally:
                    connector.disconnect()

                if needs_verdict_ids:
                    background_tasks.add_task(_grade_position_ids_background, needs_verdict_ids[:10])

                return [OpenPositionSummary(**item) for item in open_list]
        except Exception as exc:
            logger.warning(f"Live MT5 poll fallback to SQLite cache: {exc}")
        finally:
            MT5_LOCK.release()

    rows = db.query(Trade).options(joinedload(Trade.journal_entry)).filter(Trade.status == "OPEN").all()
    results = []
    for t in rows:
        je = t.journal_entry
        results.append(
            OpenPositionSummary(
                mt5_position_id=t.mt5_position_id,
                symbol=t.symbol,
                direction=t.direction,
                volume=t.volume,
                entry_price=t.entry_price,
                stop_loss=t.stop_loss,
                take_profit=t.take_profit,
                floating_pnl=t.net_profit,
                risk_amount_usd=t.risk_amount_usd,
                open_time=t.open_time,
                stage1_completed=bool(je and je.stage1_completed_at is not None),
                setup_type=je.setup_type if je else None,
                entry_thesis=je.entry_thesis if je else None,
                invalidation_plan=je.invalidation_plan if je else None,
                pre_trade_emotion=je.pre_trade_emotion if je else None,
                entry_chart_path=je.entry_chart_path if je else None,
            )
        )
    return results


@app.post("/charts/upload")
def upload_chart_screenshot(file: UploadFile = File(...)):
    ext = Path(file.filename or "chart.png").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(status_code=400, detail="Unsupported image format. Upload PNG, JPG, or WEBP.")

    ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{ts_prefix}_{os.path.basename(file.filename or 'chart.png')}"
    dest_path = UPLOAD_DIR / safe_name

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"chart_image_path": str(dest_path)}


@app.post("/trades/{mt5_position_id}/stage1")
def submit_stage1_entry_questionnaire(
    mt5_position_id: int,
    payload: Stage1EntryRequest,
    db: Session = Depends(get_db),
):
    try:
        gatekeeper = DisciplineGatekeeperAgent()
        return gatekeeper.submit_stage1_entry(db, mt5_position_id=mt5_position_id, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Stage 1 questionnaire save failed")
        raise HTTPException(status_code=500, detail=f"Stage 1 save error: {exc}")


@app.post("/trades/{mt5_position_id}/verdict", response_model=ProcessOutcomeVerdict)
def run_process_outcome_verdict(mt5_position_id: int, db: Session = Depends(get_db)):
    try:
        gatekeeper = DisciplineGatekeeperAgent()
        return gatekeeper.evaluate_closed_trade_process(db, mt5_position_id=mt5_position_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Stage 2 process verdict failed")
        raise HTTPException(status_code=500, detail=f"Process verdict error: {exc}")


@app.get("/account/drawdown", response_model=DrawdownStatus)
def get_drawdown_status(db: Session = Depends(get_db)):
    rules = load_risk_rules()
    starting_balance = float(rules["account"]["starting_balance"])
    daily_limit_pct = float(rules["drawdown_limits"]["daily_loss_pct"])
    daily_soft_pct = float(rules["drawdown_limits"]["daily_soft_lock_pct"])
    static_limit_pct = float(rules["drawdown_limits"]["max_static_drawdown_pct"])
    static_soft_pct = float(rules["drawdown_limits"]["max_soft_lock_pct"])

    snapshot = (
        db.query(DailyAccountSnapshot)
        .order_by(DailyAccountSnapshot.trading_date.desc())
        .first()
    )

    if not snapshot:
        daily_limit_usd = round(starting_balance * (daily_limit_pct / 100.0), 2)
        static_floor_usd = round(starting_balance * (1.0 - static_limit_pct / 100.0), 2)
        return DrawdownStatus(
            trading_date="Not Synced",
            starting_account_balance=starting_balance,
            day_reference_level=starting_balance,
            current_equity=starting_balance,
            daily_loss_limit_usd=daily_limit_usd,
            daily_drawdown_remaining_usd=daily_limit_usd,
            daily_drawdown_used_pct=0.0,
            static_drawdown_floor_usd=static_floor_usd,
            static_drawdown_remaining_usd=round(starting_balance - static_floor_usd, 2),
            static_drawdown_used_pct=0.0,
            soft_lock_triggered=False,
            lock_reason=None,
        )

    day_ref = max(snapshot.day_start_balance, snapshot.day_start_equity)
    daily_limit_usd = round(day_ref * (daily_limit_pct / 100.0), 2)
    daily_floor = day_ref - daily_limit_usd
    daily_rem = round(max(0.0, snapshot.current_equity - daily_floor), 2)

    static_floor = round(starting_balance * (1.0 - static_limit_pct / 100.0), 2)
    static_rem = round(max(0.0, snapshot.current_equity - static_floor), 2)

    lock_reason = None
    if snapshot.daily_drawdown_used_pct >= daily_soft_pct:
        lock_reason = f"Daily drawdown ({snapshot.daily_drawdown_used_pct}%) reached soft-lock buffer ({daily_soft_pct}%)."
    elif snapshot.static_drawdown_used_pct >= static_soft_pct:
        lock_reason = f"Static drawdown ({snapshot.static_drawdown_used_pct}%) reached soft-lock buffer ({static_soft_pct}%)."

    return DrawdownStatus(
        trading_date=snapshot.trading_date,
        starting_account_balance=starting_balance,
        day_reference_level=round(day_ref, 2),
        current_equity=round(snapshot.current_equity, 2),
        daily_loss_limit_usd=daily_limit_usd,
        daily_drawdown_remaining_usd=daily_rem,
        daily_drawdown_used_pct=snapshot.daily_drawdown_used_pct,
        static_drawdown_floor_usd=static_floor,
        static_drawdown_remaining_usd=static_rem,
        static_drawdown_used_pct=snapshot.static_drawdown_used_pct,
        soft_lock_triggered=snapshot.is_locked_out,
        lock_reason=lock_reason,
    )


@app.get("/trades", response_model=list[TradeJournalSummary])
def list_trades(
    limit: int = Query(default=100, ge=1, le=500),
    symbol: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Trade).options(joinedload(Trade.journal_entry)).filter(Trade.status == "CLOSED")
    if symbol and symbol != "ALL":
        query = query.filter(Trade.symbol == symbol)

    trades = query.order_by(Trade.close_time.desc()).limit(limit).all()
    results = []
    for t in trades:
        je = t.journal_entry
        results.append(
            TradeJournalSummary(
                mt5_position_id=t.mt5_position_id,
                symbol=t.symbol,
                direction=t.direction,
                volume=t.volume,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                stop_loss=t.stop_loss,
                net_profit=t.net_profit,
                r_multiple=t.r_multiple,
                open_time=t.open_time,
                close_time=t.close_time,
                session=je.session if je else None,
                setup_type=je.setup_type if je else None,
                emotional_state=je.emotional_state if je else None,
                followed_plan=je.followed_plan if je else None,
                mistake_tags=je.mistake_tags if (je and je.mistake_tags) else [],
                entry_thesis=je.entry_thesis if je else None,
                process_verdict=je.process_verdict if je else None,
                outcome_quadrant=je.outcome_quadrant if je else None,
                ai_feedback=je.ai_feedback if je else None,
            )
        )
    return results


@app.post("/trades/{mt5_position_id}/journal", response_model=ParsedJournalNote)
def submit_trade_note(mt5_position_id: int, payload: JournalNoteRequest, db: Session = Depends(get_db)):
    try:
        agent = TradingJournalAgent()
        return agent.log_trade_note(db, mt5_position_id=mt5_position_id, raw_notes=payload.raw_notes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to parse journal note with Gemini")
        raise HTTPException(status_code=500, detail=f"Gemini agent error: {exc}")


@app.post("/trades/{mt5_position_id}/debrief", response_model=DebriefTurnResponse)
def interactive_trade_debrief(mt5_position_id: int, payload: DebriefTurnRequest, db: Session = Depends(get_db)):
    try:
        agent = TradingJournalAgent()
        return agent.run_debrief_turn(
            db,
            mt5_position_id=mt5_position_id,
            conversation=payload.conversation,
            force_finalize=payload.force_finalize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Interactive debrief failed")
        raise HTTPException(status_code=500, detail=f"Debrief error: {exc}")


@app.post("/journal/chat", response_model=JournalChatResponse)
def chat_with_journal(payload: JournalChatRequest, db: Session = Depends(get_db)):
    try:
        agent = TradingJournalAgent()
        reply = agent.chat_with_journal(db, messages=payload.messages, lookback_days=payload.lookback_days)
        return JournalChatResponse(reply=reply)
    except Exception as exc:
        logger.exception("Journal chat failed")
        raise HTTPException(status_code=500, detail=f"Journal chat error: {exc}")


@app.get("/analytics/metrics")
def get_analytics_metrics(lookback_days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    agent = TradingJournalAgent()
    return agent.compute_performance_metrics(db, lookback_days=lookback_days)


@app.post("/analytics/audit", response_model=AuditResponse)
def run_behavioral_audit(lookback_days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    try:
        agent = TradingJournalAgent()
        report = agent.run_leak_audit(db, lookback_days=lookback_days)
        return AuditResponse(lookback_days=lookback_days, audit_report=report)
    except Exception as exc:
        logger.exception("Failed to run behavioral leak audit")
        raise HTTPException(status_code=500, detail=f"Audit error: {exc}")


@app.post("/macro/refresh")
def refresh_macro_intelligence(db: Session = Depends(get_db)):
    try:
        agent = MacroCatalystAgent()
        return agent.refresh_macro_brief(db)
    except Exception as exc:
        logger.exception("Macro refresh failed")
        raise HTTPException(status_code=500, detail=f"Macro agent error: {exc}")


@app.get("/macro/latest")
def get_latest_macro_state(db: Session = Depends(get_db)):
    snapshot = (
        db.query(MacroRegimeSnapshot)
        .order_by(MacroRegimeSnapshot.created_at.desc())
        .first()
    )

    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    events = (
        db.query(MacroEvent)
        .filter(MacroEvent.scheduled_utc >= cutoff)
        .order_by(MacroEvent.scheduled_utc.asc())
        .limit(30)
        .all()
    )

    return {
        "snapshot": {
            "regime": snapshot.regime,
            "xauusd_bias": snapshot.xauusd_bias,
            "eurusd_bias": snapshot.eurusd_bias,
            "dxy_yield_context": snapshot.dxy_yield_context,
            "central_bank_summary": snapshot.central_bank_summary,
            "trading_guideline": snapshot.trading_guideline,
            "sources": snapshot.sources or [],
            "updated_at": snapshot.created_at.isoformat(),
        }
        if snapshot
        else None,
        "events": [
            {
                "id": e.id,
                "event_name": e.event_name,
                "currency": e.currency,
                "impact": e.impact,
                "scheduled_utc": e.scheduled_utc.isoformat(),
                "forecast": e.forecast,
                "previous": e.previous,
                "actual": e.actual,
                "affected_symbols": e.affected_symbols or [],
            }
            for e in events
        ],
    }


@app.get("/macro/blackout", response_model=NewsBlackoutStatus)
def get_news_blackout_status(symbol: Optional[str] = Query(default="ALL"), db: Session = Depends(get_db)):
    agent = MacroCatalystAgent()
    return agent.check_news_blackout(db, symbol=symbol)


@app.post("/macro/events")
def create_manual_macro_event(catalyst: EconomicCatalyst, db: Session = Depends(get_db)):
    try:
        agent = MacroCatalystAgent()
        ev = agent.add_manual_event(db, catalyst)
        return {
            "id": ev.id,
            "event_name": ev.event_name,
            "currency": ev.currency,
            "scheduled_utc": ev.scheduled_utc.isoformat(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to create manual macro event")
        raise HTTPException(status_code=500, detail=f"Manual event error: {exc}")


@app.delete("/macro/events/{event_id}")
def delete_macro_event(event_id: int, db: Session = Depends(get_db)):
    ev = db.query(MacroEvent).filter(MacroEvent.id == event_id).first()
    if not ev:
        raise HTTPException(status_code=404, detail=f"MacroEvent id={event_id} not found.")
    db.delete(ev)
    db.commit()
    return {"deleted_id": event_id}


@app.post("/gatekeeper/audit")
def run_pre_trade_gatekeeper_audit(payload: PreTradeAuditRequest, db: Session = Depends(get_db)):
    try:
        gatekeeper = DisciplineGatekeeperAgent()
        return gatekeeper.run_pre_trade_chart_audit(
            db=db,
            symbol=payload.symbol,
            intent_mode=payload.intent_mode,
            proposed_direction=payload.proposed_direction,
            timeframe=payload.timeframe,
            htf_timeframe=payload.htf_timeframe,
            proposed_risk_pct=payload.proposed_risk_pct,
            setup_description=payload.setup_description,
            chart_image_path=payload.chart_image_path,
            htf_chart_image_path=payload.htf_chart_image_path,
        )
    except Exception as exc:
        logger.exception("Chart Strategist audit failed")
        raise HTTPException(status_code=500, detail=f"Chart Strategist error: {exc}")


@app.get("/gatekeeper/history")
def get_gatekeeper_audit_history(limit: int = Query(default=15, ge=1, le=100), db: Session = Depends(get_db)):
    rows = (
        db.query(DisciplineAuditLog)
        .order_by(DisciplineAuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "created_at": format_utc_and_uae(r.created_at),
            "symbol": r.symbol,
            "intent_mode": r.intent_mode,
            "direction": r.proposed_direction,
            "timeframe": r.timeframe,
            "htf_timeframe": r.htf_timeframe,
            "risk_pct": r.proposed_risk_pct,
            "verdict": r.verdict,
            "recommended_action": r.recommended_action,
            "recommended_strategy": r.recommended_strategy,
            "confluences_count": r.confluences_count,
            "discipline_score": r.discipline_score,
            "coaching_verdict": r.coaching_verdict,
        }
        for r in rows
    ]