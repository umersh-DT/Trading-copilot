import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from core.database import init_db, get_db, load_risk_rules
from core.models import Trade, DailyAccountSnapshot
from core.schemas import DrawdownStatus, ParsedJournalNote, TradeJournalSummary
from connectors.mt5_sync import MT5SyncConnector
from agents.journal_agent import TradingJournalAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized and FastAPI backend ready.")
    yield


app = FastAPI(
    title="The5ers Multi-Agent Trading Co-Pilot API",
    version="1.0.0",
    lifespan=lifespan,
)


class JournalNoteRequest(BaseModel):
    raw_notes: str = Field(..., min_length=3, description="Unstructured post-trade reflection or execution notes.")


class SyncResponse(BaseModel):
    trades_synced: int
    drawdown_status: DrawdownStatus


class AuditResponse(BaseModel):
    lookback_days: int
    audit_report: str


@app.get("/health")
def health_check():
    return {"status": "ok", "phase": "Phase 1 - Journal & Telemetry"}


@app.post("/sync/mt5", response_model=SyncResponse)
def trigger_mt5_sync(lookback_days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    """Connect to MetaTrader 5, sync closed deals, and update The5ers drawdown buffers."""
    connector = MT5SyncConnector()
    if not connector.connect():
        raise HTTPException(
            status_code=503,
            detail="Could not connect to MetaTrader 5 terminal. Verify MT5 is open and credentials in .env are valid.",
        )
    try:
        synced_count = connector.sync_closed_trades(db, lookback_days=lookback_days)
        status = connector.update_drawdown_status(db)
        return SyncResponse(trades_synced=synced_count, drawdown_status=status)
    finally:
        connector.disconnect()


@app.get("/account/drawdown", response_model=DrawdownStatus)
def get_drawdown_status(db: Session = Depends(get_db)):
    """Return latest cached The5ers High Stakes drawdown status from SQLite."""
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
        # Return baseline clean state before first MT5 sync
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
    """Fetch closed trades joined with their Gemini-parsed journal annotations."""
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
                net_profit=t.net_profit,
                r_multiple=t.r_multiple,
                open_time=t.open_time,
                close_time=t.close_time,
                session=je.session if je else None,
                setup_type=je.setup_type if je else None,
                emotional_state=je.emotional_state if je else None,
                followed_plan=je.followed_plan if je else None,
                mistake_tags=je.mistake_tags if (je and je.mistake_tags) else [],
                ai_feedback=je.ai_feedback if je else None,
            )
        )
    return results


@app.post("/trades/{mt5_position_id}/journal", response_model=ParsedJournalNote)
def submit_trade_note(mt5_position_id: int, payload: JournalNoteRequest, db: Session = Depends(get_db)):
    """Send raw trade notes to gemini-4-hydrogen for structured tagging and AI coaching feedback."""
    try:
        agent = TradingJournalAgent()
        return agent.log_trade_note(db, mt5_position_id=mt5_position_id, raw_notes=payload.raw_notes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to parse journal note with Gemini")
        raise HTTPException(status_code=500, detail=f"Gemini agent error: {exc}")


@app.get("/analytics/metrics")
def get_analytics_metrics(lookback_days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    """Return deterministic win rates, expectancy, and R-multiples across sessions, symbols, and setups."""
    agent = TradingJournalAgent()
    return agent.compute_performance_metrics(db, lookback_days=lookback_days)


@app.post("/analytics/audit", response_model=AuditResponse)
def run_behavioral_audit(lookback_days: int = Query(default=30, ge=1, le=365), db: Session = Depends(get_db)):
    """Trigger deep edge & behavioral leak diagnosis using gemini-4-hydrogen."""
    try:
        agent = TradingJournalAgent()
        report = agent.run_leak_audit(db, lookback_days=lookback_days)
        return AuditResponse(lookback_days=lookback_days, audit_report=report)
    except Exception as exc:
        logger.exception("Failed to run behavioral leak audit")
        raise HTTPException(status_code=500, detail=f"Audit error: {exc}")