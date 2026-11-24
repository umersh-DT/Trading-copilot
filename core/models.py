from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from core.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Trade(Base):
    """Canonical execution telemetry synced from MetaTrader 5."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    mt5_position_id = Column(BigInteger, unique=True, index=True, nullable=False)
    symbol = Column(String(20), index=True, nullable=False)
    direction = Column(String(10), nullable=False)  # BUY or SELL
    volume = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    open_time = Column(DateTime(timezone=True), nullable=False)
    close_time = Column(DateTime(timezone=True), nullable=True)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    gross_profit = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    risk_amount_usd = Column(Float, nullable=True)
    r_multiple = Column(Float, nullable=True)
    status = Column(String(16), default="OPEN", index=True)  # OPEN or CLOSED
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    journal_entry = relationship(
        "JournalEntry",
        back_populates="trade",
        uselist=False,
        cascade="all, delete-orphan",
    )


class JournalEntry(Base):
    """Two-stage behavioral context: Stage 1 pre-close entry thesis + Stage 2 process-over-outcome verdict."""

    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), unique=True, nullable=False)
    session = Column(String(32), index=True, nullable=True)
    setup_type = Column(String(64), index=True, nullable=True)
    timeframe = Column(String(20), nullable=True)
    entry_thesis = Column(Text, nullable=True)
    invalidation_plan = Column(Text, nullable=True)
    pre_trade_emotion = Column(String(32), nullable=True)
    macro_aligned = Column(Boolean, nullable=True)
    entry_chart_path = Column(String(255), nullable=True)
    stage1_completed_at = Column(DateTime(timezone=True), nullable=True)

    emotional_state = Column(String(32), index=True, nullable=True)
    followed_plan = Column(Boolean, default=True)
    mistake_tags = Column(JSON, default=list)
    raw_notes = Column(Text, nullable=True)
    process_verdict = Column(String(24), index=True, nullable=True)  # GOOD_TRADE | BAD_TRADE
    outcome_quadrant = Column(String(40), index=True, nullable=True)
    ai_feedback = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    trade = relationship("Trade", back_populates="journal_entry")


class DailyAccountSnapshot(Base):
    """Tracks daily starting equity/balance and drawdown headroom for The5ers rules."""

    __tablename__ = "daily_account_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    trading_date = Column(String(16), unique=True, index=True, nullable=False)
    day_start_balance = Column(Float, nullable=False)
    day_start_equity = Column(Float, nullable=False)
    current_balance = Column(Float, nullable=False)
    current_equity = Column(Float, nullable=False)
    lowest_equity_today = Column(Float, nullable=False)
    daily_drawdown_used_pct = Column(Float, default=0.0)
    static_drawdown_used_pct = Column(Float, default=0.0)
    is_locked_out = Column(Boolean, default=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class MacroEvent(Base):
    """Scheduled high-impact economic catalysts used for +/-30m news blackouts."""

    __tablename__ = "macro_events"

    id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String(160), nullable=False)
    currency = Column(String(12), index=True, nullable=False)
    impact = Column(String(16), index=True, nullable=False)
    scheduled_utc = Column(DateTime(timezone=True), index=True, nullable=False)
    forecast = Column(String(32), nullable=True)
    previous = Column(String(32), nullable=True)
    actual = Column(String(32), nullable=True)
    affected_symbols = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class MacroRegimeSnapshot(Base):
    """Grounded macro regime & directional bias summary for XAU/USD and EUR/USD."""

    __tablename__ = "macro_regime_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    regime = Column(String(32), nullable=False)
    xauusd_bias = Column(String(32), nullable=False)
    eurusd_bias = Column(String(32), nullable=False)
    dxy_yield_context = Column(Text, nullable=False)
    central_bank_summary = Column(Text, nullable=False)
    trading_guideline = Column(Text, nullable=False)
    sources = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)


class DisciplineAuditLog(Base):
    """Stores pre-trade Chart & Setup Advisor blueprints with Wave, Fib, Checklist & TP/SL levels."""

    __tablename__ = "discipline_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    intent_mode = Column(String(20), default="RECOMMEND")
    proposed_direction = Column(String(16), nullable=False)
    timeframe = Column(String(20), nullable=False)
    htf_timeframe = Column(String(20), nullable=True)
    proposed_risk_pct = Column(Float, nullable=False)
    setup_description = Column(Text, nullable=False)
    chart_image_path = Column(String(255), nullable=True)
    htf_chart_image_path = Column(String(255), nullable=True)
    verdict = Column(String(16), index=True, nullable=False)
    recommended_action = Column(String(16), default="WAIT")
    recommended_strategy = Column(String(100), default="Sit Out / Wait")
    confluences_count = Column(Integer, default=0)
    confluences_found = Column(JSON, default=list)
    strategy_checklist = Column(JSON, default=list)
    discipline_score = Column(Integer, nullable=False)
    htf_bias_read = Column(Text, nullable=True)
    wave_cycle_read = Column(Text, nullable=True)
    fib_levels_summary = Column(Text, nullable=True)
    chart_structure_read = Column(Text, nullable=True)
    trendline_ict_critique = Column(Text, nullable=True)
    trigger_to_wait_for = Column(Text, nullable=True)
    invalidation_sl_zone = Column(Text, nullable=True)
    tp1_zone = Column(Text, nullable=True)
    tp2_zone = Column(Text, nullable=True)
    rule_violations = Column(JSON, default=list)
    coaching_verdict = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)