from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """Stores live OPEN and historical CLOSED execution telemetry synced from MetaTrader 5."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mt5_position_id: Mapped[int] = mapped_column(Integer, unique=True, index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # "BUY" or "SELL"
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    commission: Mapped[float] = mapped_column(Float, default=0.0)
    swap: Mapped[float] = mapped_column(Float, default=0.0)
    gross_profit: Mapped[float] = mapped_column(Float, default=0.0)
    net_profit: Mapped[float] = mapped_column(Float, default=0.0)

    # Computed metrics
    risk_amount_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    r_multiple: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(15), index=True, default="OPEN")  # "OPEN" or "CLOSED"

    journal_entry: Mapped[Optional["JournalEntry"]] = relationship(
        back_populates="trade", uselist=False, cascade="all, delete-orphan"
    )


class JournalEntry(Base):
    """
    Two-Stage Execution Journal:
    - Stage 1 (In-Flight Entry Capture): Recorded while position is OPEN on MT5.
    - Stage 2 (Post-Close Verdict): Process-over-outcome grading (GOOD_TRADE vs BAD_TRADE) on close.
    """
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id"), unique=True, nullable=False)

    # Stage 1: In-Flight Entry Questionnaire fields
    entry_thesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    invalidation_plan: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pre_trade_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    macro_aligned: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    entry_chart_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    stage1_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Stage 2: Post-Close Structured Classification & Process Verdict
    session: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    setup_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    timeframe: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)

    raw_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emotional_state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    followed_plan: Mapped[bool] = mapped_column(Boolean, default=True)
    mistake_tags: Mapped[Optional[list]] = mapped_column(JSON, default=list)

    process_verdict: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # "GOOD_TRADE" or "BAD_TRADE"
    outcome_quadrant: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    ai_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    trade: Mapped["Trade"] = relationship(back_populates="journal_entry")


class DailyAccountSnapshot(Base):
    """Tracks The5ers daily starting balance/equity and drawdown headroom."""
    __tablename__ = "daily_account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trading_date: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)
    day_start_balance: Mapped[float] = mapped_column(Float, nullable=False)
    day_start_equity: Mapped[float] = mapped_column(Float, nullable=False)
    current_balance: Mapped[float] = mapped_column(Float, nullable=False)
    current_equity: Mapped[float] = mapped_column(Float, nullable=False)
    lowest_equity_today: Mapped[float] = mapped_column(Float, nullable=False)

    daily_drawdown_used_pct: Mapped[float] = mapped_column(Float, default=0.0)
    static_drawdown_used_pct: Mapped[float] = mapped_column(Float, default=0.0)
    is_locked_out: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class MacroEvent(Base):
    """Stores high-impact USD/EUR economic calendar releases tracked by Phase 2 Macro Agent."""
    __tablename__ = "macro_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_name: Mapped[str] = mapped_column(String(150), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    impact: Mapped[str] = mapped_column(String(15), default="HIGH")
    scheduled_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    forecast: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    previous: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    actual: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    affected_symbols: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MacroRegimeSnapshot(Base):
    """Stores periodic macro regime classifications and directional biases for XAUUSD & EURUSD."""
    __tablename__ = "macro_regime_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    regime: Mapped[str] = mapped_column(String(30), nullable=False)
    xauusd_bias: Mapped[str] = mapped_column(String(20), nullable=False)
    eurusd_bias: Mapped[str] = mapped_column(String(20), nullable=False)
    dxy_yield_context: Mapped[str] = mapped_column(Text, nullable=False)
    central_bank_summary: Mapped[str] = mapped_column(Text, nullable=False)
    trading_guideline: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DisciplineAuditLog(Base):
    """Stores Phase 3.5 Multimodal Chart Strategist (HTF + LTF ICT/Trendline) audits."""
    __tablename__ = "discipline_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    intent_mode: Mapped[str] = mapped_column(String(20), default="RECOMMEND")  # "RECOMMEND", "BUY", "SELL"
    proposed_direction: Mapped[str] = mapped_column(String(15), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(15), default="M15")
    htf_timeframe: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)
    proposed_risk_pct: Mapped[float] = mapped_column(Float, default=0.5)
    setup_description: Mapped[str] = mapped_column(Text, nullable=False)

    chart_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    htf_chart_image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    verdict: Mapped[str] = mapped_column(String(15), nullable=False)  # "PASS", "CAUTION", "BLOCK"
    recommended_action: Mapped[str] = mapped_column(String(30), default="WAIT")  # "BUY", "SELL", "WAIT"
    recommended_strategy: Mapped[str] = mapped_column(String(100), default="Wait for Confirmation")
    confluences_count: Mapped[int] = mapped_column(Integer, default=0)
    confluences_found: Mapped[Optional[list]] = mapped_column(JSON, default=list)

    discipline_score: Mapped[int] = mapped_column(Integer, default=50)  # 0-100
    htf_bias_read: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chart_structure_read: Mapped[str] = mapped_column(Text, nullable=False)
    trendline_ict_critique: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trigger_to_wait_for: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    invalidation_sl_zone: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    rule_violations: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    coaching_verdict: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)