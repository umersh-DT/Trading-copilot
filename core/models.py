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
    """Stores raw execution telemetry synced from MetaTrader 5."""
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
    status: Mapped[str] = mapped_column(String(15), default="OPEN")  # "OPEN" or "CLOSED"

    journal_entry: Mapped[Optional["JournalEntry"]] = relationship(
        back_populates="trade", uselist=False, cascade="all, delete-orphan"
    )


class JournalEntry(Base):
    """Stores Gemini-parsed setup tags, psychological notes, and discipline grading."""
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[int] = mapped_column(ForeignKey("trades.id"), unique=True, nullable=False)

    session: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # ASIA, LONDON, NY, OVERLAP
    setup_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    timeframe: Mapped[Optional[str]] = mapped_column(String(15), nullable=True)

    raw_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emotional_state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    followed_plan: Mapped[bool] = mapped_column(Boolean, default=True)
    mistake_tags: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    ai_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    trade: Mapped["Trade"] = relationship(back_populates="journal_entry")


class DailyAccountSnapshot(Base):
    """Tracks The5ers daily starting balance/equity and drawdown headroom."""
    __tablename__ = "daily_account_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trading_date: Mapped[str] = mapped_column(String(10), unique=True, index=True, nullable=False)  # YYYY-MM-DD
    day_start_balance: Mapped[float] = mapped_column(Float, nullable=False)
    day_start_equity: Mapped[float] = mapped_column(Float, nullable=False)
    current_balance: Mapped[float] = mapped_column(Float, nullable=False)
    current_equity: Mapped[float] = mapped_column(Float, nullable=False)
    lowest_equity_today: Mapped[float] = mapped_column(Float, nullable=False)

    daily_drawdown_used_pct: Mapped[float] = mapped_column(Float, default=0.0)
    static_drawdown_used_pct: Mapped[float] = mapped_column(Float, default=0.0)
    is_locked_out: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)