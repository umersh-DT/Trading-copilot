from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ParsedJournalNote(BaseModel):
    """Structured output schema used by gemini-4-hydrogen to parse raw trade notes."""
    session: Literal["ASIA", "LONDON", "NEW_YORK", "LONDON_NY_OVERLAP", "UNKNOWN"] = Field(
        description="Trading session inferred from trade timestamp or user note."
    )
    setup_type: str = Field(
        description="Concise label for the technical setup (e.g., 'London Liquidity Sweep', 'FVG Retest', 'RSI Divergence')."
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Execution or confirmation timeframe mentioned (e.g., 'M15', 'H1', 'M5')."
    )
    emotional_state: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "FRUSTRATED", "NEUTRAL"] = Field(
        description="Dominant psychological state detected in the trader's note."
    )
    followed_plan: bool = Field(
        description="True if the trader respected their entry criteria, stop-loss, and risk limits; False if rules were broken."
    )
    mistake_tags: list[str] = Field(
        default_factory=list,
        description="List of specific execution errors (e.g., 'moved_sl', 'early_entry', 'oversized_lot', 'news_gamble'). Empty if clean execution."
    )
    ai_feedback: str = Field(
        description="Direct, objective 2-sentence coaching critique referencing the trade outcome and discipline."
    )


class DrawdownStatus(BaseModel):
    """Real-time The5ers High Stakes risk buffer telemetry."""
    trading_date: str
    starting_account_balance: float
    day_reference_level: float = Field(description="Higher of day-start balance or equity per The5ers rules.")
    current_equity: float
    daily_loss_limit_usd: float
    daily_drawdown_remaining_usd: float
    daily_drawdown_used_pct: float
    static_drawdown_floor_usd: float
    static_drawdown_remaining_usd: float
    static_drawdown_used_pct: float
    soft_lock_triggered: bool
    lock_reason: Optional[str] = None


class TradeJournalSummary(BaseModel):
    """Combined trade + journal payload served to the Streamlit dashboard."""
    mt5_position_id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: Optional[float]
    net_profit: float
    r_multiple: Optional[float]
    open_time: datetime
    close_time: Optional[datetime]
    session: Optional[str] = None
    setup_type: Optional[str] = None
    emotional_state: Optional[str] = None
    followed_plan: Optional[bool] = None
    mistake_tags: list[str] = Field(default_factory=list)
    ai_feedback: Optional[str] = None


class ChatMessage(BaseModel):
    """Single turn in a debrief or journal-chat conversation."""
    role: Literal["user", "model"]
    content: str


class DebriefTurnResponse(BaseModel):
    """Structured output returned by gemini-4-hydrogen during a post-trade debrief interview."""
    status: Literal["ASKING_FOLLOWUP", "COMPLETED"] = Field(
        description="Use 'ASKING_FOLLOWUP' if critical context (setup trigger, invalidation, or emotion) is still unclear; 'COMPLETED' once ready to save."
    )
    coach_message: str = Field(
        description="Either 1-2 sharp coaching questions for the trader, or a concise closing summary when status is COMPLETED."
    )
    final_entry: Optional[ParsedJournalNote] = Field(
        default=None,
        description="Populated ONLY when status is 'COMPLETED'."
    )


class JournalChatRequest(BaseModel):
    """Payload for natural-language Q&A against the trader's SQLite journal."""
    messages: list[ChatMessage]
    lookback_days: int = Field(default=60, ge=1, le=365)


class EconomicCatalyst(BaseModel):
    """Single high-impact economic release or central bank speech."""
    event_name: str = Field(description="Official release name, e.g., 'US Core CPI m/m', 'FOMC Rate Decision', 'US NFP'.")
    currency: Literal["USD", "EUR", "GBP", "JPY"]
    impact: Literal["HIGH", "MEDIUM"] = "HIGH"
    scheduled_utc: str = Field(description="ISO-8601 UTC timestamp, e.g., '2026-11-19T13:30:00Z'.")
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    affected_symbols: list[str] = Field(default_factory=lambda: ["XAUUSD", "EURUSD"])


class MacroRegimeReport(BaseModel):
    """Structured output produced by Phase 2 Macro Agent."""
    regime: Literal["RISK_ON", "RISK_OFF", "RATE_DRIVEN", "EVENT_VOLATILITY"]
    xauusd_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE_CHOP"]
    eurusd_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE_CHOP"]
    dxy_yield_context: str = Field(description="2-sentence summary of DXY, US 10Y yields, and real rates driving Gold & EUR.")
    central_bank_summary: str = Field(description="2-sentence summary of current Fed & ECB rate-path expectations.")
    trading_guideline: str = Field(description="Actionable risk/execution guidance for intraday FX & Gold setups today.")
    upcoming_catalysts: list[EconomicCatalyst] = Field(default_factory=list)


class NewsBlackoutStatus(BaseModel):
    """Real-time news blackout gatekeeper check against config/risk_rules.yaml buffer."""
    is_blackout_active: bool
    buffer_minutes: int
    active_or_imminent_event: Optional[str] = None
    minutes_to_event: Optional[int] = None
    message: str