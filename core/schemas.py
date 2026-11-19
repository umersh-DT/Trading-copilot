from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ParsedJournalNote(BaseModel):
    """Structured output schema used by Gemini to parse raw trade notes."""
    session: Literal["ASIA", "LONDON", "NEW_YORK", "LONDON_NY_OVERLAP", "UNKNOWN"] = Field(
        description="Trading session inferred from trade timestamp or user note."
    )
    setup_type: str = Field(
        description="Concise label for the technical setup (e.g., '15M Impulse Setup', 'London Liquidity Sweep', 'FVG Retest')."
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
    """Combined closed trade + two-stage journal payload served to the Streamlit dashboard."""
    mt5_position_id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float] = None
    net_profit: float
    r_multiple: Optional[float]
    open_time: datetime
    close_time: Optional[datetime]
    session: Optional[str] = None
    setup_type: Optional[str] = None
    emotional_state: Optional[str] = None
    followed_plan: Optional[bool] = None
    mistake_tags: list[str] = Field(default_factory=list)
    entry_thesis: Optional[str] = None
    process_verdict: Optional[str] = None
    outcome_quadrant: Optional[str] = None
    ai_feedback: Optional[str] = None


class OpenPositionSummary(BaseModel):
    """Live open MT5 position awaiting or holding its Stage 1 Entry Questionnaire."""
    mt5_position_id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    floating_pnl: float
    risk_amount_usd: Optional[float] = None
    open_time: datetime
    stage1_completed: bool = False
    setup_type: Optional[str] = None
    entry_thesis: Optional[str] = None
    invalidation_plan: Optional[str] = None
    pre_trade_emotion: Optional[str] = None
    entry_chart_path: Optional[str] = None


class Stage1EntryRequest(BaseModel):
    """Fast in-flight entry questionnaire submitted while a trade is open on MT5."""
    setup_type: str = Field(..., min_length=2, description="Setup pattern name, e.g. '15M Impulse RTB' or 'Liquidity Sweep'.")
    timeframe: str = Field(default="M15")
    entry_thesis: str = Field(..., min_length=3, description="Why was this trade taken right now?")
    invalidation_plan: str = Field(..., min_length=2, description="Where is your hard SL and what proves the setup wrong?")
    pre_trade_emotion: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "BORED"] = "CALM"
    macro_aligned: bool = True
    entry_chart_path: Optional[str] = None


class ProcessOutcomeVerdict(BaseModel):
    """Stage 2 structured verdict grading GOOD_TRADE vs BAD_TRADE regardless of P&L."""
    process_verdict: Literal["GOOD_TRADE", "BAD_TRADE"] = Field(
        description="'GOOD_TRADE' if rules, SL, risk <=1%, and entry thesis were respected; 'BAD_TRADE' if rules were broken."
    )
    outcome_quadrant: Literal[
        "GOOD_PROCESS_GOOD_OUTCOME",
        "GOOD_PROCESS_BAD_OUTCOME",
        "BAD_PROCESS_GOOD_OUTCOME",
        "BAD_PROCESS_BAD_OUTCOME",
    ]
    session: Literal["ASIA", "LONDON", "NEW_YORK", "LONDON_NY_OVERLAP", "UNKNOWN"]
    setup_type: str
    emotional_state: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "FRUSTRATED", "NEUTRAL"]
    followed_plan: bool
    mistake_tags: list[str] = Field(default_factory=list)
    ai_feedback: str = Field(
        description="Plain-English 3-sentence coaching explanation of why this was a GOOD TRADE or BAD TRADE regardless of profit or loss."
    )


class PreTradeAuditReport(BaseModel):
    """Structured output returned by Phase 3 Multimodal Chart & Discipline Gatekeeper."""
    verdict: Literal["PASS", "CAUTION", "BLOCK"] = Field(
        description="'PASS' if setup, chart structure, macro bias, news buffer, and risk align; 'CAUTION' for minor friction; 'BLOCK' if any hard rule or blackout is violated."
    )
    discipline_score: int = Field(ge=0, le=100, description="0-100 alignment score across chart structure, macro, risk, and historical edge.")
    chart_structure_read: str = Field(
        description="Plain-English 2-3 sentence technical read of the TradingView screenshot (trend, key levels, liquidity, entry/SL placement)."
    )
    rule_violations: list[str] = Field(
        default_factory=list,
        description="Specific rule conflicts or warnings (e.g., 'Inside +/-30m USD news blackout', 'Counter to BEARISH Gold macro bias', 'Risk exceeds 1% cap')."
    )
    coaching_verdict: str = Field(
        description="Direct, plain-English 2-sentence action recommendation before clicking Buy/Sell on MT5."
    )


class ChatMessage(BaseModel):
    """Single turn in a debrief or journal-chat conversation."""
    role: Literal["user", "model"]
    content: str


class DebriefTurnResponse(BaseModel):
    """Structured output returned during a post-trade debrief interview."""
    status: Literal["ASKING_FOLLOWUP", "COMPLETED"]
    coach_message: str
    final_entry: Optional[ParsedJournalNote] = None


class JournalChatRequest(BaseModel):
    """Payload for natural-language Q&A against the trader's SQLite journal."""
    messages: list[ChatMessage]
    lookback_days: int = Field(default=60, ge=1, le=365)


class EconomicCatalyst(BaseModel):
    """Single high-impact economic release or central bank speech."""
    event_name: str
    currency: Literal["USD", "EUR", "GBP", "JPY"]
    impact: Literal["HIGH", "MEDIUM"] = "HIGH"
    scheduled_utc: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    affected_symbols: list[str] = Field(default_factory=lambda: ["XAUUSD", "EURUSD"])


class MacroRegimeReport(BaseModel):
    """Structured output produced by Phase 2 Macro Agent."""
    regime: Literal["RISK_ON", "RISK_OFF", "RATE_DRIVEN", "EVENT_VOLATILITY"]
    xauusd_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE_CHOP"]
    eurusd_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE_CHOP"]
    dxy_yield_context: str
    central_bank_summary: str
    trading_guideline: str
    upcoming_catalysts: list[EconomicCatalyst] = Field(default_factory=list)


class NewsBlackoutStatus(BaseModel):
    """Real-time news blackout gatekeeper check against config/risk_rules.yaml buffer."""
    is_blackout_active: bool
    buffer_minutes: int
    active_or_imminent_event: Optional[str] = None
    minutes_to_event: Optional[int] = None
    message: str