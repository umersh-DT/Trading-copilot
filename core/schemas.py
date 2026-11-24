from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class DrawdownStatus(BaseModel):
    trading_date: str
    starting_account_balance: float
    day_reference_level: float
    current_equity: float
    daily_loss_limit_usd: float
    daily_drawdown_remaining_usd: float
    daily_drawdown_used_pct: float
    static_drawdown_floor_usd: float
    static_drawdown_remaining_usd: float
    static_drawdown_used_pct: float
    soft_lock_triggered: bool
    lock_reason: Optional[str] = None


class SessionCountdownStatus(BaseModel):
    current_utc_uae: str
    active_session: Literal["ASIA", "LONDON", "LONDON_NY_OVERLAP", "NEW_YORK", "DEAD_ZONE", "WEEKEND_CLOSED"]
    active_session_label: str
    is_dead_zone: bool
    next_event_name: str
    next_event_time_utc_uae: str
    minutes_until_next: int
    pre_session_alert_active: bool
    alert_severity: Literal["INFO", "WARNING", "DANGER"]
    banner_message: str


class SymbolVolatilityMetrics(BaseModel):
    symbol: str
    bid_price: float
    spread_points: int
    atr_15m_current: float
    atr_15m_avg20: float
    atr_expansion_ratio: float
    volume_current: int
    volume_avg20: float
    volume_surge_ratio: float
    today_range_pips: float
    adr_14d_pips: float
    adr_used_pct: float
    volatility_regime: Literal["DEAD_COMPRESSED", "NORMAL_HEALTHY", "HIGH_EXPANSION"]
    plain_english_summary: str


class Stage1EntryRequest(BaseModel):
    setup_type: str = Field(..., min_length=2, max_length=120)
    timeframe: str = Field(default="M15", max_length=20)
    entry_thesis: str = Field(..., min_length=4)
    invalidation_plan: str = Field(..., min_length=4)
    pre_trade_emotion: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "BORED"]
    macro_aligned: bool = True
    entry_chart_path: Optional[str] = None


class ProcessOutcomeVerdict(BaseModel):
    process_verdict: Literal["GOOD_TRADE", "BAD_TRADE"]
    outcome_quadrant: Literal[
        "GOOD_PROCESS_GOOD_OUTCOME",
        "GOOD_PROCESS_BAD_OUTCOME",
        "BAD_PROCESS_GOOD_OUTCOME",
        "BAD_PROCESS_BAD_OUTCOME",
    ]
    session: Literal["ASIA", "LONDON", "LONDON_NY_OVERLAP", "NEW_YORK", "UNKNOWN"]
    setup_type: str
    emotional_state: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "BORED", "UNKNOWN"]
    followed_plan: bool
    mistake_tags: list[str] = Field(default_factory=list)
    ai_feedback: str


class ParsedJournalNote(BaseModel):
    session: Literal["ASIA", "LONDON", "LONDON_NY_OVERLAP", "NEW_YORK", "UNKNOWN"]
    setup_type: str
    emotional_state: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "BORED", "UNKNOWN"]
    followed_plan: bool
    mistake_tags: list[str] = Field(default_factory=list)
    ai_feedback: str


class ChatMessage(BaseModel):
    role: Literal["user", "model"]
    content: str


class DebriefTurnResponse(BaseModel):
    coach_message: str
    is_complete: bool
    parsed_entry: Optional[ParsedJournalNote] = None


class JournalChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    lookback_days: int = Field(default=30, ge=1, le=365)


class OpenPositionSummary(BaseModel):
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


class TradeJournalSummary(BaseModel):
    mt5_position_id: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float]
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


class EconomicCatalyst(BaseModel):
    event_name: str
    currency: Literal["USD", "EUR", "GBP", "JPY", "OTHER"]
    impact: Literal["HIGH", "MEDIUM", "LOW"]
    scheduled_utc: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    affected_symbols: list[str] = Field(default_factory=list)


class MacroRegimeReport(BaseModel):
    regime: Literal["RISK_ON", "RISK_OFF", "RATE_DRIVEN", "EVENT_VOLATILITY"]
    xauusd_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE_CHOP"]
    eurusd_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE_CHOP"]
    dxy_yield_context: str
    central_bank_summary: str
    trading_guideline: str
    upcoming_catalysts: list[EconomicCatalyst] = Field(default_factory=list)


class NewsBlackoutStatus(BaseModel):
    is_blackout_active: bool
    buffer_minutes: int
    active_or_imminent_event: Optional[str] = None
    minutes_to_event: Optional[int] = None
    message: str


class StrategyChecklistItem(BaseModel):
    rule_name: str
    status: Literal["PASS", "MISSING", "FAIL"]
    detail: str


class PreTradeAuditReport(BaseModel):
    verdict: Literal["PASS", "CAUTION", "BLOCK"]
    recommended_action: Literal["BUY", "SELL", "WAIT"]
    recommended_strategy: str
    analysis_mode: Literal["DRAWN_CHART_CRITIQUE", "BLANK_CHART_MAPPING", "TEXT_ONLY"]
    confluences_count: int = Field(..., ge=0, le=10)
    confluences_found: list[str] = Field(default_factory=list)
    strategy_checklist: list[StrategyChecklistItem] = Field(default_factory=list)
    discipline_score: int = Field(..., ge=0, le=100)
    htf_bias_read: str
    wave_cycle_read: str
    fib_levels_summary: str
    chart_structure_read: str
    trendline_ict_critique: str
    trigger_to_wait_for: str
    invalidation_sl_zone: str
    tp1_zone: str
    tp2_zone: str
    rule_violations: list[str] = Field(default_factory=list)
    coaching_verdict: str