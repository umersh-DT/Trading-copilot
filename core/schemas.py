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


class SessionCountdownStatus(BaseModel):
    """DST-aware trading session status and 30-minute pre-session countdown banner."""
    current_utc_uae: str
    active_session: str
    active_session_label: str
    is_dead_zone: bool = False
    next_event_name: str
    next_event_time_utc_uae: str
    minutes_until_next: int
    pre_session_alert_active: bool = False
    alert_severity: Literal["INFO", "WARNING", "DANGER"] = "INFO"
    banner_message: str


class SymbolVolatilityMetrics(BaseModel):
    """Live MT5 M15 ATR expansion, tick volume surge, and 14-day ADR exhaustion telemetry."""
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
    setup_type: str = Field(..., min_length=2)
    timeframe: str = Field(default="M15")
    entry_thesis: str = Field(..., min_length=3)
    invalidation_plan: str = Field(..., min_length=2)
    pre_trade_emotion: Literal["CALM", "CONFIDENT", "FOMO", "REVENGE", "HESITANT", "BORED"] = "CALM"
    macro_aligned: bool = True
    entry_chart_path: Optional[str] = None


class ProcessOutcomeVerdict(BaseModel):
    """Stage 2 structured verdict grading GOOD_TRADE vs BAD_TRADE regardless of P&L."""
    process_verdict: Literal["GOOD_TRADE", "BAD_TRADE"]
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
    ai_feedback: str


class PreTradeAuditReport(BaseModel):
    """Structured output returned by Phase 3.5 Chart Strategist (HTF + LTF ICT & Trendline Grader)."""
    verdict: Literal["PASS", "CAUTION", "BLOCK"] = Field(
        description="'PASS' ONLY when 2+ aligned confluences exist AND zero hard risk/news/HTF conflicts occur; 'CAUTION' if <2 confluences or counter-macro; 'BLOCK' if rule/blackout violated."
    )
    recommended_action: Literal["BUY", "SELL", "WAIT"] = Field(
        description="Concrete directional action recommended from chart structure ('BUY', 'SELL', or 'WAIT')."
    )
    recommended_strategy: str = Field(
        description="Matched playbook setup (e.g., 'ICT Liquidity Sweep + FVG Reversal', 'Trendline Break & Retest', '15M Impulse Continuation', or 'Sit Out — Messy Chop')."
    )
    analysis_mode: Literal["DRAWN_CHART_CRITIQUE", "BLANK_CHART_MAPPING", "TEXT_ONLY"] = Field(
        description="Whether user trendlines/boxes were critiqued ('DRAWN_CHART_CRITIQUE') or clean structure was mapped ('BLANK_CHART_MAPPING')."
    )
    confluences_count: int = Field(ge=0, le=6, description="Exact count of independent confluences aligning in the recommended direction.")
    confluences_found: list[str] = Field(
        default_factory=list,
        description="Explicit list of aligned confluences (e.g., '1H Discount + Bullish OB', 'Asian Low Liquidity Sweep + 15m MSS', 'Valid Trendline Break & Retest', '15M Impulse RTB + RSI Bullish Divergence')."
    )
    discipline_score: int = Field(ge=0, le=100, description="0-100 overall setup & risk quality score.")
    htf_bias_read: str = Field(
        description="Plain-English Higher-Timeframe (1H/4H) dealing range (Premium vs. Discount) and draw on liquidity read."
    )
    chart_structure_read: str = Field(
        description="Plain-English 2-3 sentence read of the Execution chart (15m/5m) liquidity pools, FVGs, Order Blocks, and momentum."
    )
    trendline_ict_critique: str = Field(
        description="Direct critique of user-drawn trendlines (VALID anchor vs FORCED cutting candles + Trendline Liquidity stop-hunt warning) OR mapped ICT zones if blank."
    )
    trigger_to_wait_for: str = Field(
        description="Exact candle close, sweep, or retest trigger required before clicking Buy/Sell."
    )
    invalidation_sl_zone: str = Field(
        description="Exact structural invalidation price/zone where the Stop Loss belongs to stay within 1% The5ers risk."
    )
    rule_violations: list[str] = Field(
        default_factory=list,
        description="Specific rule conflicts, drawdown locks, news blackouts, or missing confluences."
    )
    coaching_verdict: str = Field(
        description="Direct, plain-English 2-sentence action summary for the trader."
    )


class ChatMessage(BaseModel):
    role: Literal["user", "model"]
    content: str


class DebriefTurnResponse(BaseModel):
    status: Literal["ASKING_FOLLOWUP", "COMPLETED"]
    coach_message: str
    final_entry: Optional[ParsedJournalNote] = None


class JournalChatRequest(BaseModel):
    messages: list[ChatMessage]
    lookback_days: int = Field(default=60, ge=1, le=365)


class EconomicCatalyst(BaseModel):
    event_name: str
    currency: Literal["USD", "EUR", "GBP", "JPY"]
    impact: Literal["HIGH", "MEDIUM"] = "HIGH"
    scheduled_utc: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    affected_symbols: list[str] = Field(default_factory=lambda: ["XAUUSD", "EURUSD"])


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