import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# Ensure project root (D:\Trading Agent) is always on sys.path whether invoked via `pytest` or `python -m pytest`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base
from core.models import DailyAccountSnapshot, MacroEvent
from core.schemas import PreTradeAuditReport, StrategyChecklistItem
from agents.discipline_agent import DisciplineGatekeeperAgent, is_fx_weekend_closed
from agents.macro_agent import MacroCatalystAgent, format_duration_readable
from connectors.mt5_sync import MT5SyncConnector, fmt_duration_hm


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionTesting = sessionmaker(bind=engine)
    db = SessionTesting()
    try:
        yield db
    finally:
        db.close()


def _sample_audit_report(**overrides) -> PreTradeAuditReport:
    base = {
        "verdict": "PASS",
        "recommended_action": "SELL",
        "recommended_strategy": "5. Trendline Break & Retest",
        "analysis_mode": "DRAWN_CHART_CRITIQUE",
        "confluences_count": 3,
        "confluences_found": [
            "4H Bearish 20-EMA alignment",
            "1H ascending trendline break + retest",
            "Bearish 14-RSI divergence at swing high",
        ],
        "strategy_checklist": [
            StrategyChecklistItem(rule_name="Valid Geometry", status="PASS", detail="Touches 3 clean swing lows"),
            StrategyChecklistItem(rule_name="Impulsive Break", status="PASS", detail="Full-bodied 1H close below trendline"),
            StrategyChecklistItem(rule_name="Retest Rejection", status="PASS", detail="Bearish rejection wick at broken trendline"),
        ],
        "discipline_score": 88,
        "htf_bias_read": "4H bearish below 20-EMA.",
        "wave_cycle_read": "Wave C corrective bounce completed.",
        "fib_levels_summary": "0.618 Golden Zone at 3899; TP1 at 3593; TP2 at 3517.",
        "chart_structure_read": "Clean 1H breakdown structure.",
        "trendline_ict_critique": "Valid trendline break and retest.",
        "trigger_to_wait_for": "Bearish 15m engulfing close on retest.",
        "invalidation_sl_zone": "Above 3686.76 swing high.",
        "tp1_zone": "3593.86",
        "tp2_zone": "3517.39",
        "rule_violations": [],
        "coaching_verdict": "High-probability setup aligned with 4H trend.",
    }
    base.update(overrides)
    return PreTradeAuditReport(**base)


# ==============================================================================
# 1. DETERMINISTIC STRATEGY RULE GATE TESTS (PASS vs. CAUTION vs. BLOCK)
# ==============================================================================


def test_deterministic_gate_allows_clean_pass():
    agent = DisciplineGatekeeperAgent()
    report = _sample_audit_report()
    result = agent._enforce_deterministic_verdict(
        audit=report,
        sym="XAUUSD",
        allowed_symbols=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        proposed_risk_pct=0.5,
        max_risk_pct=1.0,
        dd_snap=None,
        blackout_active=False,
        blackout_msg="SAFE TO TRADE",
        adr_used_pct=45.0,
    )
    assert result.verdict == "PASS"
    assert result.rule_violations == []


def test_deterministic_gate_downgrades_to_caution_when_checklist_item_missing():
    agent = DisciplineGatekeeperAgent()
    report = _sample_audit_report(
        verdict="PASS",
        strategy_checklist=[
            StrategyChecklistItem(rule_name="Valid Geometry", status="PASS", detail="Valid 3-touch trendline"),
            StrategyChecklistItem(rule_name="Retest Rejection", status="MISSING", detail="Waiting for pullback to 3638"),
        ],
    )
    result = agent._enforce_deterministic_verdict(
        audit=report,
        sym="XAUUSD",
        allowed_symbols=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        proposed_risk_pct=0.5,
        max_risk_pct=1.0,
        dd_snap=None,
        blackout_active=False,
        blackout_msg="SAFE TO TRADE",
        adr_used_pct=50.0,
    )
    assert result.verdict == "CAUTION"
    assert any("Waiting for (Retest Rejection)" in v for v in result.rule_violations)


def test_deterministic_gate_blocks_on_disqualifier_or_risk_or_news_blackout():
    agent = DisciplineGatekeeperAgent()
    report = _sample_audit_report(
        strategy_checklist=[
            StrategyChecklistItem(rule_name="Wave 4 Overlap", status="FAIL", detail="Wave 4 entered Wave 1 price zone"),
        ]
    )
    locked_snap = DailyAccountSnapshot(
        trading_date="2026-11-26",
        day_start_balance=5000.0,
        day_start_equity=5000.0,
        current_balance=4800.0,
        current_equity=4800.0,
        lowest_equity_today=4800.0,
        daily_drawdown_used_pct=4.0,
        static_drawdown_used_pct=4.0,
        is_locked_out=True,
    )
    result = agent._enforce_deterministic_verdict(
        audit=report,
        sym="BTCUSD",
        allowed_symbols=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
        proposed_risk_pct=1.5,
        max_risk_pct=1.0,
        dd_snap=locked_snap,
        blackout_active=True,
        blackout_msg="USD CPI in 10m",
        adr_used_pct=50.0,
    )
    assert result.verdict == "BLOCK"
    assert len(result.rule_violations) >= 4


# ==============================================================================
# 2. THE5ERS DRAWDOWN MATH & SOFT-LOCK CIRCUIT BREAKER TESTS
# ==============================================================================


class _DummyAcc:
    def __init__(self, balance: float, equity: float):
        self.balance = balance
        self.equity = equity


def test_the5ers_daily_and_static_drawdown_soft_locks(in_memory_db):
    connector = MT5SyncConnector()
    with patch("connectors.mt5_sync.mt5.account_info", return_value=_DummyAcc(5000.0, 5100.0)):
        status_open = connector.update_drawdown_status(in_memory_db)
        assert status_open.day_reference_level == 5100.0
        assert status_open.soft_lock_triggered is False

    with patch("connectors.mt5_sync.mt5.account_info", return_value=_DummyAcc(4915.0, 4915.0)):
        status_locked = connector.update_drawdown_status(in_memory_db)
        assert status_locked.daily_drawdown_used_pct >= 3.5
        assert status_locked.soft_lock_triggered is True
        assert "soft-lock" in status_locked.lock_reason.lower()


# ==============================================================================
# 3. NEWS BLACKOUT (+/-30m) & HUMAN-READABLE DURATION TESTS
# ==============================================================================


def test_news_blackout_active_within_30_minutes(in_memory_db):
    agent = MacroCatalystAgent()
    imminent_time = datetime.now(timezone.utc) + timedelta(minutes=18)
    in_memory_db.add(
        MacroEvent(
            event_name="US Non-Farm Payrolls",
            currency="USD",
            impact="HIGH",
            scheduled_utc=imminent_time,
            affected_symbols=["XAUUSD", "EURUSD"],
        )
    )
    in_memory_db.commit()

    status = agent.check_news_blackout(in_memory_db, symbol="XAUUSD")
    assert status.is_blackout_active is True
    assert "NEWS BLACKOUT ACTIVE" in status.message


def test_news_blackout_safe_and_readable_duration(in_memory_db):
    agent = MacroCatalystAgent()
    future_time = datetime.now(timezone.utc) + timedelta(hours=54, minutes=30)
    in_memory_db.add(
        MacroEvent(
            event_name="Core PCE Price Index",
            currency="USD",
            impact="HIGH",
            scheduled_utc=future_time,
            affected_symbols=["XAUUSD", "EURUSD"],
        )
    )
    in_memory_db.commit()

    status = agent.check_news_blackout(in_memory_db, symbol="EURUSD")
    assert status.is_blackout_active is False
    assert "SAFE TO TRADE" in status.message
    assert "2d" in status.message
    assert format_duration_readable(135) == "2h 15m"
    assert fmt_duration_hm(135) == "2h 15m"


# ==============================================================================
# 4. SESSION RADAR & WEEKEND MARKET CLOSURE TESTS
# ==============================================================================


def test_session_radar_30m_london_pre_open_alert():
    thu_pre_ldn = datetime(2026, 11, 26, 7, 40, tzinfo=timezone.utc)
    status = MT5SyncConnector.get_session_countdown_status(thu_pre_ldn)
    assert status.pre_session_alert_active is True
    assert status.alert_severity == "WARNING"
    assert "20m" in status.banner_message


def test_weekend_market_closed_detection():
    sat_noon = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
    thu_morning = datetime(2026, 11, 26, 9, 40, tzinfo=timezone.utc)
    assert is_fx_weekend_closed(sat_noon) is True
    assert is_fx_weekend_closed(thu_morning) is False