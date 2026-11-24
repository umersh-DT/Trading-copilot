import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/trading_journal.db")
RISK_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "risk_rules.yaml"
STRATEGY_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "strategy_rules.yaml"

if DATABASE_URL.startswith("sqlite:///"):
    db_file = Path(DATABASE_URL.replace("sqlite:///", ""))
    db_file.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    future=True,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def load_risk_rules() -> dict:
    with open(RISK_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_strategy_rules() -> dict:
    if not STRATEGY_RULES_PATH.exists():
        return {"global_gates": {"min_confluences_for_pass": 2}, "strategies": {}}
    with open(STRATEGY_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_sqlite_migrations():
    """Safely add any missing columns to existing SQLite tables when upgrading in-place."""
    if "sqlite" not in DATABASE_URL:
        return
    migrations = [
        ("trades", "created_at", "DATETIME"),
        ("trades", "updated_at", "DATETIME"),
        ("journal_entries", "timeframe", "VARCHAR(20)"),
        ("journal_entries", "entry_thesis", "TEXT"),
        ("journal_entries", "invalidation_plan", "TEXT"),
        ("journal_entries", "pre_trade_emotion", "VARCHAR(32)"),
        ("journal_entries", "macro_aligned", "BOOLEAN"),
        ("journal_entries", "entry_chart_path", "VARCHAR(255)"),
        ("journal_entries", "stage1_completed_at", "DATETIME"),
        ("journal_entries", "process_verdict", "VARCHAR(24)"),
        ("journal_entries", "outcome_quadrant", "VARCHAR(40)"),
        ("journal_entries", "updated_at", "DATETIME"),
        ("discipline_audit_logs", "intent_mode", "VARCHAR(20) DEFAULT 'RECOMMEND'"),
        ("discipline_audit_logs", "htf_timeframe", "VARCHAR(20)"),
        ("discipline_audit_logs", "htf_chart_image_path", "VARCHAR(255)"),
        ("discipline_audit_logs", "recommended_action", "VARCHAR(16) DEFAULT 'WAIT'"),
        ("discipline_audit_logs", "recommended_strategy", "VARCHAR(100) DEFAULT 'Sit Out / Wait'"),
        ("discipline_audit_logs", "confluences_count", "INTEGER DEFAULT 0"),
        ("discipline_audit_logs", "confluences_found", "TEXT DEFAULT '[]'"),
        ("discipline_audit_logs", "strategy_checklist", "TEXT DEFAULT '[]'"),
        ("discipline_audit_logs", "htf_bias_read", "TEXT"),
        ("discipline_audit_logs", "wave_cycle_read", "TEXT"),
        ("discipline_audit_logs", "fib_levels_summary", "TEXT"),
        ("discipline_audit_logs", "chart_structure_read", "TEXT"),
        ("discipline_audit_logs", "trendline_ict_critique", "TEXT"),
        ("discipline_audit_logs", "trigger_to_wait_for", "TEXT"),
        ("discipline_audit_logs", "invalidation_sl_zone", "TEXT"),
        ("discipline_audit_logs", "tp1_zone", "TEXT"),
        ("discipline_audit_logs", "tp2_zone", "TEXT"),
    ]
    with engine.begin() as conn:
        for table_name, col_name, col_def in migrations:
            existing_cols = [
                row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            ]
            if existing_cols and col_name not in existing_cols:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"))


def init_db():
    import core.models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _run_sqlite_migrations()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()