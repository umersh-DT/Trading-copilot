import os
from pathlib import Path
from typing import Generator
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from core.models import Base

# Load environment variables from D:\Trading Agent\.env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'trading_journal.db'}")
RISK_RULES_PATH = BASE_DIR / "config" / "risk_rules.yaml"

# Ensure data directory exists for SQLite
if DATABASE_URL.startswith("sqlite:///"):
    db_file_path = DATABASE_URL.replace("sqlite:///", "", 1)
    Path(db_file_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode and foreign keys for concurrent Streamlit + FastAPI access."""
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_risk_rules() -> dict:
    """Load The5ers High Stakes rules from config/risk_rules.yaml."""
    if not RISK_RULES_PATH.exists():
        raise FileNotFoundError(f"Missing risk configuration file at {RISK_RULES_PATH}")
    with open(RISK_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)