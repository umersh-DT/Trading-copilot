# The5ers Multi-Agent Trading Co-Pilot

A modular, multi-agent trading assistant built with **Google Gemini (`google-genai`)**, **MetaTrader 5**, **FastAPI**, and **Streamlit**. Engineered specifically around **The5ers High Stakes** prop firm risk guardrails for trading **XAU/USD**, **EUR/USD**, **GBP/USD**, and **USD/JPY**, with full **UTC & UAE (UTC+4)** time localization.

---

## Architecture & Phased Roadmap

The system separates execution logging, interactive coaching, macro catalyst tracking, and pre/post-trade rule enforcement into specialized agents sharing a local SQLite database:

| Phase | Agent / Module | Gemini Model Chain | Core Responsibilities | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **MT5 Telemetry & Core Journal** (`connectors/mt5_sync.py`, `core/`) | `gemini-4-hydrogen` | Syncs MT5 closed deals, reconstructs R-multiples, calculates live daily/static drawdown buffers, and parses post-trade notes. | **Complete** |
| **Phase 1.5** | **Interactive Debrief & Journal Chat** (`agents/journal_agent.py`) | `gemini-4-hydrogen` → `gemini-3.8-flash` → `gemini-3.5-flash-lite` | Multi-turn post-trade coaching interview, fast single-pass Q&A over SQLite trade history, and deep behavioral leak audits. | **Complete** |
| **Phase 2** | **Macro & News Radar** (`agents/macro_agent.py`) | `gemini-4-hydrogen` → `gemini-3.8-flash` → `gemini-3.5-flash-lite` | Fetches weekly USD/EUR/GBP/JPY economic calendar events, classifies plain-English market regimes & biases for XAU/USD + EUR/USD, and enforces +/-30m news blackouts. | **Complete** |
| **Phase 3** | **Two-Stage Execution Journal & Discipline Gatekeeper** (`agents/discipline_agent.py`) | `gemini-4-argon` *(Multimodal Vision)* | Auto-detects live MT5 open positions for an instant entry-thesis questionnaire, audits TradingView chart screenshots, and delivers a **Good Trade vs. Bad Trade** process verdict upon close. | **Next** |

---

## Prop Firm Guardrails (`config/risk_rules.yaml`)

All agents enforce hard circuit breakers tailored to **The5ers High Stakes**:

* **Daily Loss Limit:** `5.0%` max (anchored at `00:00` broker server time on `max(day_start_balance, day_start_equity)`).
  * **Soft-Lock Circuit Breaker:** Triggers at **`3.5%`** daily drawdown to protect against spread widening and slippage.
* **Maximum Static Drawdown:** `10.0%` fixed below initial account balance.
  * **Soft-Lock Warning:** Triggers at **`7.5%`** total drawdown.
* **Position & News Rules:** Default `0.5%` risk per trade (`1.0%` hard cap), max `2` open positions, max `2` daily losses, and a **`+/-30 minute` news blackout buffer** around high-impact releases.
* **Allowed Symbols:** `XAUUSD`, `EURUSD`, `GBPUSD`, `USDJPY`.

---

## Project Structure

```text
D:\Trading Agent\
├── config/
│   └── risk_rules.yaml          # The5ers High Stakes limits, sizing & session windows
├── core/
│   ├── database.py              # SQLite (WAL mode) engine & YAML rule loader
│   ├── models.py                # SQLAlchemy 2.0 ORM tables (Trades, Journal, Drawdown, Macro)
│   └── schemas.py               # Pydantic schemas for Structured Outputs & REST API
├── connectors/
│   ├── mt5_sync.py              # MetaTrader 5 deal reconstruction & drawdown calculator
│   └── tv_webhook.py            # TradingView webhook listener (Phase 3)
├── agents/
│   ├── journal_agent.py         # Phase 1 & 1.5: Interactive debrief, journal chat & leak audit
│   ├── macro_agent.py           # Phase 2: Economic calendar, plain-English bias & news blackout
│   └── discipline_agent.py      # Phase 3: Multimodal chart TA & Good/Bad trade gatekeeper
├── api/
│   └── main.py                  # FastAPI REST backend (port 8000)
├── ui/
│   └── app.py                   # Streamlit 5-tab dashboard with UTC + UAE (UTC+4) clocks (port 8501)
├── start.ps1                    # 1-click PowerShell launcher with auto port cleanup
├── data/                        # Local SQLite database (git-ignored)
├── logs/                        # Background FastAPI logs (git-ignored)
├── .env.example                 # Environment variable template
└── requirements.txt             # Python dependencies