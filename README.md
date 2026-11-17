# The5ers Multi-Agent Trading Co-Pilot

A modular, multi-agent trading assistant built with **Google Gemini (`google-genai`)**, **MetaTrader 5**, **FastAPI**, and **Streamlit**. Designed specifically around **The5ers High Stakes** prop firm risk constraints for trading **XAU/USD**, **EUR/USD**, and major FX pairs.

---

## Architecture & Phased Roadmap

The system separates execution logging, macro regime awareness, and chart/rule enforcement into three specialized agents sharing a local SQLite database:

| Phase | Agent / Module | Gemini Model | Core Responsibilities | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Trading Journal & Telemetry Agent** (`agents/journal_agent.py`) | `gemini-4-hydrogen` | Syncs MT5 closed positions, tracks live daily/static drawdown buffers, parses post-trade notes into structured JSON, and audits behavioral leaks. | **Complete** |
| **Phase 2** | **Macro & Catalyst Monitor** (`agents/macro_agent.py`) | `gemini-4-hydrogen` *(with Search Grounding)* | Tracks high-impact USD/EUR economic releases (CPI, NFP, FOMC), classifies macro regimes (`Risk-On`, `Risk-Off`, `Rate-Driven`), and flags news blackout windows. | Planned |
| **Phase 3** | **Technical & Discipline Gatekeeper** (`agents/discipline_agent.py`) | `gemini-4-argon` | Audits TradingView chart screenshots + setups against your historical journal edge, macro bias, and risk rules before execution. | Planned |

---

## Prop Firm Guardrails (`config/risk_rules.yaml`)

All agents enforce hard circuit breakers tailored to **The5ers High Stakes**:

* **Daily Loss Limit:** `5.0%` max (anchored at `00:00` broker server time on `max(day_start_balance, day_start_equity)`).
  * **Soft-Lock Circuit Breaker:** Triggers at **`3.5%`** daily drawdown to protect against spread widening and slippage.
* **Maximum Static Drawdown:** `10.0%` below initial account balance.
  * **Soft-Lock Warning:** Triggers at **`7.5%`** total drawdown.
* **Position & Trade Limits:** Default `0.5%` risk per trade (`1.0%` hard cap), max `2` open positions, and max `2` daily losses.
* **Allowed Symbols:** `XAUUSD`, `EURUSD`, `GBPUSD`, `USDJPY`.

---

## Project Structure

```text
D:\Trading Agent\
├── config/
│   └── risk_rules.yaml          # The5ers High Stakes limits, sizing & session windows
├── core/
│   ├── database.py              # SQLite (WAL mode) engine & YAML rule loader
│   ├── models.py                # SQLAlchemy 2.0 ORM tables
│   └── schemas.py               # Pydantic schemas for Gemini Structured Outputs & API
├── connectors/
│   ├── mt5_sync.py              # MetaTrader 5 deal reconstruction & drawdown calculator
│   └── tv_webhook.py            # TradingView webhook listener (Phase 3)
├── agents/
│   ├── journal_agent.py         # Phase 1: Structured note parser & leak auditor
│   ├── macro_agent.py           # Phase 2: Economic calendar & macro regime monitor
│   └── discipline_agent.py      # Phase 3: Multimodal chart TA & rule gatekeeper
├── api/
│   └── main.py                  # FastAPI REST backend (port 8000)
├── ui/
│   └── app.py                   # Streamlit control center & analytics UI (port 8501)
├── data/                        # Local SQLite database (git-ignored)
├── .env.example                 # Environment variable template
└── requirements.txt             # Python dependencies