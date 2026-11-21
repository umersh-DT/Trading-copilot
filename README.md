# The5ers Multi-Agent Trading Co-Pilot

A modular, multi-agent trading assistant built with **Google Gemini (`google-genai`)**, **MetaTrader 5**, **FastAPI**, and **Streamlit**. Engineered around **The5ers High Stakes** prop firm risk guardrails for trading **XAU/USD**, **EUR/USD**, **GBP/USD**, and **USD/JPY**, featuring full **UTC & UAE (UTC+4)** time localization, DST-aware session countdown alerts, live MT5 ATR/volume volatility tracking, resilient multi-model fallback chains, and dual-chart (`HTF + LTF`) ICT & trendline vision analysis.

---

## Architecture & Completed Roadmap

The system separates execution telemetry, interactive coaching, macro catalyst monitoring, live session/volatility tracking, and pre/post-trade rule enforcement into specialized agents sharing a local SQLite database:

| Phase | Agent / Module | Resilient Gemini Model Chain | Core Responsibilities | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **MT5 Telemetry & Core Journal** (`connectors/mt5_sync.py`, `core/`) | `gemini-4-hydrogen` → `gemini-3.8-flash` → `gemini-3.5-flash-lite` | Syncs MT5 live open positions and historical closed deals, computes R-multiples, and tracks live daily/static drawdown buffers. | **Complete** |
| **Phase 1.5** | **Interactive Debrief & Journal Chat** (`agents/journal_agent.py`) | `gemini-4-hydrogen` → `gemini-3.8-flash` → `gemini-3.5-flash-lite` | Multi-turn post-trade coaching interviews, fast single-pass natural-language Q&A over SQLite trade history, and deep behavioral leak audits. | **Complete** |
| **Phase 2** | **Macro & News Radar** (`agents/macro_agent.py`) | `gemini-4-hydrogen` → `gemini-3.8-flash` → `gemini-3.5-flash-lite` | Fetches weekly USD/EUR/GBP/JPY economic calendar events, classifies plain-English regimes & biases for XAU/USD + EUR/USD, and enforces `+/-30m` red-folder news blackouts. | **Complete** |
| **Phase 3** | **Two-Stage Execution Journal & Automatic Process Grader** (`agents/discipline_agent.py`) | `gemini-3.8-flash` → `gemini-3.5-flash-lite` → `gemini-4-hydrogen` | Polls live MT5 open trades every 15s for an instant **Entry Thesis Questionnaire**, automatically grades closed trades as **GOOD TRADE** vs. **BAD TRADE** on close/sync, and reveals full coaching explanations when clicking any row in the trade log. | **Complete** |
| **Phase 3.5** | **Chart Strategist & Live Session/Volatility Radar** (`agents/discipline_agent.py`, `connectors/mt5_sync.py`) | `gemini-4-argon` *(Vision)* → `gemini-3.8-flash` → `gemini-4-hydrogen` → `gemini-3.5-flash-lite` | Supports **Recommend Strategy** (when unsure) + **Audit BUY/SELL** modes, Dual-Chart (`1H/4H` + `15m/5m`) ICT & Trendline grading (`VALID` vs. `FORCED` + liquidity trap warnings), a **2-Confluence Minimum** rule for `PASS`, DST-aware **30-minute London/NY pre-open alerts** (with optional audio chime), and live MT5 15m ATR/tick-volume/ADR tracking. | **Complete** |

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
│   ├── models.py                # SQLAlchemy 2.0 ORM tables (Trades, Two-Stage Journal, Drawdown, Macro, Audits)
│   └── schemas.py               # Pydantic schemas for Structured Outputs & REST API
├── connectors/
│   └── mt5_sync.py              # MetaTrader 5 live/closed sync, drawdown engine, DST session radar & ATR/volume tracker
├── agents/
│   ├── journal_agent.py         # Phase 1 & 1.5: Interactive debrief, fast journal chat & Good/Bad Trade analytics
│   ├── macro_agent.py           # Phase 2: Economic calendar, plain-English bias & news blackout
│   └── discipline_agent.py      # Phase 3 & 3.5: Dual-chart ICT/Trendline Strategist & automatic process grader
├── api/
│   └── main.py                  # FastAPI REST backend with MT5 concurrency lock & background auto-grading (port 8000)
├── ui/
│   └── app.py                   # Streamlit 6-tab control center with 15s polling, session chimes & clickable trade log (port 8501)
├── start.ps1                    # 1-click PowerShell launcher with automatic port 8000/8501 cleanup
├── data/                        # Local SQLite database & uploaded chart screenshots (git-ignored)
├── logs/                        # Background FastAPI logs (git-ignored)
├── .env.example                 # Environment variable template
└── requirements.txt             # Python dependencies