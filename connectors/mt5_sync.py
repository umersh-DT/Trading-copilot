import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
from dotenv import load_dotenv
from sqlalchemy.orm import Session, joinedload

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import Trade, DailyAccountSnapshot
from core.schemas import (
    DrawdownStatus,
    SessionCountdownStatus,
    SymbolVolatilityMetrics,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mt5_sync")

UAE_TZ = timezone(timedelta(hours=4))
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")


def fmt_utc_uae_dt(dt: datetime) -> str:
    utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    uae_dt = utc_dt.astimezone(UAE_TZ)
    return f"{utc_dt.strftime('%H:%M')} UTC ({uae_dt.strftime('%H:%M')} UAE)"


def fmt_duration_hm(total_minutes: int) -> str:
    """Format minutes into clean 'Xh Ym' or 'Ym' strings (never raw thousands of minutes)."""
    mins = max(1, int(total_minutes))
    hours, rem_mins = divmod(mins, 60)
    if hours == 0:
        return f"{rem_mins}m"
    if rem_mins == 0:
        return f"{hours}h"
    return f"{hours}h {rem_mins}m"


class MT5SyncConnector:
    """Connects to MetaTrader 5 terminal, syncs live OPEN + CLOSED positions, tracks drawdown, and measures session conditions."""

    def __init__(self):
        self.login = int(os.getenv("MT5_LOGIN", "0"))
        self.password = os.getenv("MT5_PASSWORD", "")
        self.server = os.getenv("MT5_SERVER", "")
        self.terminal_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
        self.rules = load_risk_rules()

    def connect(self) -> bool:
        init_kwargs = {}
        if self.terminal_path and os.path.exists(self.terminal_path):
            init_kwargs["path"] = self.terminal_path

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
            return False

        if self.login and self.password and self.server:
            authorized = mt5.login(login=self.login, password=self.password, server=self.server)
            if not authorized:
                logger.error(f"MT5 login failed for account {self.login}, error: {mt5.last_error()}")
                mt5.shutdown()
                return False

        acc = mt5.account_info()
        if acc is None:
            logger.error("Failed to retrieve MT5 account_info().")
            mt5.shutdown()
            return False

        return True

    def disconnect(self):
        mt5.shutdown()

    # ==========================================================================
    # DST-AWARE SESSION RADAR & 30-MINUTE PRE-SESSION COUNTDOWN (Xh Ym FORMAT)
    # ==========================================================================

    @staticmethod
    def get_session_countdown_status(now_utc: Optional[datetime] = None) -> SessionCountdownStatus:
        utc_now = now_utc or datetime.now(timezone.utc)
        if utc_now.tzinfo is None:
            utc_now = utc_now.replace(tzinfo=timezone.utc)

        wd = utc_now.weekday()  # Mon=0 ... Fri=4, Sat=5, Sun=6
        is_weekend = (wd == 5) or (wd == 4 and utc_now.hour >= 22) or (wd == 6 and utc_now.hour < 22)

        if is_weekend:
            # Next market reopen is Sunday 22:00 UTC
            days_until_sun = (6 - wd) % 7
            reopen_utc = (utc_now + timedelta(days=days_until_sun)).replace(
                hour=22, minute=0, second=0, microsecond=0
            )
            if reopen_utc <= utc_now:
                reopen_utc += timedelta(days=7)
            mins_until = max(1, int(round((reopen_utc - utc_now).total_seconds() / 60.0)))
            countdown_str = fmt_duration_hm(mins_until)
            reopen_str = fmt_utc_uae_dt(reopen_utc)
            return SessionCountdownStatus(
                current_utc_uae=fmt_utc_uae_dt(utc_now),
                active_session="WEEKEND_CLOSED",
                active_session_label="⏸️ Weekend — Markets Closed",
                is_dead_zone=True,
                next_event_name="Market Open (Asia Session)",
                next_event_time_utc_uae=reopen_str,
                minutes_until_next=mins_until,
                pre_session_alert_active=False,
                alert_severity="INFO",
                banner_message=(
                    f"⏸️ **Weekend — Markets Closed** | Market reopens in **{countdown_str}** (`{reopen_str}`). "
                    f"Great time to review charts and plan next week's levels."
                ),
            )

        ldn_now = utc_now.astimezone(LONDON_TZ)
        ny_now = utc_now.astimezone(NY_TZ)

        ldn_open_utc = ldn_now.replace(hour=8, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        ldn_close_utc = ldn_now.replace(hour=17, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        ny_open_utc = ny_now.replace(hour=8, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        ny_close_utc = ny_now.replace(hour=17, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        asia_open_utc = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

        is_ldn_active = ldn_open_utc <= utc_now < ldn_close_utc
        is_ny_active = ny_open_utc <= utc_now < ny_close_utc
        is_asia_active = asia_open_utc <= utc_now < ldn_open_utc

        if is_ldn_active and is_ny_active:
            active_session = "LONDON_NY_OVERLAP"
            active_label = "🔥 London + New York Overlap (Peak Activity)"
            is_dead = False
        elif is_ldn_active:
            active_session = "LONDON"
            active_label = "🇬🇧 London Session Active"
            is_dead = False
        elif is_ny_active:
            active_session = "NEW_YORK"
            active_label = "🇺🇸 New York Session Active"
            is_dead = False
        elif is_asia_active:
            active_session = "ASIA"
            active_label = "🌏 Asia Session (Range Building)"
            is_dead = False
        else:
            active_session = "DEAD_ZONE"
            active_label = "⏸️ Quiet Rollover Window"
            is_dead = True

        milestones = [
            ("London Session Open", ldn_open_utc),
            ("New York Open (Peak Overlap)", ny_open_utc),
            ("London Session Close", ldn_close_utc),
            ("New York Close / Rollover", ny_close_utc),
            ("Next London Session Open", ldn_open_utc + timedelta(days=1)),
        ]
        future_milestones = [(name, ts) for name, ts in milestones if ts > utc_now]
        next_name, next_ts = min(future_milestones, key=lambda x: x[1])
        mins_until = max(1, int(round((next_ts - utc_now).total_seconds() / 60.0)))
        countdown_str = fmt_duration_hm(mins_until)
        next_time_str = fmt_utc_uae_dt(next_ts)

        pre_alert = False
        severity = "INFO"
        banner_msg = (
            f"**{active_label}** | Next: **{next_name}** in **{countdown_str}** (`{next_time_str}`)."
        )

        if "New York Open" in next_name and mins_until <= 30:
            pre_alert = True
            severity = "WARNING"
            banner_msg = (
                f"⚡ **New York Session starts in {countdown_str}** (`{next_time_str}`) — "
                f"London–NY Overlap ahead! Expect fast moves and stop-hunts; manage risk carefully."
            )
        elif "London Session Open" in next_name and mins_until <= 30:
            pre_alert = True
            severity = "WARNING"
            banner_msg = (
                f"🔔 **London Session starts in {countdown_str}** (`{next_time_str}`) — "
                f"Get ready! Mark Asian High/Low levels and check today's market direction."
            )
        elif "London Session Close" in next_name and mins_until <= 30:
            pre_alert = True
            severity = "INFO"
            banner_msg = (
                f"🌇 **London Session closes in {countdown_str}** (`{next_time_str}`) — "
                f"Watch for profit-taking pullbacks before the New York afternoon."
            )
        elif is_dead:
            severity = "DANGER"
            banner_msg = (
                f"⚠️ **Quiet Rollover Window** — Spreads are wider right now. "
                f"Next major open: **{next_name}** in **{countdown_str}** (`{next_time_str}`)."
            )

        return SessionCountdownStatus(
            current_utc_uae=fmt_utc_uae_dt(utc_now),
            active_session=active_session,
            active_session_label=active_label,
            is_dead_zone=is_dead,
            next_event_name=next_name,
            next_event_time_utc_uae=next_time_str,
            minutes_until_next=mins_until,
            pre_session_alert_active=pre_alert,
            alert_severity=severity,
            banner_message=banner_msg,
        )

    # ==========================================================================
    # PLAIN-ENGLISH LIVE MARKET SPEED & DAILY MOVE LEFT TRACKER
    # ==========================================================================

    def get_symbol_volatility(self, symbol: str) -> Optional[SymbolVolatilityMetrics]:
        sym = symbol.upper().strip()
        sym_info = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        if sym_info is None or tick is None or sym_info.point == 0:
            return None

        m15_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 35)
        d1_rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 15)
        if m15_rates is None or len(m15_rates) < 22 or d1_rates is None or len(d1_rates) < 5:
            return None

        tr_list = []
        for i in range(1, len(m15_rates)):
            high = float(m15_rates[i]["high"])
            low = float(m15_rates[i]["low"])
            prev_close = float(m15_rates[i - 1]["close"])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)

        atr_series = []
        period = 14
        for idx in range(period, len(tr_list) + 1):
            atr_series.append(sum(tr_list[idx - period : idx]) / period)

        atr_current = atr_series[-1]
        atr_avg20 = sum(atr_series[-20:]) / min(20, len(atr_series))
        atr_ratio = round(atr_current / atr_avg20, 2) if atr_avg20 > 0 else 1.0

        vols = [int(r["tick_volume"]) for r in m15_rates[-21:-1]]
        vol_avg20 = sum(vols) / len(vols) if vols else 1.0
        vol_current = max(int(m15_rates[-1]["tick_volume"]), int(m15_rates[-2]["tick_volume"]))
        vol_ratio = round(vol_current / vol_avg20, 2) if vol_avg20 > 0 else 1.0

        pip_size = sym_info.point * 10.0

        today_bar = d1_rates[-1]
        today_range_pips = round((float(today_bar["high"]) - float(today_bar["low"])) / pip_size, 1)
        past_days = d1_rates[:-1]
        adr_14d_pips = round(
            sum((float(b["high"]) - float(b["low"])) / pip_size for b in past_days) / len(past_days),
            1,
        )
        adr_used_pct = round((today_range_pips / adr_14d_pips) * 100.0, 1) if adr_14d_pips > 0 else 0.0
        room_left_pct = max(0, int(round(100.0 - adr_used_pct)))

        # Non-repetitive, plain-English single-sentence trading guidance
        if atr_ratio < 0.72 and vol_ratio < 0.8:
            regime = "DEAD_COMPRESSED"
            summary = "😴 Quiet market — wait for London/NY volume before entering."
        elif room_left_pct <= 20:
            regime = "HIGH_EXPANSION" if (atr_ratio >= 1.35 or vol_ratio >= 1.5) else "NORMAL_HEALTHY"
            summary = f"⚠️ Only ~{room_left_pct}% of today's typical move is left — take profits quicker and avoid chasing."
        elif atr_ratio >= 1.35 or vol_ratio >= 1.5:
            regime = "HIGH_EXPANSION"
            summary = f"🔥 Fast momentum & heavy volume — stick to your stop loss and don't over-size."
        else:
            regime = "NORMAL_HEALTHY"
            summary = f"🟢 Clean movement with ~{room_left_pct}% of today's typical range still available."

        return SymbolVolatilityMetrics(
            symbol=sym,
            bid_price=round(float(tick.bid), sym_info.digits),
            spread_points=int(sym_info.spread),
            atr_15m_current=round(atr_current, sym_info.digits),
            atr_15m_avg20=round(atr_avg20, sym_info.digits),
            atr_expansion_ratio=atr_ratio,
            volume_current=vol_current,
            volume_avg20=round(vol_avg20, 1),
            volume_surge_ratio=vol_ratio,
            today_range_pips=today_range_pips,
            adr_14d_pips=adr_14d_pips,
            adr_used_pct=adr_used_pct,
            volatility_regime=regime,
            plain_english_summary=summary,
        )

    def get_market_volatility(self, symbols: Optional[list[str]] = None) -> list[SymbolVolatilityMetrics]:
        target_symbols = symbols or ["XAUUSD", "EURUSD"]
        results = []
        for s in target_symbols:
            metrics = self.get_symbol_volatility(s)
            if metrics:
                results.append(metrics)
        return results

    # ==========================================================================
    # MT5 OPEN + CLOSED POSITION SYNC (WITH HISTORY ORDERS SL/TP RECONSTRUCTION)
    # ==========================================================================

    def _estimate_sl_risk_usd(self, symbol: str, entry_price: float, stop_loss: Optional[float], volume: float) -> Optional[float]:
        if not stop_loss or stop_loss <= 0:
            return None

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None or sym_info.trade_tick_size == 0:
            return None

        price_diff = abs(entry_price - stop_loss)
        ticks = price_diff / sym_info.trade_tick_size
        risk_usd = ticks * sym_info.trade_tick_value * volume
        return round(risk_usd, 2)

    def _build_order_sl_tp_map(self, utc_from: datetime, utc_to: datetime) -> dict[int, dict]:
        orders = mt5.history_orders_get(utc_from, utc_to)
        if not orders:
            logger.warning(f"mt5.history_orders_get() returned empty/None: {mt5.last_error()}")
            return {}

        grouped: dict[int, list] = {}
        for o in orders:
            pos_id = getattr(o, "position_id", 0) or 0
            if pos_id > 0:
                grouped.setdefault(pos_id, []).append(o)

        sl_tp_map: dict[int, dict] = {}
        for pos_id, pos_orders in grouped.items():
            pos_orders.sort(key=lambda x: (getattr(x, "time_setup", 0), getattr(x, "ticket", 0)))
            sl_values = [round(float(o.sl), 5) for o in pos_orders if getattr(o, "sl", 0) and float(o.sl) > 0]
            tp_values = [round(float(o.tp), 5) for o in pos_orders if getattr(o, "tp", 0) and float(o.tp) > 0]

            sl_tp_map[pos_id] = {
                "initial_sl": sl_values[0] if sl_values else None,
                "final_sl": sl_values[-1] if sl_values else None,
                "initial_tp": tp_values[0] if tp_values else None,
                "final_tp": tp_values[-1] if tp_values else None,
            }

        return sl_tp_map

    def sync_open_positions(self, db: Session) -> list[dict]:
        positions = mt5.positions_get()
        if positions is None:
            logger.warning(f"mt5.positions_get() returned None: {mt5.last_error()}")
            return []

        open_summaries = []

        for pos in positions:
            ticket = int(pos.ticket)
            direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            open_dt = datetime.fromtimestamp(pos.time, tz=timezone.utc)
            sl_val = float(pos.sl) if pos.sl and pos.sl > 0 else None
            tp_val = float(pos.tp) if pos.tp and pos.tp > 0 else None
            risk_usd = self._estimate_sl_risk_usd(pos.symbol, float(pos.price_open), sl_val, float(pos.volume))

            existing = db.query(Trade).filter(Trade.mt5_position_id == ticket).first()
            if existing:
                existing.status = "OPEN"
                existing.stop_loss = sl_val
                existing.take_profit = tp_val
                existing.gross_profit = round(float(pos.profit), 2)
                existing.net_profit = round(float(pos.profit) + float(pos.swap), 2)
                if risk_usd is not None:
                    existing.risk_amount_usd = risk_usd
            else:
                existing = Trade(
                    mt5_position_id=ticket,
                    symbol=pos.symbol,
                    direction=direction,
                    volume=float(pos.volume),
                    entry_price=float(pos.price_open),
                    stop_loss=sl_val,
                    take_profit=tp_val,
                    open_time=open_dt,
                    swap=round(float(pos.swap), 2),
                    gross_profit=round(float(pos.profit), 2),
                    net_profit=round(float(pos.profit) + float(pos.swap), 2),
                    risk_amount_usd=risk_usd,
                    status="OPEN",
                )
                db.add(existing)
                logger.info(f"Detected new LIVE MT5 position #{ticket} ({pos.symbol} {direction} {pos.volume} lots)")

            db.flush()
            je = existing.journal_entry
            open_summaries.append(
                {
                    "mt5_position_id": ticket,
                    "symbol": pos.symbol,
                    "direction": direction,
                    "volume": float(pos.volume),
                    "entry_price": float(pos.price_open),
                    "stop_loss": sl_val,
                    "take_profit": tp_val,
                    "floating_pnl": round(float(pos.profit) + float(pos.swap), 2),
                    "risk_amount_usd": risk_usd,
                    "open_time": open_dt,
                    "stage1_completed": bool(je and je.stage1_completed_at is not None),
                    "setup_type": je.setup_type if je else None,
                    "entry_thesis": je.entry_thesis if je else None,
                    "invalidation_plan": je.invalidation_plan if je else None,
                    "pre_trade_emotion": je.pre_trade_emotion if je else None,
                    "entry_chart_path": je.entry_chart_path if je else None,
                }
            )

        db.commit()
        return open_summaries

    @staticmethod
    def get_ungraded_closed_position_ids(db: Session, limit: int = 10) -> list[int]:
        rows = (
            db.query(Trade)
            .options(joinedload(Trade.journal_entry))
            .filter(Trade.status == "CLOSED")
            .order_by(Trade.close_time.desc())
            .all()
        )
        ungraded = []
        for t in rows:
            if not t.journal_entry or not t.journal_entry.process_verdict:
                ungraded.append(t.mt5_position_id)
                if len(ungraded) >= limit:
                    break
        return ungraded

    def sync_closed_trades(self, db: Session, lookback_days: int = 30) -> tuple[int, list[int]]:
        utc_to = datetime.now(timezone.utc) + timedelta(days=1)
        utc_from = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        deals = mt5.history_deals_get(utc_from, utc_to)
        if deals is None:
            logger.warning(f"No deals returned from MT5: {mt5.last_error()}")
            return 0, self.get_ungraded_closed_position_ids(db, limit=10)

        order_sl_tp_map = self._build_order_sl_tp_map(utc_from, utc_to)

        positions_map: dict[int, list] = {}
        for d in deals:
            if d.entry in (mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT) and d.position_id > 0:
                positions_map.setdefault(d.position_id, []).append(d)

        synced_count = 0
        needs_auto_verdict: list[int] = []

        for pos_id, pos_deals in positions_map.items():
            pos_deals.sort(key=lambda x: x.time)
            in_deals = [d for d in pos_deals if d.entry == mt5.DEAL_ENTRY_IN]
            out_deals = [d for d in pos_deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)]

            if not in_deals or not out_deals:
                continue

            first_in = in_deals[0]
            last_out = out_deals[-1]

            direction = "BUY" if first_in.type == mt5.DEAL_TYPE_BUY else "SELL"
            total_volume = sum(d.volume for d in in_deals)
            entry_price = sum(d.price * d.volume for d in in_deals) / total_volume
            exit_price = sum(d.price * d.volume for d in out_deals) / sum(d.volume for d in out_deals)

            commission = sum(d.commission for d in pos_deals)
            swap = sum(d.swap for d in pos_deals)
            gross_profit = sum(d.profit for d in pos_deals)
            net_profit = round(gross_profit + commission + swap, 2)

            open_time = datetime.fromtimestamp(first_in.time, tz=timezone.utc)
            close_time = datetime.fromtimestamp(last_out.time, tz=timezone.utc)

            existing = db.query(Trade).filter(Trade.mt5_position_id == pos_id).first()
            order_levels = order_sl_tp_map.get(pos_id, {})

            initial_sl = order_levels.get("initial_sl")
            final_sl = order_levels.get("final_sl")
            final_tp = order_levels.get("final_tp")

            stop_loss = final_sl or (existing.stop_loss if (existing and existing.stop_loss) else None)
            take_profit = final_tp or (existing.take_profit if (existing and existing.take_profit) else None)

            if not stop_loss and first_in.comment and "sl" in first_in.comment.lower():
                try:
                    parts = first_in.comment.lower().split("sl")
                    stop_loss = float(parts[1].strip().split()[0])
                except (IndexError, ValueError):
                    stop_loss = None

            risk_sl_reference = initial_sl or stop_loss
            computed_risk_usd = self._estimate_sl_risk_usd(first_in.symbol, entry_price, risk_sl_reference, total_volume)
            risk_usd = computed_risk_usd if computed_risk_usd is not None else (
                existing.risk_amount_usd if existing else None
            )
            r_multiple = round(net_profit / risk_usd, 2) if (risk_usd and risk_usd > 0) else None

            if existing:
                was_open = existing.status == "OPEN"
                sl_was_missing = not existing.stop_loss and bool(stop_loss)

                existing.exit_price = round(exit_price, 5)
                if stop_loss:
                    existing.stop_loss = stop_loss
                if take_profit:
                    existing.take_profit = take_profit
                existing.close_time = close_time
                existing.commission = round(commission, 2)
                existing.swap = round(swap, 2)
                existing.gross_profit = round(gross_profit, 2)
                existing.net_profit = net_profit
                if risk_usd is not None:
                    existing.risk_amount_usd = risk_usd
                existing.r_multiple = r_multiple
                existing.status = "CLOSED"

                je = existing.journal_entry
                if sl_was_missing and je and je.process_verdict:
                    je.process_verdict = None
                    je.outcome_quadrant = None
                    je.ai_feedback = None
                    je.mistake_tags = []
                    logger.info(f"Recovered SL ({stop_loss}) for MT5 #{pos_id}; queued for re-grading.")

                if was_open or not je or not je.process_verdict:
                    needs_auto_verdict.append(pos_id)
            else:
                trade_row = Trade(
                    mt5_position_id=pos_id,
                    symbol=first_in.symbol,
                    direction=direction,
                    volume=round(total_volume, 2),
                    entry_price=round(entry_price, 5),
                    exit_price=round(exit_price, 5),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    open_time=open_time,
                    close_time=close_time,
                    commission=round(commission, 2),
                    swap=round(swap, 2),
                    gross_profit=round(gross_profit, 2),
                    net_profit=net_profit,
                    risk_amount_usd=risk_usd,
                    r_multiple=r_multiple,
                    status="CLOSED",
                )
                db.add(trade_row)
                needs_auto_verdict.append(pos_id)

            synced_count += 1

        db.commit()
        logger.info(f"Synced {synced_count} closed positions from MT5 ({len(needs_auto_verdict)} queued for auto-grading).")
        return synced_count, needs_auto_verdict

    def update_drawdown_status(self, db: Session) -> DrawdownStatus:
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("MT5 account_info() unavailable.")

        starting_balance = float(self.rules["account"]["starting_balance"])
        daily_limit_pct = float(self.rules["drawdown_limits"]["daily_loss_pct"])
        daily_soft_pct = float(self.rules["drawdown_limits"]["daily_soft_lock_pct"])
        static_limit_pct = float(self.rules["drawdown_limits"]["max_static_drawdown_pct"])
        static_soft_pct = float(self.rules["drawdown_limits"]["max_soft_lock_pct"])

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot = db.query(DailyAccountSnapshot).filter(DailyAccountSnapshot.trading_date == today_str).first()

        if not snapshot:
            snapshot = DailyAccountSnapshot(
                trading_date=today_str,
                day_start_balance=float(acc.balance),
                day_start_equity=float(acc.equity),
                current_balance=float(acc.balance),
                current_equity=float(acc.equity),
                lowest_equity_today=float(acc.equity),
            )
            db.add(snapshot)
        else:
            snapshot.current_balance = float(acc.balance)
            snapshot.current_equity = float(acc.equity)
            if acc.equity < snapshot.lowest_equity_today:
                snapshot.lowest_equity_today = float(acc.equity)

        day_reference = max(snapshot.day_start_balance, snapshot.day_start_equity)
        daily_loss_limit_usd = round(day_reference * (daily_limit_pct / 100.0), 2)
        daily_floor_equity = day_reference - daily_loss_limit_usd
        daily_dd_remaining_usd = round(max(0.0, acc.equity - daily_floor_equity), 2)
        daily_dd_used_pct = round(max(0.0, ((day_reference - acc.equity) / day_reference) * 100.0), 2)

        static_floor_equity = round(starting_balance * (1.0 - static_limit_pct / 100.0), 2)
        static_dd_remaining_usd = round(max(0.0, acc.equity - static_floor_equity), 2)
        static_dd_used_pct = round(max(0.0, ((starting_balance - acc.equity) / starting_balance) * 100.0), 2)

        soft_lock = False
        lock_reason = None
        if daily_dd_used_pct >= daily_soft_pct:
            soft_lock = True
            lock_reason = f"Daily drawdown ({daily_dd_used_pct}%) hit soft-lock threshold ({daily_soft_pct}%)."
        elif static_dd_used_pct >= static_soft_pct:
            soft_lock = True
            lock_reason = f"Static drawdown ({static_dd_used_pct}%) hit soft-lock threshold ({static_soft_pct}%)."

        snapshot.daily_drawdown_used_pct = daily_dd_used_pct
        snapshot.static_drawdown_used_pct = static_dd_used_pct
        snapshot.is_locked_out = soft_lock
        db.commit()

        return DrawdownStatus(
            trading_date=today_str,
            starting_account_balance=starting_balance,
            day_reference_level=round(day_reference, 2),
            current_equity=round(float(acc.equity), 2),
            daily_loss_limit_usd=daily_loss_limit_usd,
            daily_drawdown_remaining_usd=daily_dd_remaining_usd,
            daily_drawdown_used_pct=daily_dd_used_pct,
            static_drawdown_floor_usd=static_floor_equity,
            static_drawdown_remaining_usd=static_dd_remaining_usd,
            static_drawdown_used_pct=static_dd_used_pct,
            soft_lock_triggered=soft_lock,
            lock_reason=lock_reason,
        )