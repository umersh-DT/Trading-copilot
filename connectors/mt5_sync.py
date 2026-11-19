import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import MetaTrader5 as mt5
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import Trade, DailyAccountSnapshot
from core.schemas import DrawdownStatus

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mt5_sync")


class MT5SyncConnector:
    """Connects to MetaTrader 5 terminal, syncs live OPEN + CLOSED positions, and tracks The5ers drawdown."""

    def __init__(self):
        self.login = int(os.getenv("MT5_LOGIN", "0"))
        self.password = os.getenv("MT5_PASSWORD", "")
        self.server = os.getenv("MT5_SERVER", "")
        self.terminal_path = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
        self.rules = load_risk_rules()

    def connect(self) -> bool:
        """Initialize connection to the local MT5 terminal."""
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

        logger.info(f"Connected to MT5 | Account: {acc.login} | Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f}")
        return True

    def disconnect(self):
        mt5.shutdown()

    def _estimate_sl_risk_usd(self, symbol: str, entry_price: float, stop_loss: Optional[float], volume: float) -> Optional[float]:
        """Calculate exact USD dollar risk from entry_price to stop_loss using MT5 symbol tick specs."""
        if not stop_loss or stop_loss <= 0:
            return None

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None or sym_info.trade_tick_size == 0:
            return None

        price_diff = abs(entry_price - stop_loss)
        ticks = price_diff / sym_info.trade_tick_size
        risk_usd = ticks * sym_info.trade_tick_value * volume
        return round(risk_usd, 2)

    def sync_open_positions(self, db: Session) -> list[dict]:
        """
        Poll mt5.positions_get() for currently active positions.
        - Upserts live OPEN trades in SQLite so the Stage 1 Entry Questionnaire pops up immediately.
        - Returns all currently open positions with live floating PnL and SL risk.
        """
        positions = mt5.positions_get()
        if positions is None:
            logger.warning(f"mt5.positions_get() returned None: {mt5.last_error()}")
            return []

        active_tickets = set()
        open_summaries = []

        for pos in positions:
            ticket = int(pos.ticket)
            active_tickets.add(ticket)
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

    def sync_closed_trades(self, db: Session, lookback_days: int = 30) -> tuple[int, list[int]]:
        """
        Fetch closed deals from MT5 history, reconstruct round-trip positions, and upsert into SQLite.
        Returns (total_closed_upserted, newly_closed_position_ids_with_stage1_journal).
        """
        utc_to = datetime.now(timezone.utc) + timedelta(days=1)
        utc_from = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        deals = mt5.history_deals_get(utc_from, utc_to)
        if deals is None:
            logger.warning(f"No deals returned from MT5: {mt5.last_error()}")
            return 0, []

        positions_map: dict[int, list] = {}
        for d in deals:
            if d.entry in (mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT) and d.position_id > 0:
                positions_map.setdefault(d.position_id, []).append(d)

        synced_count = 0
        newly_closed_for_verdict: list[int] = []

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

            # Check if SL was already captured while OPEN, otherwise parse from comment
            existing = db.query(Trade).filter(Trade.mt5_position_id == pos_id).first()
            stop_loss = existing.stop_loss if (existing and existing.stop_loss) else None
            if not stop_loss and first_in.comment and "sl" in first_in.comment.lower():
                try:
                    parts = first_in.comment.lower().split("sl")
                    stop_loss = float(parts[1].strip().split()[0])
                except (IndexError, ValueError):
                    stop_loss = None

            risk_usd = (
                existing.risk_amount_usd
                if (existing and existing.risk_amount_usd)
                else self._estimate_sl_risk_usd(first_in.symbol, entry_price, stop_loss, total_volume)
            )
            r_multiple = round(net_profit / risk_usd, 2) if (risk_usd and risk_usd > 0) else None

            if existing:
                was_open = existing.status == "OPEN"
                existing.exit_price = round(exit_price, 5)
                existing.close_time = close_time
                existing.commission = round(commission, 2)
                existing.swap = round(swap, 2)
                existing.gross_profit = round(gross_profit, 2)
                existing.net_profit = net_profit
                existing.r_multiple = r_multiple
                existing.status = "CLOSED"

                # Flag if this trade just closed and needs its Stage 2 Good/Bad Trade verdict
                je = existing.journal_entry
                if (was_open or (je and je.stage1_completed_at and not je.process_verdict)):
                    newly_closed_for_verdict.append(pos_id)
            else:
                trade_row = Trade(
                    mt5_position_id=pos_id,
                    symbol=first_in.symbol,
                    direction=direction,
                    volume=round(total_volume, 2),
                    entry_price=round(entry_price, 5),
                    exit_price=round(exit_price, 5),
                    stop_loss=stop_loss,
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

            synced_count += 1

        db.commit()
        logger.info(f"Synced {synced_count} closed positions from MT5 into SQLite.")
        return synced_count, newly_closed_for_verdict

    def update_drawdown_status(self, db: Session) -> DrawdownStatus:
        """Calculate real-time The5ers High Stakes Daily (5%) and Static (10%) drawdown buffers."""
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


if __name__ == "__main__":
    init_db()
    connector = MT5SyncConnector()
    if connector.connect():
        db = SessionLocal()
        try:
            open_pos = connector.sync_open_positions(db)
            closed_count, newly_closed = connector.sync_closed_trades(db, lookback_days=30)
            dd = connector.update_drawdown_status(db)
            print(f"Open Positions: {len(open_pos)} | Closed Synced: {closed_count} | Newly Closed: {newly_closed}")
            print(f"Daily DD Used: {dd.daily_drawdown_used_pct}% (${dd.daily_drawdown_remaining_usd} left)")
        finally:
            db.close()
            connector.disconnect()