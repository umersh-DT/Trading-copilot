import os
import re
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import MetaTrader5 as mt5
from sqlalchemy.orm import Session

from core.database import SessionLocal, init_db, load_risk_rules
from core.models import Trade, DailyAccountSnapshot
from core.schemas import DrawdownStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mt5_sync")


class MT5SyncConnector:
    def __init__(self):
        self.rules = load_risk_rules()
        self.login = os.getenv("MT5_LOGIN")
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        self.mt5_path = os.getenv("MT5_PATH")

    def connect(self) -> bool:
        """Initialize connection to the running MetaTrader 5 terminal."""
        init_kwargs = {}
        if self.mt5_path and os.path.exists(self.mt5_path):
            init_kwargs["path"] = self.mt5_path
        if self.login and self.password and self.server:
            init_kwargs["login"] = int(self.login)
            init_kwargs["password"] = self.password
            init_kwargs["server"] = self.server

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
            return False

        account = mt5.account_info()
        if account is None:
            logger.error(f"Failed to retrieve MT5 account_info(): {mt5.last_error()}")
            mt5.shutdown()
            return False

        logger.info(f"Connected to MT5 | Account: {account.login} | Server: {account.server} | Equity: {account.equity:.2f}")
        return True

    def disconnect(self) -> None:
        mt5.shutdown()

    @staticmethod
    def _extract_sl_from_comment(comment: str) -> Optional[float]:
        """Parse MT5 auto-generated stop-loss tags like '[sl 2645.50]' from closing deals."""
        if not comment:
            return None
        match = re.search(r"\[sl\s+([0-9.]+)\]", comment, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _calculate_r_multiple(
        self, symbol: str, direction: str, volume: float, entry_price: float, stop_loss: Optional[float], net_profit: float
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute initial USD risk and realized R-multiple using MT5 symbol contract specs."""
        if not stop_loss or stop_loss <= 0 or entry_price == stop_loss:
            return None, None

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None or sym_info.trade_tick_size <= 0 or sym_info.trade_tick_value <= 0:
            return None, None

        price_risk = (entry_price - stop_loss) if direction == "BUY" else (stop_loss - entry_price)
        if price_risk <= 0:
            return None, None

        ticks_at_risk = price_risk / sym_info.trade_tick_size
        risk_usd = round(ticks_at_risk * sym_info.trade_tick_value * volume, 2)
        if risk_usd <= 0:
            return None, None

        r_multiple = round(net_profit / risk_usd, 2)
        return risk_usd, r_multiple

    def sync_closed_trades(self, db: Session, lookback_days: int = 30) -> int:
        """Fetch historical deals from MT5, reconstruct closed positions, and upsert into SQLite."""
        date_to = datetime.now(timezone.utc) + timedelta(days=1)
        date_from = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            logger.warning(f"No deals returned or error: {mt5.last_error()}")
            return 0

        # Group deals by position_id (ignore balance/credit operations where position_id == 0)
        positions_map = defaultdict(list)
        for d in deals:
            if d.position_id > 0 and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                positions_map[d.position_id].append(d)

        synced_count = 0
        for position_id, pos_deals in positions_map.items():
            pos_deals.sort(key=lambda x: x.time_msc)

            in_deals = [d for d in pos_deals if d.entry == mt5.DEAL_ENTRY_IN]
            out_deals = [d for d in pos_deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]

            if not in_deals or not out_deals:
                continue  # Position is still open or incomplete

            total_in_vol = sum(d.volume for d in in_deals)
            total_out_vol = sum(d.volume for d in out_deals)
            if round(total_out_vol, 4) < round(total_in_vol, 4):
                continue  # Partially closed; wait for full exit

            first_in = in_deals[0]
            last_out = out_deals[-1]
            symbol = first_in.symbol
            direction = "BUY" if first_in.type == mt5.DEAL_TYPE_BUY else "SELL"

            # Volume-weighted average entry & exit prices
            entry_price = sum(d.price * d.volume for d in in_deals) / total_in_vol
            exit_price = sum(d.price * d.volume for d in out_deals) / total_out_vol

            commission = sum(d.commission for d in pos_deals)
            swap = sum(d.swap for d in pos_deals)
            gross_profit = sum(d.profit for d in out_deals)
            net_profit = round(gross_profit + commission + swap, 2)

            stop_loss = self._extract_sl_from_comment(last_out.comment)
            risk_usd, r_multiple = self._calculate_r_multiple(
                symbol=symbol,
                direction=direction,
                volume=total_in_vol,
                entry_price=entry_price,
                stop_loss=stop_loss,
                net_profit=net_profit,
            )

            open_time = datetime.fromtimestamp(first_in.time, tz=timezone.utc)
            close_time = datetime.fromtimestamp(last_out.time, tz=timezone.utc)

            existing = db.query(Trade).filter(Trade.mt5_position_id == position_id).first()
            if existing:
                if existing.status != "CLOSED":
                    existing.exit_price = round(exit_price, 5)
                    existing.close_time = close_time
                    existing.commission = round(commission, 2)
                    existing.swap = round(swap, 2)
                    existing.gross_profit = round(gross_profit, 2)
                    existing.net_profit = net_profit
                    existing.stop_loss = stop_loss or existing.stop_loss
                    existing.risk_amount_usd = risk_usd
                    existing.r_multiple = r_multiple
                    existing.status = "CLOSED"
                    synced_count += 1
            else:
                trade = Trade(
                    mt5_position_id=position_id,
                    symbol=symbol,
                    direction=direction,
                    volume=round(total_in_vol, 2),
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
                db.add(trade)
                synced_count += 1

        db.commit()
        logger.info(f"Synced {synced_count} new/updated closed trades into SQLite.")
        return synced_count

    def update_drawdown_status(self, db: Session) -> DrawdownStatus:
        """
        Compute live The5ers High Stakes drawdown buffers:
        - Daily 5% limit: anchored at 00:00 server time on max(day_start_balance, day_start_equity)
        - Static 10% limit: anchored permanently at starting_balance * (1 - 0.10)
        """
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 account_info() unavailable.")

        # Use MT5 server tick time to align with 00:00 broker server midnight reset
        tick = mt5.symbol_info_tick("EURUSD") or mt5.symbol_info_tick("XAUUSD")
        server_now = datetime.fromtimestamp(tick.time, tz=timezone.utc) if tick else datetime.now(timezone.utc)
        trading_date = server_now.strftime("%Y-%m-%d")

        starting_balance = float(self.rules["account"]["starting_balance"])
        daily_limit_pct = float(self.rules["drawdown_limits"]["daily_loss_pct"])
        daily_soft_pct = float(self.rules["drawdown_limits"]["daily_soft_lock_pct"])
        static_limit_pct = float(self.rules["drawdown_limits"]["max_static_drawdown_pct"])
        static_soft_pct = float(self.rules["drawdown_limits"]["max_soft_lock_pct"])

        snapshot = db.query(DailyAccountSnapshot).filter(
            DailyAccountSnapshot.trading_date == trading_date
        ).first()

        if not snapshot:
            snapshot = DailyAccountSnapshot(
                trading_date=trading_date,
                day_start_balance=account.balance,
                day_start_equity=account.equity,
                current_balance=account.balance,
                current_equity=account.equity,
                lowest_equity_today=account.equity,
            )
            db.add(snapshot)
        else:
            snapshot.current_balance = account.balance
            snapshot.current_equity = account.equity
            snapshot.lowest_equity_today = min(snapshot.lowest_equity_today, account.equity)

        # The5ers High Stakes Daily Reference Level = higher of day-start balance or equity
        day_reference = max(snapshot.day_start_balance, snapshot.day_start_equity)
        daily_loss_limit_usd = round(day_reference * (daily_limit_pct / 100.0), 2)
        daily_floor_usd = day_reference - daily_loss_limit_usd

        daily_drawdown_usd = max(0.0, day_reference - account.equity)
        daily_used_pct = round((daily_drawdown_usd / day_reference) * 100.0, 2) if day_reference > 0 else 0.0
        daily_remaining_usd = round(max(0.0, account.equity - daily_floor_usd), 2)

        # Static Max Drawdown = 10% below initial account starting balance
        static_max_loss_usd = starting_balance * (static_limit_pct / 100.0)
        static_floor_usd = round(starting_balance - static_max_loss_usd, 2)
        static_drawdown_usd = max(0.0, starting_balance - account.equity)
        static_used_pct = round((static_drawdown_usd / starting_balance) * 100.0, 2) if starting_balance > 0 else 0.0
        static_remaining_usd = round(max(0.0, account.equity - static_floor_usd), 2)

        # Evaluate soft-lock circuit breakers
        soft_lock = False
        lock_reason = None
        if daily_used_pct >= daily_soft_pct:
            soft_lock = True
            lock_reason = f"Daily drawdown ({daily_used_pct}%) reached soft-lock buffer ({daily_soft_pct}%)."
        elif static_used_pct >= static_soft_pct:
            soft_lock = True
            lock_reason = f"Static drawdown ({static_used_pct}%) reached soft-lock buffer ({static_soft_pct}%)."

        snapshot.daily_drawdown_used_pct = daily_used_pct
        snapshot.static_drawdown_used_pct = static_used_pct
        snapshot.is_locked_out = soft_lock
        db.commit()

        return DrawdownStatus(
            trading_date=trading_date,
            starting_account_balance=starting_balance,
            day_reference_level=round(day_reference, 2),
            current_equity=round(account.equity, 2),
            daily_loss_limit_usd=daily_loss_limit_usd,
            daily_drawdown_remaining_usd=daily_remaining_usd,
            daily_drawdown_used_pct=daily_used_pct,
            static_drawdown_floor_usd=static_floor_usd,
            static_drawdown_remaining_usd=static_remaining_usd,
            static_drawdown_used_pct=static_used_pct,
            soft_lock_triggered=soft_lock,
            lock_reason=lock_reason,
        )


def run_sync_once() -> None:
    """Standalone CLI entry point to test MT5 sync and verify database writes."""
    init_db()
    connector = MT5SyncConnector()
    if not connector.connect():
        return

    db = SessionLocal()
    try:
        synced = connector.sync_closed_trades(db, lookback_days=60)
        status = connector.update_drawdown_status(db)
        print("\n=== The5ers High Stakes Live Telemetry ===")
        print(f"Trades Synced       : {synced}")
        print(f"Current Equity      : ${status.current_equity:,.2f}")
        print(f"Daily Buffer Left   : ${status.daily_drawdown_remaining_usd:,.2f} ({status.daily_drawdown_used_pct}% used of 5.0%)")
        print(f"Static Buffer Left  : ${status.static_drawdown_remaining_usd:,.2f} ({status.static_drawdown_used_pct}% used of 10.0%)")
        print(f"Circuit Breaker     : {'LOCKED - ' + status.lock_reason if status.soft_lock_triggered else 'ACTIVE (Clear to Trade)'}")
    finally:
        db.close()
        connector.disconnect()


if __name__ == "__main__":
    run_sync_once()