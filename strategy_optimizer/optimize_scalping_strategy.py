#!/usr/bin/env python3
"""
Brute-force optimizer for a leveraged HYPE dip-buying scalping strategy.

This is a historical simulator, not trading advice. It models each long candle
as a low-first path: recurring buys fill on the way down, then the candle high
can close the fully averaged position at its take-profit target.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RESOURCES_DIR = PROJECT_ROOT / "resources"
CANDLE_INTERVAL_HOURS = 1
CSV_PATH = RESOURCES_DIR / "hype_1h_candles.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hype_candles import update_hype_csv  # noqa: E402


STARTING_CASH = 15_000.0
LEVERAGE = 5.0
HYPE_MAX_LEVERAGE = 10.0
HYPE_MAINTENANCE_MARGIN_RATE = 1.0 / (2.0 * HYPE_MAX_LEVERAGE)


@dataclass(frozen=True)
class StrategyParams:
    initial_drop_pct: float
    recurring_drop_pct: float
    first_buy_fraction: float
    recurring_buy_fraction: float
    take_profit_margin_pct: float


@dataclass
class Position:
    margin_used: float = 0.0
    notional: float = 0.0
    quantity: float = 0.0
    average_entry: float = 0.0
    first_entry: float = 0.0
    last_buy_price: float = 0.0
    buy_count: int = 0

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def add_buy(self, margin_amount: float, price: float, leverage: float) -> None:
        notional_amount = margin_amount * leverage
        quantity_added = notional_amount / price

        if not self.is_open:
            self.first_entry = price

        self.margin_used += margin_amount
        self.notional += notional_amount
        self.quantity += quantity_added
        self.average_entry = self.notional / self.quantity
        self.last_buy_price = price
        self.buy_count += 1

    def unrealized_pnl(self, mark_price: float) -> float:
        if not self.is_open:
            return 0.0
        return self.quantity * (mark_price - self.average_entry)

    def reduce(self, quantity_to_close: float, exit_price: float) -> tuple[float, float]:
        """Close part of a long position and return released margin and realized PnL."""
        if quantity_to_close <= 0 or quantity_to_close > self.quantity:
            raise ValueError("Quantity to close must be within the open position size")

        fraction_closed = quantity_to_close / self.quantity
        released_margin = self.margin_used * fraction_closed
        realized_pnl = quantity_to_close * (exit_price - self.average_entry)

        self.quantity -= quantity_to_close
        self.margin_used -= released_margin
        self.notional = self.quantity * self.average_entry

        if self.quantity <= 1e-12:
            self.__init__()

        return released_margin, realized_pnl


@dataclass
class BacktestResult:
    params: StrategyParams
    leverage: float
    final_equity: float
    profit: float
    profit_pct: float
    trades_closed: int
    liquidations: int
    buys: int
    wins: int
    losses: int
    max_drawdown_pct: float
    partial_reductions: int
    margin_trimmed: float
    reduced_target_exits: int


@dataclass(frozen=True)
class TradeEvent:
    candle_index: int
    timestamp: str
    event_type: str
    price: float
    quantity: float
    margin_amount: float
    detail: str


@dataclass
class BacktestRun:
    result: BacktestResult
    events: list[TradeEvent]


def percent_values(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def fraction_values(raw: str) -> list[float]:
    values = percent_values(raw)
    return [value / 100.0 if value > 1 else value for value in values]


def load_candles(update_csv: bool) -> pd.DataFrame:
    if update_csv:
        print("Updating hype_1h_candles.csv before optimization...")
        update_hype_csv()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find candle CSV at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required_columns = {"open_time", "open", "high", "low", "close"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").reset_index(drop=True)
    for column in ["open", "high", "low", "close"]:
        df[column] = df[column].astype(float)
    return df


def equity(cash: float, position: Position, mark_price: float) -> float:
    return cash + position.margin_used + position.unrealized_pnl(mark_price)


def close_quantity(cash: float, position: Position, quantity: float, exit_price: float) -> tuple[float, float]:
    released_margin, realized_pnl = position.reduce(quantity, exit_price)
    cash += released_margin + realized_pnl
    return max(cash, 0.0), realized_pnl


def close_position(cash: float, position: Position, exit_price: float) -> tuple[float, bool]:
    cash, pnl = close_quantity(cash, position, position.quantity, exit_price)
    was_win = pnl > 0
    return cash, was_win


def maintenance_margin_rate(leverage: float, initial_liquidation_drop_pct: float) -> float:
    """Infer a maintenance rate that gives the requested initial long liq distance."""
    if leverage <= 1:
        raise ValueError("Leverage must be greater than 1")
    if not 0 < initial_liquidation_drop_pct < 100:
        raise ValueError("Liquidation drop percent must be between 0 and 100")

    requested_multiplier = 1.0 - initial_liquidation_drop_pct / 100.0
    rate = 1.0 - (1.0 - 1.0 / leverage) / requested_multiplier
    if not 0 <= rate < 1:
        raise ValueError("Liquidation drop percent is incompatible with the selected leverage")
    return rate


def hype_maintenance_margin_rate() -> float:
    """HYPE's 5% maintenance rate in its <=$20M notional margin tier."""
    return HYPE_MAINTENANCE_MARGIN_RATE


def liquidation_price(position: Position, maintenance_rate: float) -> float:
    """Compute isolated-long liquidation from the live average entry and margin."""
    if not position.is_open:
        return 0.0
    return (position.quantity * position.average_entry - position.margin_used) / (
        position.quantity * (1.0 - maintenance_rate)
    )


def enforce_margin_cap(
    cash: float,
    position: Position,
    exit_price: float,
    cycle_start_equity: float,
    max_cycle_margin_pct: float,
) -> tuple[float, bool, float]:
    """Trim immediately so allocated margin stays below the cycle's risk budget."""
    max_margin = cycle_start_equity * max_cycle_margin_pct / 100.0
    if not position.is_open or position.margin_used <= max_margin:
        return cash, False, 0.0

    quantity_to_close = position.quantity * (1.0 - max_margin / position.margin_used)
    trimmed_margin = position.margin_used - max_margin
    cash, _ = close_quantity(cash, position, quantity_to_close, exit_price)
    return cash, True, trimmed_margin


def simulate_strategy(
    candles: pd.DataFrame,
    params: StrategyParams,
    *,
    starting_cash: float,
    leverage: float,
    max_cycle_margin_pct: float,
    reduced_target_margin_threshold_pct: float,
    reduced_target_multiplier: float,
    dip_lookback_hours: int,
    max_additional_buys: int,
    capture_events: bool = False,
) -> BacktestRun:
    cash = starting_cash
    position = Position()
    trades_closed = 0
    liquidations = 0
    buys = 0
    wins = 0
    losses = 0
    peak_equity = starting_cash
    max_drawdown_pct = 0.0
    partial_reductions = 0
    margin_trimmed = 0.0
    reduced_target_exits = 0
    cycle_start_equity = 0.0
    maintenance_rate = hype_maintenance_margin_rate()
    lookback_candles = max(1, -(-dip_lookback_hours // CANDLE_INTERVAL_HOURS))
    events: list[TradeEvent] = []

    def record_event(
        index: int,
        row: pd.Series,
        event_type: str,
        price: float,
        quantity: float,
        margin_amount: float,
        detail: str,
    ) -> None:
        if not capture_events:
            return
        timestamp = str(row["open_time_iso"]) if "open_time_iso" in row.index else str(row["open_time"])
        events.append(
            TradeEvent(
                candle_index=index,
                timestamp=timestamp,
                event_type=event_type,
                price=price,
                quantity=quantity,
                margin_amount=margin_amount,
                detail=detail,
            )
        )

    highs = candles["high"].tolist()
    lows = candles["low"].tolist()
    closes = candles["close"].tolist()

    for index, row in candles.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        liquidated_this_candle = False
        started_candle_flat = not position.is_open

        if started_candle_flat and cash > 0:
            lookback_start = max(0, index - lookback_candles)
            if lookback_start < index:
                recent_high = max(highs[lookback_start:index])
                initial_buy_price = recent_high * (1.0 - params.initial_drop_pct / 100.0)
                cycle_start_equity = equity(cash, position, close)
                margin_amount = cycle_start_equity * params.first_buy_fraction

                if low <= initial_buy_price and cash >= margin_amount and margin_amount > 0:
                    cash -= margin_amount
                    position.add_buy(margin_amount, initial_buy_price, leverage)
                    record_event(
                        index,
                        row,
                        "initial_buy",
                        initial_buy_price,
                        position.quantity,
                        margin_amount,
                        "Initial dip purchase triggered by candle low.",
                    )
                    buys += 1
                    quantity_before_trim = position.quantity
                    cash, was_trimmed, trimmed_amount = enforce_margin_cap(
                        cash,
                        position,
                        initial_buy_price,
                        cycle_start_equity,
                        max_cycle_margin_pct,
                    )
                    if was_trimmed:
                        record_event(
                            index,
                            row,
                            "margin_trim",
                            initial_buy_price,
                            quantity_before_trim - position.quantity,
                            trimmed_amount,
                            "Position reduced to the cycle margin cap.",
                        )
                    partial_reductions += int(was_trimmed)
                    margin_trimmed += trimmed_amount

        # Model the descending portion of the candle. Each crossed average-down
        # level fills before a lower liquidation price can be reached.
        while position.is_open and position.buy_count - 1 < max_additional_buys:
            next_buy_price = position.last_buy_price * (1.0 - params.recurring_drop_pct / 100.0)
            current_liquidation_price = liquidation_price(position, maintenance_rate)

            # The position liquidates before the next buy only when its current
            # liquidation price lies above that next buy level and is reached.
            if current_liquidation_price >= next_buy_price:
                if low <= current_liquidation_price:
                    record_event(
                        index,
                        row,
                        "liquidation",
                        current_liquidation_price,
                        position.quantity,
                        position.margin_used,
                        "Liquidation was reached before the next average-down level.",
                    )
                    position.__init__()
                    trades_closed += 1
                    liquidations += 1
                    losses += 1
                    liquidated_this_candle = True
                break

            margin_amount = cycle_start_equity * params.recurring_buy_fraction
            if low > next_buy_price or cash < margin_amount or margin_amount <= 0:
                break

            cash -= margin_amount
            quantity_before_buy = position.quantity
            position.add_buy(margin_amount, next_buy_price, leverage)
            record_event(
                index,
                row,
                "recurring_buy",
                next_buy_price,
                position.quantity - quantity_before_buy,
                margin_amount,
                "Average-down purchase triggered by the descending candle path.",
            )
            buys += 1
            quantity_before_trim = position.quantity
            cash, was_trimmed, trimmed_amount = enforce_margin_cap(
                cash,
                position,
                next_buy_price,
                cycle_start_equity,
                max_cycle_margin_pct,
            )
            if was_trimmed:
                record_event(
                    index,
                    row,
                    "margin_trim",
                    next_buy_price,
                    quantity_before_trim - position.quantity,
                    trimmed_amount,
                    "Position reduced to the cycle margin cap.",
                )
            partial_reductions += int(was_trimmed)
            margin_trimmed += trimmed_amount

        if position.is_open:
            current_liquidation_price = liquidation_price(position, maintenance_rate)
            if low <= current_liquidation_price:
                record_event(
                    index,
                    row,
                    "liquidation",
                    current_liquidation_price,
                    position.quantity,
                    position.margin_used,
                    "Candle low reached liquidation after all eligible average-down buys.",
                )
                position.__init__()
                trades_closed += 1
                liquidations += 1
                losses += 1
                liquidated_this_candle = True

        # After the modeled downward path, the candle can recover to its high.
        if position.is_open and not liquidated_this_candle:
            current_portfolio_equity = equity(cash, position, low)
            current_margin_pct = (
                position.margin_used / current_portfolio_equity * 100.0
                if current_portfolio_equity > 0
                else float("inf")
            )
            used_reduced_target = current_margin_pct >= reduced_target_margin_threshold_pct
            effective_take_profit_margin_pct = params.take_profit_margin_pct
            if used_reduced_target:
                effective_take_profit_margin_pct *= reduced_target_multiplier

            price_return_target = effective_take_profit_margin_pct / (100.0 * leverage)
            target_price = position.average_entry * (1.0 + price_return_target)
            if high >= target_price:
                quantity_closed = position.quantity
                released_margin = position.margin_used
                cash, was_win = close_position(cash, position, target_price)
                record_event(
                    index,
                    row,
                    "take_profit",
                    target_price,
                    quantity_closed,
                    released_margin,
                    (
                        f"Reduced {effective_take_profit_margin_pct:g}% margin target reached; "
                        f"allocated margin was {current_margin_pct:.2f}% of current portfolio equity."
                        if used_reduced_target
                        else "Take-profit target reached after the modeled downward path."
                    ),
                )
                trades_closed += 1
                wins += int(was_win)
                losses += int(not was_win)
                reduced_target_exits += int(used_reduced_target)

        current_equity = equity(cash, position, close)
        peak_equity = max(peak_equity, current_equity)
        if peak_equity > 0:
            drawdown_pct = (peak_equity - current_equity) / peak_equity * 100.0
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        if cash <= 0 and not position.is_open:
            break

    final_equity = equity(cash, position, closes[-1])
    profit = final_equity - starting_cash
    result = BacktestResult(
        params=params,
        leverage=leverage,
        final_equity=final_equity,
        profit=profit,
        profit_pct=profit / starting_cash * 100.0,
        trades_closed=trades_closed,
        liquidations=liquidations,
        buys=buys,
        wins=wins,
        losses=losses,
        max_drawdown_pct=max_drawdown_pct,
        partial_reductions=partial_reductions,
        margin_trimmed=margin_trimmed,
        reduced_target_exits=reduced_target_exits,
    )
    return BacktestRun(result=result, events=events)


def backtest(
    candles: pd.DataFrame,
    params: StrategyParams,
    *,
    starting_cash: float,
    leverage: float,
    max_cycle_margin_pct: float,
    reduced_target_margin_threshold_pct: float,
    reduced_target_multiplier: float,
    dip_lookback_hours: int,
    max_additional_buys: int,
) -> BacktestResult:
    return simulate_strategy(
        candles,
        params,
        starting_cash=starting_cash,
        leverage=leverage,
        max_cycle_margin_pct=max_cycle_margin_pct,
        reduced_target_margin_threshold_pct=reduced_target_margin_threshold_pct,
        reduced_target_multiplier=reduced_target_multiplier,
        dip_lookback_hours=dip_lookback_hours,
        max_additional_buys=max_additional_buys,
    ).result


def parameter_grid(args: argparse.Namespace) -> Iterable[tuple[float, StrategyParams]]:
    return (
        (
            leverage,
            StrategyParams(
                initial_drop_pct=initial_drop,
                recurring_drop_pct=recurring_drop,
                first_buy_fraction=first_fraction,
                recurring_buy_fraction=recurring_fraction,
                take_profit_margin_pct=take_profit,
            ),
        )
        for leverage, initial_drop, recurring_drop, first_fraction, recurring_fraction, take_profit in itertools.product(
            leverage_values(args),
            percent_values(args.initial_drop_pcts),
            percent_values(args.recurring_drop_pcts),
            fraction_values(args.first_buy_fractions),
            fraction_values(args.recurring_buy_fractions),
            percent_values(args.take_profit_margin_pcts),
        )
    )


def format_result(result: BacktestResult) -> str:
    params = result.params
    return (
        f"leverage={result.leverage:.1f}x | "
        f"initial_drop={params.initial_drop_pct:.2f}% | "
        f"recurring_drop={params.recurring_drop_pct:.2f}% | "
        f"first_buy={params.first_buy_fraction * 100:.2f}% | "
        f"recurring_buy={params.recurring_buy_fraction * 100:.2f}% | "
        f"take_profit_margin={params.take_profit_margin_pct:.2f}% | "
        f"profit=${result.profit:,.2f} ({result.profit_pct:.2f}%) | "
        f"final_equity=${result.final_equity:,.2f} | "
        f"buys={result.buys} | closed={result.trades_closed} | "
        f"wins={result.wins} | losses={result.losses} | "
        f"liquidations={result.liquidations} | max_dd={result.max_drawdown_pct:.2f}%"
        f" | partial_reductions={result.partial_reductions} | margin_trimmed=${result.margin_trimmed:,.2f}"
        f" | reduced_target_exits={result.reduced_target_exits}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a leveraged dip-buying HYPE scalping strategy over 1-hour candles."
    )
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--leverages", default="5,6", help="Leverage values to test (default: 5,6).")
    parser.add_argument("--leverage", type=float, help="Test one leverage value instead of the --leverages grid.")
    parser.add_argument(
        "--max-cycle-margin-pct",
        type=float,
        default=12.0,
        help="Maximum margin allocated to one cycle, as a percent of account equity before that cycle starts.",
    )
    parser.add_argument(
        "--reduced-target-margin-threshold-pct",
        type=float,
        default=12.0,
        help="Use the reduced take-profit target when allocated margin reaches this percent of current portfolio equity.",
    )
    parser.add_argument(
        "--reduced-target-multiplier",
        type=float,
        default=0.5,
        help="Multiplier for the take-profit margin target after the reduced-target threshold is reached.",
    )
    parser.add_argument(
        "--dip-lookback-hours",
        type=int,
        default=168,
        help="Hours used to find the prior high for an initial dip (168 one-hour candles by default).",
    )
    parser.add_argument(
        "--max-additional-buys",
        type=int,
        default=15,
        help="Maximum average-down buys after the initial purchase (15 by default).",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-update", action="store_true", help="Skip the 1-hour CSV update and use the current CSV.")
    parser.add_argument("--initial-drop-pcts", default="2,3,4,5,6,8,10")
    parser.add_argument("--recurring-drop-pcts", default="2,3,4,5,6,8")
    parser.add_argument("--first-buy-fractions", default="2.5,5,7.5,10")
    parser.add_argument("--recurring-buy-fractions", default="2.5,5,7.5")
    parser.add_argument("--take-profit-margin-pcts", default="10,15,20,25,30,40")
    return parser.parse_args()


def leverage_values(args: argparse.Namespace) -> list[float]:
    values = [args.leverage] if args.leverage is not None else percent_values(args.leverages)
    if not values or any(value <= 1 for value in values):
        raise ValueError("Leverage values must be greater than 1.")
    return values


def main() -> None:
    args = parse_args()
    candles = load_candles(update_csv=not args.no_update)
    print(f"Loaded {len(candles):,} candles from {CSV_PATH}")
    print(
        f"Starting cash=${args.starting_cash:,.2f}, leverage values={', '.join(f'{value:g}x' for value in leverage_values(args))}, "
        f"HYPE maintenance margin rate={HYPE_MAINTENANCE_MARGIN_RATE:.2%} (10x max-leverage tier), "
        f"with a {args.max_cycle_margin_pct:.2f}% cycle margin cap and "
        f"a {args.reduced_target_multiplier:.2f}x take-profit target at "
        f"{args.reduced_target_margin_threshold_pct:.2f}% current margin exposure."
    )

    results: list[BacktestResult] = []
    params_to_try = list(parameter_grid(args))
    print(f"Trying {len(params_to_try):,} parameter combinations...\n")

    for run_number, (leverage, params) in enumerate(params_to_try, start=1):
        print(
            f"[{run_number:,}/{len(params_to_try):,}] Trying "
            f"leverage={leverage:.1f}x, "
            f"initial_drop={params.initial_drop_pct:.2f}%, "
            f"recurring_drop={params.recurring_drop_pct:.2f}%, "
            f"first_buy={params.first_buy_fraction * 100:.2f}%, "
            f"recurring_buy={params.recurring_buy_fraction * 100:.2f}%, "
            f"take_profit_margin={params.take_profit_margin_pct:.2f}%"
        )
        result = backtest(
            candles,
            params,
            starting_cash=args.starting_cash,
            leverage=leverage,
            max_cycle_margin_pct=args.max_cycle_margin_pct,
            reduced_target_margin_threshold_pct=args.reduced_target_margin_threshold_pct,
            reduced_target_multiplier=args.reduced_target_multiplier,
            dip_lookback_hours=args.dip_lookback_hours,
            max_additional_buys=args.max_additional_buys,
        )
        print(f"    Total profit: ${result.profit:,.2f} ({result.profit_pct:.2f}%)")
        results.append(result)

    results.sort(key=lambda item: item.profit, reverse=True)

    print("\nBest settings:")
    for rank, result in enumerate(results[: args.top], start=1):
        print(f"{rank:>2}. {format_result(result)}")


if __name__ == "__main__":
    main()
