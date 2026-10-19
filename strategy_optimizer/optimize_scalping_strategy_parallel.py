#!/usr/bin/env python3
"""Multi-core version of the HYPE scalping-strategy optimizer.

The original optimizer remains unchanged. This runner evaluates independent
parameter sets in separate processes so CPU-bound simulations can use multiple
cores instead of being constrained by Python's thread lock.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd

from optimize_scalping_strategy import (
    CSV_PATH,
    LEVERAGE,
    STARTING_CASH,
    BacktestResult,
    StrategyParams,
    backtest,
    format_result,
    load_candles,
    parameter_grid,
)


@dataclass(frozen=True)
class WorkerSettings:
    starting_cash: float
    leverage: float
    liquidation_drop_pct: float
    max_cycle_margin_pct: float
    reduced_target_margin_threshold_pct: float
    reduced_target_multiplier: float
    dip_lookback_hours: int
    max_additional_buys: int


_CANDLES: pd.DataFrame | None = None
_SETTINGS: WorkerSettings | None = None


def initialize_worker(candles: pd.DataFrame, settings: WorkerSettings) -> None:
    """Load the candles once for each worker process, rather than each task."""
    global _CANDLES, _SETTINGS
    _CANDLES = candles
    _SETTINGS = settings


def evaluate_parameter_set(params: StrategyParams) -> BacktestResult:
    if _CANDLES is None or _SETTINGS is None:
        raise RuntimeError("Parallel optimizer worker was not initialized.")
    return backtest(
        _CANDLES,
        params,
        starting_cash=_SETTINGS.starting_cash,
        leverage=_SETTINGS.leverage,
        liquidation_drop_pct=_SETTINGS.liquidation_drop_pct,
        max_cycle_margin_pct=_SETTINGS.max_cycle_margin_pct,
        reduced_target_margin_threshold_pct=_SETTINGS.reduced_target_margin_threshold_pct,
        reduced_target_multiplier=_SETTINGS.reduced_target_multiplier,
        dip_lookback_hours=_SETTINGS.dip_lookback_hours,
        max_additional_buys=_SETTINGS.max_additional_buys,
    )


def parse_args() -> argparse.Namespace:
    default_workers = min(os.cpu_count() or 1, 8)
    parser = argparse.ArgumentParser(
        description="Optimize the HYPE scalping strategy across multiple CPU processes."
    )
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--leverage", type=float, default=LEVERAGE)
    parser.add_argument("--liquidation-drop-pct", type=float, default=15.0)
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
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Worker processes to use (default: {default_workers}).")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--no-update", action="store_true", help="Skip the 1-hour CSV update and use the current CSV.")
    parser.add_argument("--initial-drop-pcts", default="2,3,4,5,6,8,10")
    parser.add_argument("--recurring-drop-pcts", default="2,3,4,5,6,8")
    parser.add_argument("--first-buy-fractions", default="2.5,5,7.5,10")
    parser.add_argument("--recurring-buy-fractions", default="2.5,5,7.5")
    parser.add_argument("--take-profit-margin-pcts", default="10,15,20,25,30,40")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> WorkerSettings:
    return WorkerSettings(
        starting_cash=args.starting_cash,
        leverage=args.leverage,
        liquidation_drop_pct=args.liquidation_drop_pct,
        max_cycle_margin_pct=args.max_cycle_margin_pct,
        reduced_target_margin_threshold_pct=args.reduced_target_margin_threshold_pct,
        reduced_target_multiplier=args.reduced_target_multiplier,
        dip_lookback_hours=args.dip_lookback_hours,
        max_additional_buys=args.max_additional_buys,
    )


def print_queued(run_number: int, total: int, params: StrategyParams) -> None:
    print(
        f"[{run_number:,}/{total:,}] Queued "
        f"initial_drop={params.initial_drop_pct:.2f}%, "
        f"recurring_drop={params.recurring_drop_pct:.2f}%, "
        f"first_buy={params.first_buy_fraction * 100:.2f}%, "
        f"recurring_buy={params.recurring_buy_fraction * 100:.2f}%, "
        f"take_profit_margin={params.take_profit_margin_pct:.2f}%"
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    candles = load_candles(update_csv=not args.no_update)
    params_to_try = list(parameter_grid(args))
    settings = settings_from_args(args)
    print(f"Loaded {len(candles):,} candles from {CSV_PATH}")
    print(
        f"Starting cash=${args.starting_cash:,.2f}, leverage={args.leverage:.1f}x, "
        f"workers={args.workers}, combinations={len(params_to_try):,}."
    )
    print("Results print as workers finish, so completion order is intentionally not parameter-grid order.\n")

    results: list[BacktestResult] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(candles, settings),
    ) as executor:
        futures: dict[Any, tuple[int, StrategyParams]] = {}
        for run_number, params in enumerate(params_to_try, start=1):
            print_queued(run_number, len(params_to_try), params)
            futures[executor.submit(evaluate_parameter_set, params)] = (run_number, params)

        for future in as_completed(futures):
            run_number, params = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"    [{run_number:,}/{len(params_to_try):,}] Finished "
                f"profit=${result.profit:,.2f} ({result.profit_pct:.2f}%) "
                f"for initial={params.initial_drop_pct:.2f}%, recurring={params.recurring_drop_pct:.2f}%, "
                f"first={params.first_buy_fraction * 100:.2f}%, recurring_buy={params.recurring_buy_fraction * 100:.2f}%, "
                f"TP={params.take_profit_margin_pct:.2f}%"
            )

    results.sort(key=lambda item: item.profit, reverse=True)
    print("\nBest settings:")
    for rank, result in enumerate(results[: args.top], start=1):
        print(f"{rank:>2}. {format_result(result)}")


if __name__ == "__main__":
    main()
