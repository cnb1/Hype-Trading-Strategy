# Strategy Optimizer

This folder contains a brute-force backtester for the leveraged HYPE dip-buying scalping strategy using 1-hour candles. By default, the optimizer tests both 5x and 6x leverage.

Run it from the project root:

```bash
python strategy_optimizer/optimize_scalping_strategy.py
```

By default it:

- calls `update_hype_csv()` from `src/hype_candles.py`
- loads and updates `resources/hype_1h_candles.csv`; Hyperliquid retains the most recent 5,000 1-hour candles
- starts with `$15,000`
- tests 5x and 6x leverage (change the grid with `--leverages`, or select one value with `--leverage`)
- recalculates each new cycle's first and recurring-buy allocations from the current account equity, so the full buy ladder compounds after profitable closes and shrinks after losses
- uses HYPE's 5% maintenance-margin rate for the notional tier below $20M, then recalculates liquidation after every average-down using the live average entry and allocated margin
- limits allocated margin in a buying cycle to 12% of the account equity immediately before that cycle began; an add that exceeds the cap is partially sold at its fill price to return the allocated margin to the cap
- prints every parameter set it tries and the total profit for that run
- prints the best parameter sets at the end

All trade triggers are range-based: initial and recurring buys use the candle **low**, take-profit exits use the candle **high**, and liquidation checks use the candle **low**. Open and close values are never used to trigger an order; close is used only to value an open position at the end of a candle or the end of the data.

The simulator models each long candle as **low-first**: it fills every recurring buy level crossed on the move down (up to 15 additional buys by default), recalculates liquidation after every fill, and then checks whether the candle high reaches the take-profit target based on the fully averaged entry. It closes the entire position at that target and waits for a later candle to start the next cycle. Because a 1-hour OHLC candle cannot reveal its true intrabar order, this is a defined approximation rather than a record of exact market sequencing.

When allocated margin reaches 12% of the current mark-to-market portfolio equity at the candle low, the recovery target is reduced to half of the configured take-profit margin percentage. For example, a 15% target becomes 7.5%. Change the threshold with `--reduced-target-margin-threshold-pct` or the half-target factor with `--reduced-target-multiplier`.

Use `--no-update` to run only on the CSV already on disk:

```bash
python strategy_optimizer/optimize_scalping_strategy.py --no-update
```

## Faster Parallel Runner

`optimize_scalping_strategy_parallel.py` is a separate multi-core runner that leaves the original optimizer untouched. It uses processes, rather than Python threads, so CPU-bound backtests can run on multiple cores. The parameter settings and result calculations are the same as the original optimizer.

```bash
python strategy_optimizer/optimize_scalping_strategy_parallel.py --workers 6
```

It defaults to the available CPU count, capped at eight workers. Use `--workers 1` to run it serially. Results finish out of parameter-grid order, but the final best-settings list is still sorted by profit.

Example with custom parameter grids:

```bash
python strategy_optimizer/optimize_scalping_strategy.py \
  --initial-drop-pcts 3,4,5 \
  --recurring-drop-pcts 3,5,7 \
  --first-buy-fractions 5,7.5,10 \
  --recurring-buy-fractions 2.5,5 \
  --take-profit-margin-pcts 15,20,25
```

Fractions can be entered as percentages (`5`) or decimals (`0.05`). A 20% take-profit target means 20% return on margin, so at 5x leverage the script targets roughly a 4% asset-price move above weighted average entry.

The 12% cap refers to **allocated margin**, not 5x notional exposure. This matches a 5% first purchase: its notional is roughly 25% of equity but the capital allocated to the position is 5%. Change the cap with `--max-cycle-margin-pct`, for example `--max-cycle-margin-pct 10`.

The liquidation model follows Hyperliquid's isolated-position formula for HYPE's first margin tier. HYPE has a 10x max leverage below $20M notional, so its maintenance-margin rate is half its max-leverage initial-margin rate: 5%. At the optimizer's position sizes, the tier has no maintenance deduction. For a long position, the modeled price is `(quantity × average_entry − isolated_margin) / (quantity × (1 − 0.05))`. On the first entry, where isolated margin is notional divided by selected leverage, this is `entry_price × (1 − 1 / leverage) / 0.95`: approximately 84.21% of entry at 5x and 87.72% at 6x. For a $100 entry, these are about $84.21 and $87.72, respectively. Liquidation is checked against candle lows as an approximation; Hyperliquid liquidates using mark price. The model does not include margin tiers above $20M, maintenance deductions, fees, funding, slippage, or intrahour price order. Cross-margin liquidation can also depend on other positions and account equity. Use results to compare configurations, not as an expected live return.

## Inspect one strategy run

`view_scalping_strategy.py` runs one parameter set through the same backtest and writes an interactive HTML candlestick chart. It marks every initial buy, average-down buy, take-profit sell, margin-cap trim, and liquidation. Use the mouse wheel to zoom, drag to pan, and the bottom range slider to jump through the history.

```bash
python strategy_optimizer/view_scalping_strategy.py \
  --initial-drop-pct 5 \
  --recurring-drop-pct 5 \
  --first-buy-fraction 5 \
  --recurring-buy-fraction 5 \
  --take-profit-margin-pct 20
```

It updates the candle CSV by default and opens the chart in the browser. Add `--no-update` to use the existing data, `--no-browser` to only write the file, and `--start` / `--end` to inspect a smaller date range. The default output is `strategy_optimizer/output/hype_1h_scalping_strategy_chart.html`.
