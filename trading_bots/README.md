# HYPE Paper Trading Bots

`run_hype_paper_bots.py` starts five concurrent HYPE **paper-trading** bots using the five parameter sets supplied for the optimizer's top results. The combined paper portfolio starts at `$100,000`, allocated evenly as `$20,000` to each bot. By default, new cycles use 7x leverage, HYPE's 5% first-tier maintenance-margin model, a 12% cycle margin cap, 15 maximum average-down buys, and the reduced take-profit rule used by the optimizer.

Each new cycle calculates both its first-buy and recurring-buy margins from that bot's current portfolio equity. A profitable closed cycle therefore increases its following full buy ladder, while a loss reduces it.

The script only reads public Hyperliquid market data. It does not submit orders, connect a wallet, read private keys, or read exchange credentials.

Run from the project root:

```bash
python trading_bots/run_hype_paper_bots.py
```

It opens a dashboard at `http://127.0.0.1:8765`. Use `Ctrl+C` in the terminal to stop the five workers and the dashboard.

Useful options:

```bash
python trading_bots/run_hype_paper_bots.py --poll-seconds 30
python trading_bots/run_hype_paper_bots.py --port 8766 --no-browser
python trading_bots/run_hype_paper_bots.py --leverage 7 --dip-lookback-hours 168
```

By default, the public HYPE market snapshot is refreshed every minute. The bots use the highest completed 1-hour candle from the prior 96 hours (four days) to determine their initial-dip trigger. Change this with `--dip-lookback-hours` (or `--lookback-hours`). Change leverage for future cycles with `--leverage`; it must be greater than 1x and no more than HYPE's configured 10x limit. An open position restores the leverage it had when that cycle started, so restarting with new leverage or lookback settings does not alter an existing position. Once it fully closes or liquidates, its next initial buy uses the new settings. The dashboard itself refreshes every two seconds and displays the most recent snapshot between market checks. Its header shows combined portfolio equity and total P&L across the five bots. It also shows each bot's simulated portfolio equity and P&L, allocated margin, leveraged notional position, cash, average entry, take-profit level, modeled liquidation level, initial trigger, trades, cycles, max drawdown, and latest action.

Paper fills use the latest observed HYPE mid price after a level is crossed. Because polling cannot capture every intrasecond move, the live paper results can differ from the historical candle-based backtest. Funding, fees, slippage, and actual Hyperliquid margin rules are not modeled.
