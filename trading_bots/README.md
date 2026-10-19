# HYPE Paper Trading Bots

`run_hype_paper_bots.py` starts five concurrent HYPE **paper-trading** bots using the five parameter sets supplied for the optimizer's top results. The combined paper portfolio starts at `$100,000`, allocated evenly as `$20,000` to each bot. Every bot uses the same 5x leverage, 15% initial liquidation-distance model, 12% cycle margin cap, 15 maximum average-down buys, and reduced take-profit rule as the optimizer.

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
```

By default, the public HYPE market snapshot is refreshed every minute. The bots use the highest completed 1-hour candle from the prior 96 hours (four days) to determine their initial-dip trigger. The dashboard itself refreshes every two seconds and displays the most recent snapshot between market checks. Its header shows combined portfolio equity and total P&L across the five bots. It also shows each bot's simulated portfolio equity and P&L, allocated margin, 5x notional position, cash, average entry, take-profit level, modeled liquidation level, initial trigger, trades, cycles, max drawdown, and latest action.

Paper fills use the latest observed HYPE mid price after a level is crossed. Because polling cannot capture every intrasecond move, the live paper results can differ from the historical candle-based backtest. Funding, fees, slippage, and actual Hyperliquid margin rules are not modeled.
