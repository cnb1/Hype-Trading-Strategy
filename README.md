# HYPE Trading Strategy

Standalone HYPE strategy research and paper-trading tools. This repository contains the historical optimizer, interactive trade viewer, concurrent paper-bot dashboard, candle updater, and local 1-hour and 4-hour HYPE candle datasets.

The paper bots use public Hyperliquid market data but never submit orders or access wallet credentials.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optimize Strategies

Run the single-process optimizer:

```bash
python strategy_optimizer/optimize_scalping_strategy.py
```

Run the multi-core optimizer:

```bash
python strategy_optimizer/optimize_scalping_strategy_parallel.py --workers 6
```

Both commands update `resources/hype_1h_candles.csv` by default. Add `--no-update` to use the local CSV without requesting newer candles.

## Review One Run

```bash
python strategy_optimizer/view_scalping_strategy.py \
  --initial-drop-pct 5 \
  --recurring-drop-pct 4 \
  --first-buy-fraction 10 \
  --recurring-buy-fraction 5 \
  --take-profit-margin-pct 15
```

The viewer creates an interactive HTML chart with every simulated action marked on the HYPE candlesticks.

## Paper Bots

```bash
python trading_bots/run_hype_paper_bots.py
```

This starts five independent paper-trading workers and a local dashboard at `http://127.0.0.1:8765`. The bots share public market data but maintain separate portfolios and strategy parameters.

Each bot saves its paper portfolio in its own JSON file under `trading_bots/state/`. On restart, it restores its cash, open position details, counters, and risk metrics before processing market data. The files are local runtime state and are excluded from Git. If a state file is missing, that bot starts with its configured starting cash and waits for its usual entry signal. These files track simulated positions only; the bots do not inspect or restore live exchange positions.

The dashboard has controls on each bot card to pause or resume strategy trades, then choose Buy, Sell, or Add margin from one action menu. All manual inputs are in USD: Buy and Sell use position value, and Add margin uses collateral. A Buy uses 5x leverage, so its margin requirement is one-fifth of the entered position value. While paused, automated entries, average-downs, and take-profit exits are disabled; liquidation monitoring remains active. Manual actions use the latest public HYPE mid price and are saved to that bot's JSON state immediately. Adding collateral reduces the modeled liquidation price without increasing position size. These controls only change paper portfolios; they do not send exchange orders.

Each bot card also has an expandable trade history with up to the latest 50 events. It records strategy and manual buys, sales, profitable or losing position closes, margin additions, and liquidations, including price, margin, leveraged value, quantity, realized P&L, and liquidation price after the event where applicable. New events are saved in the bot's JSON file. An already-open position from a state file without a history is shown as an imported position; older closed trades cannot be reconstructed if they were not previously logged.

## Layout

```text
resources/             Historical HYPE candles
src/hype_candles.py    Hyperliquid candle downloader and updater
strategy_optimizer/    Backtest optimizer, parallel runner, and chart viewer
trading_bots/          Five-bot paper-trading monitor and dashboard
```
