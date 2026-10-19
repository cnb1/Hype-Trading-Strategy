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

## Layout

```text
resources/             Historical HYPE candles
src/hype_candles.py    Hyperliquid candle downloader and updater
strategy_optimizer/    Backtest optimizer, parallel runner, and chart viewer
trading_bots/          Five-bot paper-trading monitor and dashboard
```
