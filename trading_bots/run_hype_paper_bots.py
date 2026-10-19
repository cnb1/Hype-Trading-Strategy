#!/usr/bin/env python3
"""Run five concurrent HYPE paper-trading bots with a local monitoring dashboard.

This program intentionally never sends an order or reads exchange credentials. It
uses Hyperliquid's public market-data API to simulate the configured strategies
against the observed HYPE mid price.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_DIR = PROJECT_ROOT / "strategy_optimizer"
if str(OPTIMIZER_DIR) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_DIR))

from optimize_scalping_strategy import (  # noqa: E402
    LEVERAGE,
    Position,
    StrategyParams,
    close_quantity,
    close_position,
    enforce_margin_cap,
    equity,
    liquidation_price,
    maintenance_margin_rate,
)


API_URL = "https://api.hyperliquid.xyz/info"
COIN = "HYPE"
LOOKBACK_HOURS = 96
LIQUIDATION_DROP_PCT = 15.0
MAX_CYCLE_MARGIN_PCT = 12.0
REDUCED_TARGET_MARGIN_THRESHOLD_PCT = 12.0
REDUCED_TARGET_MULTIPLIER = 0.5
MAX_ADDITIONAL_BUYS = 15
TOTAL_STARTING_CASH = 100_000.0
STATE_DIR = Path(__file__).resolve().parent / "state"
STATE_VERSION = 1


@dataclass(frozen=True)
class BotDefinition:
    name: str
    params: StrategyParams


BOT_DEFINITIONS = (
    BotDefinition("Bot 1", StrategyParams(8.0, 4.0, 0.10, 0.05, 10.0)),
    BotDefinition("Bot 2", StrategyParams(5.0, 3.0, 0.10, 0.025, 10.0)),
    BotDefinition("Bot 3", StrategyParams(5.0, 5.0, 0.075, 0.05, 20.0)),
    BotDefinition("Bot 4", StrategyParams(8.0, 5.0, 0.10, 0.075, 10.0)),
    BotDefinition("Bot 5", StrategyParams(5.0, 4.0, 0.10, 0.075, 30.0)),
)
BOT_STARTING_CASH = TOTAL_STARTING_CASH / len(BOT_DEFINITIONS)


@dataclass(frozen=True)
class MarketSnapshot:
    price: float | None
    rolling_high: float | None
    fetched_at: float
    error: str | None = None


class MarketDataCache:
    """Shares one public API snapshot among the five paper-trading threads."""

    def __init__(self, refresh_seconds: float) -> None:
        self.refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        self._snapshot: MarketSnapshot | None = None

    def get_snapshot(self) -> MarketSnapshot:
        with self._lock:
            now = time.time()
            if self._snapshot and now - self._snapshot.fetched_at < self.refresh_seconds:
                return self._snapshot

            try:
                price = self._fetch_mid_price()
                rolling_high = self._fetch_rolling_high(now)
                self._snapshot = MarketSnapshot(
                    price=price,
                    rolling_high=rolling_high,
                    fetched_at=now,
                )
            except requests.RequestException as error:
                if self._snapshot:
                    self._snapshot = replace(self._snapshot, fetched_at=now, error=str(error))
                else:
                    self._snapshot = MarketSnapshot(None, None, now, str(error))
            return self._snapshot

    @staticmethod
    def _post(payload: dict[str, Any]) -> Any:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    def _fetch_mid_price(self) -> float:
        mids = self._post({"type": "allMids"})
        return float(mids[COIN])

    def _fetch_rolling_high(self, now: float) -> float:
        end_time_ms = int(now * 1000)
        start_time_ms = end_time_ms - (LOOKBACK_HOURS + 2) * 60 * 60 * 1000
        candles = self._post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": COIN,
                    "interval": "1h",
                    "startTime": start_time_ms,
                    "endTime": end_time_ms,
                },
            }
        )
        current_hour_start_ms = end_time_ms - (end_time_ms % (60 * 60 * 1000))
        completed_candles = [candle for candle in candles if int(candle["t"]) < current_hour_start_ms]
        if not completed_candles:
            raise requests.RequestException("Hyperliquid returned no completed 1-hour candles.")
        return max(float(candle["h"]) for candle in completed_candles[-LOOKBACK_HOURS:])


class PaperBot(threading.Thread):
    """A paper-only worker that applies one optimizer configuration to live data."""

    def __init__(
        self,
        definition: BotDefinition,
        market_data: MarketDataCache,
        poll_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=definition.name, daemon=True)
        self.definition = definition
        self.market_data = market_data
        self.poll_seconds = poll_seconds
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self.cash = BOT_STARTING_CASH
        self.position = Position()
        self.cycle_start_equity = BOT_STARTING_CASH
        self.maintenance_rate = maintenance_margin_rate(LEVERAGE, LIQUIDATION_DROP_PCT)
        self.buys = 0
        self.closed_cycles = 0
        self.wins = 0
        self.liquidations = 0
        self.partial_reductions = 0
        self.margin_trimmed = 0.0
        self.reduced_target_exits = 0
        self.peak_equity = BOT_STARTING_CASH
        self.max_drawdown_pct = 0.0
        self.last_action = "Waiting for the initial dip."
        self.last_action_at: float | None = None
        self.last_error: str | None = None
        self.last_market: MarketSnapshot | None = None
        self.trading_enabled = True
        self.state_path = STATE_DIR / f"{self._state_slug(definition.name)}.json"
        self._load_state()

    @staticmethod
    def _state_slug(name: str) -> str:
        return "".join(character.lower() if character.isalnum() else "-" for character in name).strip("-")

    def _load_state(self) -> None:
        """Restore this paper bot's portfolio and counters after a restart."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") != STATE_VERSION or state.get("bot") != self.definition.name:
                raise ValueError("state version or bot identity does not match")
            values = state["portfolio"]
            self.cash = float(values["cash"])
            self.cycle_start_equity = float(values["cycle_start_equity"])
            position = values["position"]
            self.position = Position(**{key: float(position[key]) if key != "buy_count" else int(position[key])
                                        for key in ("margin_used", "notional", "quantity", "average_entry",
                                                    "first_entry", "last_buy_price", "buy_count")})
            for name in ("buys", "closed_cycles", "wins", "liquidations", "partial_reductions",
                         "reduced_target_exits"):
                setattr(self, name, int(values.get(name, 0)))
            self.margin_trimmed = float(values.get("margin_trimmed", 0.0))
            self.peak_equity = float(values.get("peak_equity", BOT_STARTING_CASH))
            self.max_drawdown_pct = float(values.get("max_drawdown_pct", 0.0))
            self.trading_enabled = bool(values.get("trading_enabled", True))
            self.last_action = str(values.get("last_action", "Restored saved bot state."))
            self.last_action_at = values.get("last_action_at")
            self.last_error = None
            self._record_action("Restored saved paper portfolio; resuming strategy.")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Cannot safely restore {self.definition.name} from {self.state_path}: {error}") from error

    def _save_state(self) -> None:
        """Atomically persist the complete simulated portfolio for this bot."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        position = self.position
        state = {
            "version": STATE_VERSION,
            "bot": self.definition.name,
            "saved_at": time.time(),
            "portfolio": {
                "cash": self.cash,
                "cycle_start_equity": self.cycle_start_equity,
                "position": {"margin_used": position.margin_used, "notional": position.notional,
                             "quantity": position.quantity, "average_entry": position.average_entry,
                             "first_entry": position.first_entry, "last_buy_price": position.last_buy_price,
                             "buy_count": position.buy_count,
                             "liquidation_price": liquidation_price(position, self.maintenance_rate)
                             if position.is_open else None},
                "buys": self.buys, "closed_cycles": self.closed_cycles, "wins": self.wins,
                "liquidations": self.liquidations, "partial_reductions": self.partial_reductions,
                "margin_trimmed": self.margin_trimmed, "reduced_target_exits": self.reduced_target_exits,
                "peak_equity": self.peak_equity, "max_drawdown_pct": self.max_drawdown_pct,
                "trading_enabled": self.trading_enabled,
                "last_action": self.last_action, "last_action_at": self.last_action_at,
            },
        }
        temporary_path = self.state_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary_path, self.state_path)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                market = self.market_data.get_snapshot()
                with self._lock:
                    self.last_market = market
                    self.last_error = market.error
                    if market.price is not None and market.rolling_high is not None:
                        self._process_price(market.price, market.rolling_high)
                    self._save_state()
            except Exception as error:  # Keep the remaining bots and dashboard alive.
                with self._lock:
                    self.last_error = f"{type(error).__name__}: {error}"
            self.stop_event.wait(self.poll_seconds)

    def _record_action(self, message: str) -> None:
        self.last_action = message
        self.last_action_at = time.time()

    def _buy(self, margin_amount: float, fill_price: float, action: str) -> None:
        self.cash -= margin_amount
        self.position.add_buy(margin_amount, fill_price, LEVERAGE)
        self.buys += 1
        self._record_action(action)

    def control(self, action: str, amount: float | None = None) -> None:
        """Apply a manual paper-trading action or pause/resume strategy automation."""
        with self._lock:
            if action == "pause":
                self.trading_enabled = False
                self._record_action("Automated entries, average-downs, and take-profit exits paused; liquidation monitoring remains active.")
            elif action == "resume":
                self.trading_enabled = True
                self._record_action("Automated strategy resumed.")
            elif action in {"buy", "sell", "add_margin"}:
                if amount is None or not math.isfinite(amount) or amount <= 0:
                    raise ValueError("Enter an amount greater than zero.")
                market = self.last_market
                if market is None or market.price is None:
                    raise ValueError("Market price is not available yet.")
                price = market.price
                if action == "buy":
                    if amount > self.cash:
                        raise ValueError(f"Insufficient available cash (${self.cash:,.2f}).")
                    if not self.position.is_open:
                        self.cycle_start_equity = self.cash
                    self._buy(amount, price, f"Manual paper buy with ${amount:,.2f} margin at ${price:,.4f}.")
                elif action == "add_margin":
                    if not self.position.is_open:
                        raise ValueError("Open a position before adding margin.")
                    if amount > self.cash:
                        raise ValueError(f"Insufficient available cash (${self.cash:,.2f}).")
                    self.cash -= amount
                    self.position.margin_used += amount
                    self._record_action(f"Added ${amount:,.2f} paper margin at ${price:,.4f}; position size unchanged.")
                else:
                    if not self.position.is_open:
                        raise ValueError("There is no open position to sell.")
                    if amount - self.position.quantity > 0.000001:
                        raise ValueError(f"Sell quantity exceeds the open position ({self.position.quantity:.6f} HYPE).")
                    quantity_sold = min(amount, self.position.quantity)
                    self.cash, realized_pnl = close_quantity(self.cash, self.position, quantity_sold, price)
                    if self.position.is_open:
                        self.partial_reductions += 1
                        self._record_action(
                            f"Manual paper sale of {quantity_sold:.6f} HYPE at ${price:,.4f}; "
                            f"realized P&L ${realized_pnl:,.2f}."
                        )
                    else:
                        self.closed_cycles += 1
                        self.wins += int(realized_pnl > 0)
                        self._record_action(
                            f"Manual paper sale closed the position at ${price:,.4f}; "
                            f"realized P&L ${realized_pnl:,.2f}."
                        )
            else:
                raise ValueError("Unknown bot action.")
            self._save_state()

    def _update_equity_metrics(self, price: float) -> None:
        current_equity = equity(self.cash, self.position, price)
        self.peak_equity = max(self.peak_equity, current_equity)
        if self.peak_equity > 0:
            self.max_drawdown_pct = max(
                self.max_drawdown_pct,
                (self.peak_equity - current_equity) / self.peak_equity * 100.0,
            )

    def _trim_to_margin_cap(self, fill_price: float) -> None:
        quantity_before_trim = self.position.quantity
        self.cash, was_trimmed, trimmed_amount = enforce_margin_cap(
            self.cash,
            self.position,
            fill_price,
            self.cycle_start_equity,
            MAX_CYCLE_MARGIN_PCT,
        )
        if was_trimmed:
            trimmed_quantity = quantity_before_trim - self.position.quantity
            self.partial_reductions += 1
            self.margin_trimmed += trimmed_amount
            self._record_action(
                f"Trimmed {trimmed_quantity:.4f} HYPE to keep cycle margin at {MAX_CYCLE_MARGIN_PCT:g}%."
            )

    def _liquidate(self, liquidation_price_value: float) -> None:
        lost_margin = self.position.margin_used
        self.position.__init__()
        self.closed_cycles += 1
        self.liquidations += 1
        self._record_action(f"Paper liquidation at ${liquidation_price_value:,.4f}; margin lost: ${lost_margin:,.2f}.")

    def _take_profit_target(self, mark_price: float) -> tuple[float, bool, float]:
        portfolio_equity = equity(self.cash, self.position, mark_price)
        margin_pct = self.position.margin_used / portfolio_equity * 100.0 if portfolio_equity > 0 else float("inf")
        use_reduced_target = margin_pct >= REDUCED_TARGET_MARGIN_THRESHOLD_PCT
        target_margin_pct = self.definition.params.take_profit_margin_pct
        if use_reduced_target:
            target_margin_pct *= REDUCED_TARGET_MULTIPLIER
        target_price = self.position.average_entry * (1.0 + target_margin_pct / (100.0 * LEVERAGE))
        return target_price, use_reduced_target, margin_pct

    def _process_price(self, price: float, rolling_high: float) -> None:
        params = self.definition.params
        if self.trading_enabled and not self.position.is_open:
            initial_trigger = rolling_high * (1.0 - params.initial_drop_pct / 100.0)
            self.cycle_start_equity = self.cash
            first_margin = self.cycle_start_equity * params.first_buy_fraction
            if price <= initial_trigger and self.cash >= first_margin:
                self._buy(first_margin, price, f"Initial paper buy at ${price:,.4f}.")
                self._trim_to_margin_cap(price)

        liquidated_this_tick = False
        while self.trading_enabled and self.position.is_open and self.position.buy_count - 1 < MAX_ADDITIONAL_BUYS:
            next_buy_price = self.position.last_buy_price * (1.0 - params.recurring_drop_pct / 100.0)
            current_liquidation_price = liquidation_price(self.position, self.maintenance_rate)
            if current_liquidation_price >= next_buy_price and price <= current_liquidation_price:
                self._liquidate(current_liquidation_price)
                liquidated_this_tick = True
                break

            recurring_margin = self.cycle_start_equity * params.recurring_buy_fraction
            if price > next_buy_price or self.cash < recurring_margin:
                break
            self._buy(recurring_margin, price, f"Average-down paper buy at ${price:,.4f}.")
            self._trim_to_margin_cap(price)

        if self.position.is_open:
            current_liquidation_price = liquidation_price(self.position, self.maintenance_rate)
            if price <= current_liquidation_price:
                self._liquidate(current_liquidation_price)
                liquidated_this_tick = True

        if self.trading_enabled and self.position.is_open and not liquidated_this_tick:
            target_price, used_reduced_target, margin_pct = self._take_profit_target(price)
            if price >= target_price:
                quantity_closed = self.position.quantity
                self.cash, was_win = close_position(self.cash, self.position, price)
                self.closed_cycles += 1
                self.wins += int(was_win)
                self.reduced_target_exits += int(used_reduced_target)
                target_label = "reduced " if used_reduced_target else ""
                self._record_action(
                    f"{target_label.title()}take-profit paper sell of {quantity_closed:.4f} HYPE at ${price:,.4f} "
                    f"({margin_pct:.2f}% margin exposure)."
                )

        self._update_equity_metrics(price)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            market = self.last_market
            price = market.price if market else None
            rolling_high = market.rolling_high if market else None
            current_equity = equity(self.cash, self.position, price) if price is not None else self.cash + self.position.margin_used
            profit = current_equity - BOT_STARTING_CASH
            initial_trigger = (
                rolling_high * (1.0 - self.definition.params.initial_drop_pct / 100.0)
                if rolling_high is not None
                else None
            )
            target_price = None
            liquidation_price_value = None
            reduced_target = False
            if self.position.is_open and price is not None:
                target_price, reduced_target, _ = self._take_profit_target(price)
                liquidation_price_value = liquidation_price(self.position, self.maintenance_rate)
            return {
                "name": self.definition.name,
                "parameters": {
                    "initial_drop_pct": self.definition.params.initial_drop_pct,
                    "recurring_drop_pct": self.definition.params.recurring_drop_pct,
                    "first_buy_pct": self.definition.params.first_buy_fraction * 100.0,
                    "recurring_buy_pct": self.definition.params.recurring_buy_fraction * 100.0,
                    "take_profit_margin_pct": self.definition.params.take_profit_margin_pct,
                },
                "bot_id": self._state_slug(self.definition.name),
                "position_open": self.position.is_open,
                "trading_enabled": self.trading_enabled,
                "status": ("In position" if self.position.is_open else "Waiting for dip")
                          if self.trading_enabled else ("Paused · in position" if self.position.is_open else "Paused"),
                "price": price,
                "rolling_high": rolling_high,
                "initial_trigger": initial_trigger,
                "portfolio_equity": current_equity,
                "profit": profit,
                "profit_pct": profit / BOT_STARTING_CASH * 100.0,
                "starting_cash": BOT_STARTING_CASH,
                "cash": self.cash,
                "margin_used": self.position.margin_used,
                "notional": self.position.notional,
                "quantity": self.position.quantity,
                "buy_count": self.position.buy_count,
                "average_entry": self.position.average_entry if self.position.is_open else None,
                "target_price": target_price,
                "liquidation_price": liquidation_price_value,
                "reduced_target": reduced_target,
                "buys": self.buys,
                "closed_cycles": self.closed_cycles,
                "wins": self.wins,
                "liquidations": self.liquidations,
                "partial_reductions": self.partial_reductions,
                "max_drawdown_pct": self.max_drawdown_pct,
                "last_action": self.last_action,
                "last_action_at": self.last_action_at,
                "last_error": self.last_error,
                "market_updated_at": market.fetched_at if market else None,
            }


class DashboardState:
    def __init__(self, bots: list[PaperBot], started_at: float) -> None:
        self.bots = bots
        self.started_at = started_at

    def snapshot(self) -> dict[str, Any]:
        bot_snapshots = [bot.snapshot() for bot in self.bots]
        total_portfolio_equity = sum(bot["portfolio_equity"] for bot in bot_snapshots)
        total_profit = total_portfolio_equity - TOTAL_STARTING_CASH
        return {
            "started_at": self.started_at,
            "leverage": LEVERAGE,
            "starting_cash": TOTAL_STARTING_CASH,
            "per_bot_starting_cash": BOT_STARTING_CASH,
            "total_portfolio_equity": total_portfolio_equity,
            "total_profit": total_profit,
            "total_profit_pct": total_profit / TOTAL_STARTING_CASH * 100.0,
            "paper_trading": True,
            "bots": bot_snapshots,
        }


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HYPE Paper Bot Monitor</title>
  <style>
    :root { color-scheme: dark; --bg:#0f1116; --panel:#181b23; --line:#2a2f3a; --text:#f0f2f5; --muted:#9299a8; --up:#16c784; --down:#ea3943; --blue:#4c8bf5; --amber:#f0b90b; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { display:flex; justify-content:space-between; gap:20px; align-items:flex-end; padding:26px max(24px, calc((100vw - 1500px)/2)); border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    .subtitle, .updated { color:var(--muted); margin-top:5px; font-size:12px; }
    .updated { text-align:right; }
    main { max-width:1500px; margin:0 auto; padding:20px 24px 36px; }
    .market { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); border:1px solid var(--line); background:var(--panel); }
    .market-item { padding:14px 16px; border-right:1px solid var(--line); min-height:70px; }
    .market-item:last-child { border-right:0; }
    .label { color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-size:10px; font-weight:700; }
    .value { margin-top:5px; font-size:18px; font-variant-numeric:tabular-nums; }
    .grid { display:grid; grid-template-columns:repeat(5, minmax(250px, 1fr)); gap:12px; margin-top:16px; }
    .bot { border:1px solid var(--line); background:var(--panel); min-width:0; }
    .bot-header { display:flex; justify-content:space-between; align-items:center; padding:14px 14px 12px; border-bottom:1px solid var(--line); }
    .bot-name { font-size:16px; font-weight:700; }
    .badge { border:1px solid var(--line); color:var(--muted); padding:4px 7px; font-size:10px; white-space:nowrap; }
    .badge.open { color:var(--up); border-color:rgba(22,199,132,.35); background:rgba(22,199,132,.08); }
    .badge.paused { color:var(--amber); border-color:rgba(240,185,11,.35); background:rgba(240,185,11,.08); }
    .params { display:grid; grid-template-columns:1fr 1fr; gap:0; padding:10px 14px; border-bottom:1px solid var(--line); }
    .param { padding:5px 0; font-size:11px; color:var(--muted); }
    .param b { display:block; color:#cdd2dc; font-weight:600; margin-top:2px; }
    .metrics { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); }
    .metric { background:var(--panel); padding:12px 14px; min-height:64px; }
    .metric .value { font-size:16px; }
    .positive { color:var(--up); } .negative { color:var(--down); } .neutral { color:var(--text); }
    .levels { margin:0; padding:12px 14px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
    .level { display:flex; justify-content:space-between; gap:8px; padding:4px 0; color:var(--muted); font-size:11px; }
    .level span:last-child { color:#dfe3ea; font-variant-numeric:tabular-nums; text-align:right; }
    .action { padding:12px 14px 14px; min-height:77px; color:#c8ced8; font-size:12px; line-height:1.45; }
    .action-time { color:var(--muted); font-size:10px; margin-top:8px; }
    .controls { padding:12px 14px; border-top:1px solid var(--line); }
    .control-row { display:grid; grid-template-columns:minmax(0, 1fr) auto; gap:6px; margin-top:7px; }
    .controls input { min-width:0; width:100%; border:1px solid var(--line); background:#10131a; color:var(--text); padding:8px; font-size:12px; }
    .controls button { border:1px solid #3b4659; background:#222a38; color:var(--text); padding:8px 10px; font-size:11px; cursor:pointer; }
    .controls button:hover:not(:disabled) { background:#303b4e; }
    .controls button:disabled { opacity:.45; cursor:not-allowed; }
    .control-heading { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:700; }
    .control-message { min-height:15px; margin-top:8px; color:var(--amber); font-size:10px; line-height:1.4; }
    .error { color:var(--down); border-top:1px solid var(--line); padding:10px 14px; font-size:11px; line-height:1.4; }
    @media (max-width:1300px) { .market { grid-template-columns:repeat(3, minmax(0, 1fr)); } .market-item:nth-child(3n) { border-right:0; } .market-item:nth-child(-n+3) { border-bottom:1px solid var(--line); } .grid { grid-template-columns:repeat(3, minmax(250px, 1fr)); } }
    @media (max-width:850px) { header { align-items:flex-start; flex-direction:column; } .updated { text-align:left; } .market { grid-template-columns:1fr 1fr; } .market-item:nth-child(2n) { border-right:0; } .market-item:nth-child(-n+4) { border-bottom:1px solid var(--line); } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>HYPE Paper Bot Monitor</h1><div class="subtitle">Five independent 5x paper simulations. Manual actions use the current public price; pausing disables strategy trades while liquidation monitoring continues. No exchange orders are submitted.</div></div>
    <div class="updated" id="updated">Connecting to market data...</div>
  </header>
  <main>
    <section class="market" id="market"></section>
    <section class="grid" id="bots"></section>
  </main>
  <script>
    const money = value => value == null ? '—' : new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', minimumFractionDigits:2, maximumFractionDigits:2}).format(value);
    const price = value => value == null ? '—' : '$' + Number(value).toLocaleString('en-US', {minimumFractionDigits:4, maximumFractionDigits:4});
    const number = value => value == null ? '—' : Number(value).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
    const time = value => value == null ? '—' : new Date(value * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    const signed = value => (value >= 0 ? '+' : '') + money(value);
    const color = value => value > 0 ? 'positive' : value < 0 ? 'negative' : 'neutral';
    const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const level = (label, value) => `<div class="level"><span>${label}</span><span>${value}</span></div>`;
    const controlMessages = {};
    const pendingControls = {};
    async function controlBot(event, botId, action, inputId) {
      if (pendingControls[botId]) return;
      pendingControls[botId] = true;
      const message = document.getElementById(`message-${botId}`);
      const button = event.currentTarget;
      if (button) button.disabled = true;
      try {
        const payload = {bot_id:botId, action};
        if (inputId) payload.amount = Number(document.getElementById(inputId).value);
        const response = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Action failed.');
        controlMessages[botId] = 'Action saved.';
        if (inputId) document.getElementById(inputId).value = '';
      } catch (error) {
        controlMessages[botId] = error.message;
      } finally {
        pendingControls[botId] = false;
        if (message) message.textContent = controlMessages[botId] || '';
        refresh();
      }
    }
    function render(data) {
      const focusedId = document.activeElement?.matches('[data-control-input]') ? document.activeElement.id : null;
      const savedInputValues = Object.fromEntries([...document.querySelectorAll('[data-control-input]')].map(input => [input.id, input.value]));
      const first = data.bots[0] || {};
      document.getElementById('updated').innerHTML = `Last public-market update: ${time(first.market_updated_at)}<br>Dashboard refreshes every 2 seconds`;
      document.getElementById('market').innerHTML = [
        ['Total portfolio', money(data.total_portfolio_equity), 'neutral'],
        ['Total P&L', signed(data.total_profit) + ` (${data.total_profit_pct >= 0 ? '+' : ''}${data.total_profit_pct.toFixed(2)}%)`, color(data.total_profit)],
        ['HYPE mid', price(first.price), 'neutral'], ['96h rolling high', price(first.rolling_high), 'neutral'],
        ['Leverage', data.leverage.toFixed(1) + 'x', 'neutral'], ['Mode', 'Paper trading', 'neutral']
      ].map(([label, value, valueClass]) => `<div class="market-item"><div class="label">${label}</div><div class="value ${valueClass}">${value}</div></div>`).join('');
      document.getElementById('bots').innerHTML = data.bots.map(bot => {
        const p = bot.parameters;
        const isOpen = bot.position_open;
        const paused = !bot.trading_enabled;
        const buyId = `${bot.bot_id}-buy-amount`;
        const sellId = `${bot.bot_id}-sell-amount`;
        const marginId = `${bot.bot_id}-margin-amount`;
        return `<article class="bot">
          <div class="bot-header"><div class="bot-name">${bot.name}</div><div class="badge ${paused ? 'paused' : (isOpen ? 'open' : '')}">${bot.status}</div></div>
          <div class="params">
            <div class="param">Initial dip<b>${p.initial_drop_pct}%</b></div><div class="param">Recurring dip<b>${p.recurring_drop_pct}%</b></div>
            <div class="param">First / recurring<b>${p.first_buy_pct}% / ${p.recurring_buy_pct}%</b></div><div class="param">Margin target<b>${p.take_profit_margin_pct}%${bot.reduced_target ? ' reduced' : ''}</b></div>
          </div>
          <div class="metrics">
            <div class="metric"><div class="label">Portfolio</div><div class="value">${money(bot.portfolio_equity)}</div></div>
            <div class="metric"><div class="label">P&amp;L</div><div class="value ${color(bot.profit)}">${signed(bot.profit)}<br><small>${bot.profit_pct >= 0 ? '+' : ''}${bot.profit_pct.toFixed(2)}%</small></div></div>
            <div class="metric"><div class="label">Margin used</div><div class="value">${money(bot.margin_used)}</div></div>
            <div class="metric"><div class="label">5x position</div><div class="value">${money(bot.notional)}</div></div>
          </div>
          <div class="levels">
            ${level('Available cash', money(bot.cash))}
            ${level('Position size', isOpen ? `${Number(bot.quantity).toFixed(6)} HYPE` : '—')}
            ${level('Average entry', price(bot.average_entry))}
            ${level('Take-profit', price(bot.target_price))}
            ${level('Liquidation', price(bot.liquidation_price))}
            ${level('Initial trigger', price(bot.initial_trigger))}
            ${level('Buys / cycles', `${bot.buys} / ${bot.closed_cycles}`)}
            ${level('Max drawdown', bot.max_drawdown_pct.toFixed(2) + '%')}
          </div>
          <div class="action"><div class="label">Latest action</div><div>${bot.last_action}</div><div class="action-time">${time(bot.last_action_at)}</div></div>
          <div class="controls">
            <div class="control-heading">Manual paper controls</div>
            <div class="control-row"><button onclick="controlBot(event, '${bot.bot_id}', '${paused ? 'resume' : 'pause'}')">${paused ? 'Resume automation' : 'Pause automation'}</button><span></span></div>
            <div class="control-row"><input id="${buyId}" data-control-input type="number" min="0.01" step="0.01" placeholder="Margin in USD" aria-label="${bot.name} buy margin in USD"><button onclick="controlBot(event, '${bot.bot_id}', 'buy', '${buyId}')">Buy</button></div>
            <div class="control-row"><input id="${sellId}" data-control-input type="number" min="0.000001" step="any" placeholder="HYPE quantity" aria-label="${bot.name} sell quantity" ${isOpen ? '' : 'disabled'}><button onclick="controlBot(event, '${bot.bot_id}', 'sell', '${sellId}')" ${isOpen ? '' : 'disabled'}>Sell</button></div>
            <div class="control-row"><input id="${marginId}" data-control-input type="number" min="0.01" step="0.01" placeholder="Margin in USD" aria-label="${bot.name} added margin" ${isOpen ? '' : 'disabled'}><button onclick="controlBot(event, '${bot.bot_id}', 'add_margin', '${marginId}')" ${isOpen ? '' : 'disabled'}>Add margin</button></div>
            <div class="control-message" id="message-${bot.bot_id}">${escapeHtml(controlMessages[bot.bot_id] || '')}</div>
          </div>
          ${bot.last_error ? `<div class="error">Market-data issue: ${bot.last_error}</div>` : ''}
        </article>`;
      }).join('');
      for (const [id, value] of Object.entries(savedInputValues)) {
        const input = document.getElementById(id);
        if (input) input.value = value;
      }
      for (const botId of Object.keys(pendingControls)) {
        if (pendingControls[botId]) document.querySelectorAll(`[onclick*="${botId}"]`).forEach(control => control.disabled = true);
      }
      if (focusedId) document.getElementById(focusedId)?.focus({preventScroll:true});
    }
    async function refresh() {
      try { render(await (await fetch('/api/state', {cache:'no-store'})).json()); }
      catch (error) { document.getElementById('updated').textContent = 'Dashboard connection lost. Retrying...'; }
    }
    refresh(); setInterval(refresh, 2000);
  </script>
</body>
</html>"""


def build_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = DASHBOARD_HTML.encode("utf-8")
                self._respond(HTTPStatus.OK, "text/html; charset=utf-8", body)
            elif self.path == "/api/state":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self._respond(HTTPStatus.OK, "application/json", body)
            else:
                self._respond(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/control":
                self._respond(HTTPStatus.NOT_FOUND, "application/json", b'{"error":"Not found."}')
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 4096:
                    raise ValueError("Invalid request size.")
                request = json.loads(self.rfile.read(content_length))
                bot_id = str(request["bot_id"])
                action = str(request["action"])
                bot = next((item for item in state.bots if item._state_slug(item.definition.name) == bot_id), None)
                if bot is None:
                    raise ValueError("Unknown bot.")
                amount = float(request["amount"]) if "amount" in request else None
                bot.control(action, amount)
                response = json.dumps({"ok": True, "bot": bot.snapshot()}).encode("utf-8")
                self._respond(HTTPStatus.OK, "application/json; charset=utf-8", response)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                response = json.dumps({"error": str(error)}).encode("utf-8")
                self._respond(HTTPStatus.BAD_REQUEST, "application/json; charset=utf-8", response)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _respond(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five HYPE paper-trading bots and a local dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host. Defaults to local-only access.")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port. Defaults to 8765.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Public market-data polling interval. Defaults to 1 minute.")
    parser.add_argument("--no-browser", action="store_true", help="Start the dashboard without opening a browser tab.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be greater than zero.")

    stop_event = threading.Event()
    market_data = MarketDataCache(refresh_seconds=args.poll_seconds)
    bots = [PaperBot(definition, market_data, args.poll_seconds, stop_event) for definition in BOT_DEFINITIONS]
    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), build_handler(DashboardState(bots, time.time())))
    for bot in bots:
        bot.start()
    print(f"Started {len(bots)} HYPE paper-trading worker threads.")
    print(f"Dashboard: {url}")
    print("Paper trading only: no orders, account credentials, or wallet access are used.")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping paper bots...")
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
        for bot in bots:
            bot.join(timeout=args.poll_seconds + 1)


if __name__ == "__main__":
    main()
