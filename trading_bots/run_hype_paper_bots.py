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
    HYPE_MAX_LEVERAGE,
    LEVERAGE,
    Position,
    StrategyParams,
    close_quantity,
    close_position,
    enforce_margin_cap,
    equity,
    hype_maintenance_margin_rate,
    liquidation_price,
    maintenance_margin_rate,
)


API_URL = "https://api.hyperliquid.xyz/info"
COIN = "HYPE"
LOOKBACK_HOURS = 96
DEFAULT_BOT_LEVERAGE = 7.0
LEGACY_INITIAL_LIQUIDATION_DROP_PCT = 15.0
MAX_CYCLE_MARGIN_PCT = 12.0
REDUCED_TARGET_MARGIN_THRESHOLD_PCT = 12.0
REDUCED_TARGET_MULTIPLIER = 0.5
MAX_ADDITIONAL_BUYS = 15
TOTAL_STARTING_CASH = 100_000.0
STATE_DIR = Path(__file__).resolve().parent / "state"
STATE_VERSION = 2
MAX_TRADE_HISTORY = 50


@dataclass(frozen=True)
class BotDefinition:
    name: str
    params: StrategyParams


BOT_DEFINITIONS = (
    BotDefinition("Bot 1", StrategyParams(8.0, 2.0, 0.10, 0.025, 10.0)),
    BotDefinition("Bot 2", StrategyParams(8.0, 2.0, 0.10, 0.025, 15.0)),
    BotDefinition("Bot 3", StrategyParams(8.0, 2.0, 0.075, 0.025, 10.0)),
    BotDefinition("Bot 4", StrategyParams(8.0, 3.0, 0.10, 0.05, 10.0)),
    BotDefinition("Bot 5", StrategyParams(8.0, 4.0, 0.10, 0.075, 25.0)),
)
BOT_STARTING_CASH = TOTAL_STARTING_CASH / len(BOT_DEFINITIONS)


@dataclass(frozen=True)
class MarketSnapshot:
    price: float | None
    rolling_high: float | None
    price_history: list[dict[str, float]]
    fetched_at: float
    error: str | None = None


class MarketDataCache:
    """Shares one public API snapshot among the five paper-trading threads."""

    def __init__(self, refresh_seconds: float, lookback_hours: int) -> None:
        self.refresh_seconds = refresh_seconds
        self.lookback_hours = lookback_hours
        self._lock = threading.Lock()
        self._snapshot: MarketSnapshot | None = None

    def get_snapshot(self) -> MarketSnapshot:
        with self._lock:
            now = time.time()
            if self._snapshot and now - self._snapshot.fetched_at < self.refresh_seconds:
                return self._snapshot

            try:
                price = self._fetch_mid_price()
                rolling_high, price_history = self._fetch_price_history(now)
                self._snapshot = MarketSnapshot(
                    price=price,
                    rolling_high=rolling_high,
                    price_history=price_history,
                    fetched_at=now,
                )
            except requests.RequestException as error:
                if self._snapshot:
                    self._snapshot = replace(self._snapshot, fetched_at=now, error=str(error))
                else:
                    self._snapshot = MarketSnapshot(None, None, [], now, str(error))
            return self._snapshot

    @staticmethod
    def _post(payload: dict[str, Any]) -> Any:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    def _fetch_mid_price(self) -> float:
        mids = self._post({"type": "allMids"})
        return float(mids[COIN])

    def _fetch_price_history(self, now: float) -> tuple[float, list[dict[str, float]]]:
        end_time_ms = int(now * 1000)
        start_time_ms = end_time_ms - (self.lookback_hours + 2) * 60 * 60 * 1000
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
        recent_candles = completed_candles[-self.lookback_hours:]
        rolling_high = max(float(candle["h"]) for candle in recent_candles)
        price_history = [
            {"time": float(candle["t"]), "price": float(candle["c"])}
            for candle in recent_candles
        ]
        return rolling_high, price_history


class PaperBot(threading.Thread):
    """A paper-only worker that applies one optimizer configuration to live data."""

    def __init__(
        self,
        definition: BotDefinition,
        market_data: MarketDataCache,
        poll_seconds: float,
        stop_event: threading.Event,
        next_cycle_leverage: float,
        dip_lookback_hours: int,
    ) -> None:
        super().__init__(name=definition.name, daemon=True)
        self.definition = definition
        self.market_data = market_data
        self.poll_seconds = poll_seconds
        self.stop_event = stop_event
        self.next_cycle_leverage = next_cycle_leverage
        self.dip_lookback_hours = dip_lookback_hours
        self._lock = threading.Lock()
        self.cash = BOT_STARTING_CASH
        self.position = Position()
        self.cycle_start_equity = BOT_STARTING_CASH
        self.cycle_leverage: float | None = None
        self.maintenance_rate = hype_maintenance_margin_rate()
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
        self.trade_history: list[dict[str, Any]] = []
        self.state_path = STATE_DIR / f"{self._state_slug(definition.name)}.json"
        self._load_state()

    @staticmethod
    def _state_slug(name: str) -> str:
        return "".join(character.lower() if character.isalnum() else "-" for character in name).strip("-")

    @property
    def active_leverage(self) -> float:
        """Use the cycle's locked leverage until its position is fully closed."""
        return self.cycle_leverage if self.position.is_open and self.cycle_leverage is not None else self.next_cycle_leverage

    def _start_cycle(self) -> None:
        self.cycle_start_equity = self.cash
        self.cycle_leverage = self.next_cycle_leverage
        self.maintenance_rate = hype_maintenance_margin_rate()

    def _finish_cycle(self) -> None:
        self.cycle_leverage = None
        self.maintenance_rate = hype_maintenance_margin_rate()

    def _load_state(self) -> None:
        """Restore this paper bot's portfolio and counters after a restart."""
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("version") not in {1, STATE_VERSION} or state.get("bot") != self.definition.name:
                raise ValueError("state version or bot identity does not match")
            values = state["portfolio"]
            self.cash = float(values["cash"])
            self.cycle_start_equity = float(values["cycle_start_equity"])
            position = values["position"]
            self.position = Position(**{key: float(position[key]) if key != "buy_count" else int(position[key])
                                        for key in ("margin_used", "notional", "quantity", "average_entry",
                                                    "first_entry", "last_buy_price", "buy_count")})
            if self.position.is_open:
                # Version 1 had a fixed 5x bot configuration and did not save this field.
                self.cycle_leverage = float(values.get("cycle_leverage", LEVERAGE))
                self.maintenance_rate = float(values.get(
                    "cycle_maintenance_rate",
                    maintenance_margin_rate(LEVERAGE, LEGACY_INITIAL_LIQUIDATION_DROP_PCT),
                ))
            for name in ("buys", "closed_cycles", "wins", "liquidations", "partial_reductions",
                         "reduced_target_exits"):
                setattr(self, name, int(values.get(name, 0)))
            self.margin_trimmed = float(values.get("margin_trimmed", 0.0))
            self.peak_equity = float(values.get("peak_equity", BOT_STARTING_CASH))
            self.max_drawdown_pct = float(values.get("max_drawdown_pct", 0.0))
            self.trading_enabled = bool(values.get("trading_enabled", True))
            history = values.get("trade_history", [])
            if not isinstance(history, list):
                raise ValueError("trade history must be a list")
            self.trade_history = history[-MAX_TRADE_HISTORY:]
            if not self.trade_history and self.position.is_open:
                self._record_trade(
                    "position_imported", source="state_import", reason="open_position_before_history_tracking",
                    price=self.position.average_entry, margin=self.position.margin_used,
                    leveraged_value=self.position.notional, quantity=self.position.quantity,
                )
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
                "cycle_leverage": self.cycle_leverage,
                "cycle_maintenance_rate": self.maintenance_rate if self.position.is_open else None,
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
                "trade_history": self.trade_history[-MAX_TRADE_HISTORY:],
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

    def _record_trade(
        self,
        event: str,
        *,
        source: str,
        reason: str,
        price: float,
        margin: float,
        leveraged_value: float,
        quantity: float,
        realized_pnl: float | None = None,
        position_closed: bool = False,
        leverage: float | None = None,
    ) -> None:
        self.trade_history.append({
            "timestamp": time.time(),
            "event": event,
            "source": source,
            "reason": reason,
            "price": price,
            "margin": margin,
            "leverage": self.active_leverage if leverage is None else leverage,
            "leveraged_value": leveraged_value,
            "quantity": quantity,
            "realized_pnl": realized_pnl,
            "position_closed": position_closed,
            "profitable_close": realized_pnl > 0 if position_closed and realized_pnl is not None else None,
            "position_quantity_after": self.position.quantity,
            "position_leveraged_value_after": self.position.notional,
            "margin_used_after": self.position.margin_used,
            "average_entry_after": self.position.average_entry if self.position.is_open else None,
            "liquidation_price_after": liquidation_price(self.position, self.maintenance_rate)
            if self.position.is_open else None,
        })
        del self.trade_history[:-MAX_TRADE_HISTORY]

    def _buy(self, margin_amount: float, fill_price: float, action: str, source: str = "strategy") -> None:
        was_open = self.position.is_open
        self.cash -= margin_amount
        self.position.add_buy(margin_amount, fill_price, self.active_leverage)
        self.buys += 1
        self._record_action(action)
        self._record_trade(
            "average_down" if was_open and source == "strategy" else "buy",
            source=source,
            reason="average_down" if was_open and source == "strategy" else ("manual_buy" if source == "manual" else "entry"),
            price=fill_price,
            margin=margin_amount,
            leveraged_value=margin_amount * self.active_leverage,
            quantity=margin_amount * self.active_leverage / fill_price,
        )

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
                    margin_amount = amount / self.next_cycle_leverage if not self.position.is_open else amount / self.active_leverage
                    if margin_amount > self.cash:
                        raise ValueError(
                            f"This buy needs ${margin_amount:,.2f} margin, but only ${self.cash:,.2f} is available."
                        )
                    if not self.position.is_open:
                        self._start_cycle()
                    self._buy(
                        margin_amount, price,
                        f"Manual paper buy of ${amount:,.2f} position value using ${margin_amount:,.2f} margin at ${price:,.4f}.",
                        source="manual",
                    )
                elif action == "add_margin":
                    if not self.position.is_open:
                        raise ValueError("Open a position before adding margin.")
                    if amount > self.cash:
                        raise ValueError(f"Insufficient available cash (${self.cash:,.2f}).")
                    self.cash -= amount
                    self.position.margin_used += amount
                    self._record_action(f"Added ${amount:,.2f} paper margin at ${price:,.4f}; position size unchanged.")
                    self._record_trade(
                        "margin_added", source="manual", reason="collateral_added", price=price,
                        margin=amount, leveraged_value=0.0, quantity=0.0,
                    )
                else:
                    if not self.position.is_open:
                        raise ValueError("There is no open position to sell.")
                    current_position_value = self.position.quantity * price
                    if amount - current_position_value > 0.01:
                        raise ValueError(
                            f"Sell value exceeds the open position (${current_position_value:,.2f} at the current price)."
                        )
                    quantity_sold = self.position.quantity if current_position_value - amount <= 0.01 else amount / price
                    was_fully_closed = quantity_sold >= self.position.quantity
                    margin_released = self.position.margin_used * quantity_sold / self.position.quantity
                    cycle_leverage = self.active_leverage
                    self.cash, realized_pnl = close_quantity(self.cash, self.position, quantity_sold, price)
                    self._record_trade(
                        "position_closed" if was_fully_closed else "sell",
                        source="manual", reason="manual_close" if was_fully_closed else "manual_partial_sale",
                        price=price, margin=margin_released, leveraged_value=quantity_sold * price,
                        quantity=quantity_sold, realized_pnl=realized_pnl, position_closed=was_fully_closed,
                        leverage=cycle_leverage,
                    )
                    if self.position.is_open:
                        self.partial_reductions += 1
                        self._record_action(
                            f"Manual paper sale of {quantity_sold:.6f} HYPE at ${price:,.4f}; "
                            f"realized P&L ${realized_pnl:,.2f}."
                        )
                    else:
                        self.closed_cycles += 1
                        self.wins += int(realized_pnl > 0)
                        self._finish_cycle()
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
        average_entry_before_trim = self.position.average_entry
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
            self._record_trade(
                "sell", source="strategy", reason="margin_cap_reduction", price=fill_price,
                margin=trimmed_amount, leveraged_value=trimmed_quantity * fill_price,
                quantity=trimmed_quantity, realized_pnl=trimmed_quantity * (fill_price - average_entry_before_trim),
            )
            self._record_action(
                f"Trimmed {trimmed_quantity:.4f} HYPE to keep cycle margin at {MAX_CYCLE_MARGIN_PCT:g}%."
            )

    def _liquidate(self, liquidation_price_value: float) -> None:
        lost_margin = self.position.margin_used
        quantity = self.position.quantity
        notional = self.position.notional
        cycle_leverage = self.active_leverage
        self.position.__init__()
        self._record_trade(
            "liquidation", source="risk_monitor", reason="liquidation_price_reached",
            price=liquidation_price_value, margin=lost_margin, leveraged_value=notional,
            quantity=quantity, realized_pnl=-lost_margin, position_closed=True, leverage=cycle_leverage,
        )
        self.closed_cycles += 1
        self.liquidations += 1
        self._finish_cycle()
        self._record_action(f"Paper liquidation at ${liquidation_price_value:,.4f}; margin lost: ${lost_margin:,.2f}.")

    def _take_profit_target(self, mark_price: float) -> tuple[float, bool, float]:
        portfolio_equity = equity(self.cash, self.position, mark_price)
        margin_pct = self.position.margin_used / portfolio_equity * 100.0 if portfolio_equity > 0 else float("inf")
        use_reduced_target = margin_pct >= REDUCED_TARGET_MARGIN_THRESHOLD_PCT
        target_margin_pct = self.definition.params.take_profit_margin_pct
        if use_reduced_target:
            target_margin_pct *= REDUCED_TARGET_MULTIPLIER
        target_price = self.position.average_entry * (1.0 + target_margin_pct / (100.0 * self.active_leverage))
        return target_price, use_reduced_target, margin_pct

    def _process_price(self, price: float, rolling_high: float) -> None:
        params = self.definition.params
        if self.trading_enabled and not self.position.is_open:
            initial_trigger = rolling_high * (1.0 - params.initial_drop_pct / 100.0)
            self._start_cycle()
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
                closed_margin = self.position.margin_used
                cycle_leverage = self.active_leverage
                realized_pnl = quantity_closed * (price - self.position.average_entry)
                self.cash, was_win = close_position(self.cash, self.position, price)
                self._record_trade(
                    "position_closed", source="strategy", reason="take_profit",
                    price=price, margin=closed_margin, leveraged_value=quantity_closed * price,
                    quantity=quantity_closed, realized_pnl=realized_pnl, position_closed=True, leverage=cycle_leverage,
                )
                self.closed_cycles += 1
                self.wins += int(was_win)
                self.reduced_target_exits += int(used_reduced_target)
                self._finish_cycle()
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
            position_profit = (
                self.position.unrealized_pnl(price)
                if self.position.is_open and price is not None
                else (0.0 if not self.position.is_open else None)
            )
            position_profit_pct = (
                position_profit / self.position.margin_used * 100.0
                if position_profit is not None and self.position.margin_used > 0
                else (0.0 if not self.position.is_open else None)
            )
            first_buy_basis = self.cycle_start_equity if self.position.is_open else self.cash
            first_buy_margin = first_buy_basis * self.definition.params.first_buy_fraction
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
                "first_buy_label": "Current cycle first buy" if self.position.is_open else "Next first buy estimate",
                "first_buy_margin": first_buy_margin,
                "first_buy_position_value": first_buy_margin * self.active_leverage,
                "first_buy_basis": first_buy_basis,
                "leverage": self.active_leverage,
                "next_cycle_leverage": self.next_cycle_leverage,
                "dip_lookback_hours": self.dip_lookback_hours,
                "bot_id": self._state_slug(self.definition.name),
                "position_open": self.position.is_open,
                "trading_enabled": self.trading_enabled,
                "status": ("In position" if self.position.is_open else "Waiting for dip")
                          if self.trading_enabled else ("Paused · in position" if self.position.is_open else "Paused"),
                "price": price,
                "price_history": market.price_history if market else [],
                "rolling_high": rolling_high,
                "initial_trigger": initial_trigger,
                "portfolio_equity": current_equity,
                "profit": profit,
                "profit_pct": profit / BOT_STARTING_CASH * 100.0,
                "position_profit": position_profit,
                "position_profit_pct": position_profit_pct,
                "starting_cash": BOT_STARTING_CASH,
                "cash": self.cash,
                "margin_used": self.position.margin_used,
                "notional": self.position.notional,
                "quantity": self.position.quantity,
                "buy_count": self.position.buy_count,
                "trade_history": list(reversed(self.trade_history[-MAX_TRADE_HISTORY:])),
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
    def __init__(self, bots: list[PaperBot], started_at: float, next_cycle_leverage: float, dip_lookback_hours: int) -> None:
        self.bots = bots
        self.started_at = started_at
        self.next_cycle_leverage = next_cycle_leverage
        self.dip_lookback_hours = dip_lookback_hours

    def snapshot(self) -> dict[str, Any]:
        bot_snapshots = [bot.snapshot() for bot in self.bots]
        total_portfolio_equity = sum(bot["portfolio_equity"] for bot in bot_snapshots)
        total_profit = total_portfolio_equity - TOTAL_STARTING_CASH
        return {
            "started_at": self.started_at,
            "next_cycle_leverage": self.next_cycle_leverage,
            "dip_lookback_hours": self.dip_lookback_hours,
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
    header { display:flex; justify-content:space-between; gap:20px; align-items:flex-end; padding:clamp(12px, 2vw, 26px) clamp(10px, 2vw, 32px); border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    .subtitle, .updated { color:var(--muted); margin-top:5px; font-size:12px; }
    .updated { text-align:right; }
    main { width:100%; max-width:none; margin:0; padding:clamp(8px, 1vw, 16px); }
    .market { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); border:1px solid var(--line); background:var(--panel); }
    .market-item { padding:14px 16px; border-right:1px solid var(--line); min-height:70px; }
    .market-item:last-child { border-right:0; }
    .chart-panel { margin-top:12px; padding:12px 14px 8px; border:1px solid var(--line); background:var(--panel); }
    .chart-heading { display:flex; justify-content:space-between; align-items:baseline; gap:12px; color:var(--text); font-size:12px; }
    .chart-heading span { color:var(--muted); font-variant-numeric:tabular-nums; }
    .price-chart { height:clamp(135px, 21vh, 260px); margin-top:5px; }
    .price-chart svg { display:block; width:100%; height:100%; }
    .chart-grid { stroke:#2a2f3a; stroke-width:1; }
    .chart-line { fill:none; stroke:var(--blue); stroke-width:2.5; vector-effect:non-scaling-stroke; }
    .chart-dot { fill:var(--blue); }
    .chart-axis-label { fill:var(--muted); font-size:11px; font-family:inherit; }
    .chart-empty { height:100%; display:grid; place-items:center; color:var(--muted); font-size:12px; }
    .label { color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-size:clamp(8px, .65vw + 3px, 10px); font-weight:700; }
    .value { margin-top:5px; font-size:18px; font-variant-numeric:tabular-nums; }
    .grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:clamp(4px, .8vw, 12px); margin-top:16px; align-items:start; }
    .bot { border:1px solid var(--line); background:var(--panel); min-width:0; }
    .bot-header { display:flex; justify-content:space-between; align-items:center; padding:14px 14px 12px; border-bottom:1px solid var(--line); }
    .first-buy-stat { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:10px 14px; border-bottom:1px solid var(--line); background:rgba(76,139,245,.06); }
    .first-buy-copy { min-width:0; }
    .first-buy-label { color:var(--muted); font-size:clamp(8px, .65vw + 3px, 10px); text-transform:uppercase; letter-spacing:.04em; }
    .first-buy-basis { color:var(--muted); font-size:clamp(8px, .65vw + 3px, 10px); margin-top:3px; }
    .first-buy-values { text-align:right; min-width:0; font-variant-numeric:tabular-nums; }
    .first-buy-margin, .first-buy-notional { overflow-wrap:anywhere; }
    .first-buy-margin { color:var(--text); font-size:clamp(11px, 1vw + 4px, 16px); font-weight:700; }
    .first-buy-notional { color:var(--blue); font-size:clamp(8px, .7vw + 3px, 11px); margin-top:2px; }
    .bot-name { font-size:clamp(9px, 1vw + 4px, 16px); font-weight:700; }
    .badge { border:1px solid var(--line); color:var(--muted); padding:4px 7px; font-size:10px; white-space:nowrap; }
    .badge.open { color:var(--up); border-color:rgba(22,199,132,.35); background:rgba(22,199,132,.08); }
    .badge.paused { color:var(--amber); border-color:rgba(240,185,11,.35); background:rgba(240,185,11,.08); }
    .params { display:grid; grid-template-columns:1fr 1fr; gap:0; padding:10px 14px; border-bottom:1px solid var(--line); }
    .param { padding:5px 0; font-size:clamp(8px, .7vw + 3px, 11px); color:var(--muted); overflow-wrap:anywhere; }
    .param b { display:block; color:#cdd2dc; font-weight:600; margin-top:2px; }
    .metrics { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); }
    .metric { background:var(--panel); padding:12px 14px; min-height:64px; }
    .metric .value { font-size:clamp(10px, 1vw + 4px, 16px); overflow-wrap:anywhere; }
    .positive { color:var(--up); } .negative { color:var(--down); } .neutral { color:var(--text); }
    .levels { margin:0; padding:12px 14px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
    .level { display:flex; justify-content:space-between; gap:4px; padding:4px 0; color:var(--muted); font-size:clamp(8px, .7vw + 3px, 11px); }
    .level span:last-child { color:#dfe3ea; font-variant-numeric:tabular-nums; text-align:right; }
    .action { padding:12px 14px 14px; min-height:77px; color:#c8ced8; font-size:12px; line-height:1.45; }
    .action-time { color:var(--muted); font-size:10px; margin-top:8px; }
    .controls { padding:12px 14px; border-top:1px solid var(--line); }
    .control-row { display:grid; grid-template-columns:minmax(0, 1fr) auto; gap:6px; margin-top:7px; }
    .controls input { min-width:0; width:100%; border:1px solid var(--line); background:#10131a; color:var(--text); padding:8px; font-size:12px; }
    .controls select { min-width:0; width:100%; border:1px solid var(--line); background:#10131a; color:var(--text); padding:8px; font-size:12px; }
    .control-row.manual-control-row { grid-template-columns:minmax(0, 1fr) minmax(0, 1.2fr) auto; }
    .control-unit { color:var(--muted); text-align:right; font-size:10px; margin-top:4px; }
    .control-help { color:var(--muted); font-size:10px; line-height:1.4; margin-top:6px; }
    .controls button { border:1px solid #3b4659; background:#222a38; color:var(--text); padding:8px 10px; font-size:11px; cursor:pointer; }
    .controls button:hover:not(:disabled) { background:#303b4e; }
    .controls button:disabled { opacity:.45; cursor:not-allowed; }
    .control-heading { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:700; }
    .control-message { min-height:15px; margin-top:8px; color:var(--amber); font-size:10px; line-height:1.4; }
    .history { padding:0 14px 12px; }
    .history summary { color:var(--blue); font-size:11px; cursor:pointer; }
    .history-scroll { max-height:360px; overflow-x:scroll; overflow-y:scroll; overscroll-behavior:contain; scrollbar-gutter:stable; margin-top:9px; }
    .history-scroll::-webkit-scrollbar { width:10px; height:10px; }
    .history-scroll::-webkit-scrollbar-track { background:#10131a; }
    .history-scroll::-webkit-scrollbar-thumb { background:#41495a; border:2px solid #10131a; }
    .history table { width:100%; min-width:820px; border-collapse:collapse; font-size:10px; }
    .history th, .history td { padding:6px 7px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
    .history th { color:var(--muted); font-weight:600; }
    .history th:nth-child(2), .history td:nth-child(2) { text-align:left; }
    .history td { color:#dfe3ea; font-variant-numeric:tabular-nums; }
    .history-reason { color:var(--muted); }
    .history-empty { color:var(--muted); padding:8px; font-size:11px; }
    .error { color:var(--down); border-top:1px solid var(--line); padding:10px 14px; font-size:11px; line-height:1.4; }
    @media (max-width:1300px) { .market { grid-template-columns:repeat(3, minmax(0, 1fr)); } .market-item:nth-child(3n) { border-right:0; } .market-item:nth-child(-n+3) { border-bottom:1px solid var(--line); } }
    @media (max-width:850px) { header { align-items:flex-start; flex-direction:column; } .updated { text-align:left; } .market { grid-template-columns:1fr 1fr; } .market-item:nth-child(2n) { border-right:0; } .market-item:nth-child(-n+4) { border-bottom:1px solid var(--line); } .control-row.manual-control-row { grid-template-columns:1fr; } }
    @media (max-width:420px) { main { padding:10px 8px 24px; } .grid { gap:4px; } .bot-header, .params, .metric, .levels, .action, .controls { padding-left:6px; padding-right:6px; } .history { padding-left:6px; padding-right:6px; } }
  </style>
</head>
<body>
  <header>
    <div><h1>HYPE Paper Bot Monitor</h1><div class="subtitle">Five independent paper simulations. Manual actions use the current public price; pausing disables strategy trades while liquidation monitoring continues. No exchange orders are submitted.</div></div>
    <div class="updated" id="updated">Connecting to market data...</div>
  </header>
  <main>
    <section class="market" id="market"></section>
    <section class="chart-panel">
      <div class="chart-heading"><strong>HYPE / USD · 1-hour candles</strong><span id="chart-last-price">Waiting for market data...</span></div>
      <div class="price-chart" id="price-chart"><div class="chart-empty">Loading HYPE price history...</div></div>
    </section>
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
    function renderPriceChart(bot) {
      const chart = document.getElementById('price-chart');
      const priceLabel = document.getElementById('chart-last-price');
      const history = (bot.price_history || []).map(point => ({time:Number(point.time), price:Number(point.price)}))
        .filter(point => Number.isFinite(point.time) && Number.isFinite(point.price));
      if (bot.price != null && Number.isFinite(Number(bot.price))) {
        history.push({time:Number(bot.market_updated_at || Date.now() / 1000) * 1000, price:Number(bot.price)});
      }
      if (history.length < 2) {
        chart.innerHTML = '<div class="chart-empty">Price history will appear after completed hourly candles load.</div>';
        priceLabel.textContent = bot.price == null ? 'Waiting for market data...' : price(bot.price);
        return;
      }
      const width = 1200, height = 240, left = 68, right = 16, top = 12, bottom = 28;
      const values = history.map(point => point.price);
      const low = Math.min(...values), high = Math.max(...values);
      const spread = Math.max(high - low, high * 0.002, 0.0001);
      const min = low - spread * 0.08, max = high + spread * 0.08;
      const x = index => left + index * (width - left - right) / (history.length - 1);
      const y = value => top + (max - value) * (height - top - bottom) / (max - min);
      const path = history.map((point, index) => `${index ? 'L' : 'M'}${x(index).toFixed(1)},${y(point.price).toFixed(1)}`).join(' ');
      const grid = Array.from({length:5}, (_, index) => {
        const value = max - index * (max - min) / 4;
        const yy = y(value).toFixed(1);
        return `<line class="chart-grid" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"/><text class="chart-axis-label" x="${left-8}" y="${Number(yy)+4}" text-anchor="end">${price(value)}</text>`;
      }).join('');
      const last = history[history.length - 1];
      chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="HYPE hourly price chart">
        ${grid}<path class="chart-line" d="${path}"/><circle class="chart-dot" cx="${x(history.length-1)}" cy="${y(last.price)}" r="4"/>
        <text class="chart-axis-label" x="${left}" y="${height-5}">${new Date(history[0].time).toLocaleDateString()}</text>
        <text class="chart-axis-label" x="${width-right}" y="${height-5}" text-anchor="end">${new Date(last.time).toLocaleString()}</text>
      </svg>`;
      priceLabel.textContent = `Current mid ${price(bot.price)} · ${history.length - (bot.price != null ? 1 : 0)} hourly candles`;
    }
    const eventLabel = trade => {
      if (trade.event === 'position_closed') return trade.profitable_close ? 'Position closed · profit' : 'Position closed';
      return ({buy:'Buy', average_down:'Average down', sell:'Sell', liquidation:'Liquidation', margin_added:'Margin added', position_imported:'Imported position'}[trade.event] || trade.event);
    };
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
    function updateManualControl(botId) {
      const action = document.getElementById(`${botId}-action`)?.value || 'buy';
      const amount = document.getElementById(`${botId}-amount`);
      const placeholder = {
        buy:'Buy USD position value',
        sell:'Sell USD position value',
        add_margin:'Add USD collateral',
      }[action] || 'Amount in USD';
      if (amount) {
        amount.placeholder = placeholder;
        amount.setAttribute('aria-label', placeholder);
        amount.step = '0.01';
        amount.min = '0.01';
      }
      const unit = document.getElementById(`${botId}-amount-unit`);
      if (unit) unit.textContent = 'All amounts in USD';
    }
    function render(data) {
      const focusedId = document.activeElement?.matches('[data-control-input]') ? document.activeElement.id : null;
      let selectionStart = null;
      let selectionEnd = null;
      if (focusedId && document.activeElement instanceof HTMLInputElement) {
        selectionStart = document.activeElement.selectionStart;
        selectionEnd = document.activeElement.selectionEnd;
      }
      const savedInputValues = Object.fromEntries([...document.querySelectorAll('[data-control-input]')].map(input => [input.id, input.value]));
      const openHistoryIds = new Set([...document.querySelectorAll('details[data-history-id][open]')].map(details => details.dataset.historyId));
      const first = data.bots[0] || {};
      document.getElementById('updated').innerHTML = `Last public-market update: ${time(first.market_updated_at)}<br>Dashboard refreshes every 2 seconds`;
      document.getElementById('market').innerHTML = [
        ['Total portfolio', money(data.total_portfolio_equity), 'neutral'],
        ['Total P&L', signed(data.total_profit) + ` (${data.total_profit_pct >= 0 ? '+' : ''}${data.total_profit_pct.toFixed(2)}%)`, color(data.total_profit)],
        ['HYPE mid', price(first.price), 'neutral'], [`${data.dip_lookback_hours}h rolling high`, price(first.rolling_high), 'neutral'],
        ['Next-cycle leverage', data.next_cycle_leverage.toFixed(1) + 'x', 'neutral'], ['Mode', 'Paper trading', 'neutral']
      ].map(([label, value, valueClass]) => `<div class="market-item"><div class="label">${label}</div><div class="value ${valueClass}">${value}</div></div>`).join('');
      renderPriceChart(first);
      document.getElementById('bots').innerHTML = data.bots.map(bot => {
        const p = bot.parameters;
        const isOpen = bot.position_open;
        const paused = !bot.trading_enabled;
        const actionId = `${bot.bot_id}-action`;
        const amountId = `${bot.bot_id}-amount`;
        const history = bot.trade_history || [];
        const historyId = `history-${bot.bot_id}`;
        const historyRows = history.map(trade => `<tr>
          <td>${new Date(trade.timestamp * 1000).toLocaleString()}</td>
          <td>${eventLabel(trade)}<br><small>${escapeHtml(trade.source)} · <span class="history-reason">${escapeHtml((trade.reason || '').replaceAll('_', ' '))}</span></small></td>
          <td>${price(trade.price)}</td>
          <td>${money(trade.margin)}</td>
          <td>${money(trade.leveraged_value)}</td>
          <td>${Number(trade.quantity).toFixed(6)}</td>
          <td class="${color(trade.realized_pnl || 0)}">${trade.realized_pnl == null ? '—' : signed(trade.realized_pnl)}</td>
          <td>${price(trade.liquidation_price_after)}</td>
        </tr>`).join('');
        return `<article class="bot">
          <div class="bot-header"><div class="bot-name">${bot.name}</div><div class="badge ${paused ? 'paused' : (isOpen ? 'open' : '')}">${bot.status}</div></div>
          <div class="first-buy-stat">
            <div class="first-buy-copy"><div class="first-buy-label">${bot.first_buy_label}</div><div class="first-buy-basis">${(bot.parameters.first_buy_pct).toFixed(1)}% of ${money(bot.first_buy_basis)}</div></div>
            <div class="first-buy-values"><div class="first-buy-margin">${money(bot.first_buy_margin)} margin</div><div class="first-buy-notional">${money(bot.first_buy_position_value)} position value at ${bot.leverage.toFixed(1)}×</div></div>
          </div>
          <div class="params">
            <div class="param">Initial dip<b>${p.initial_drop_pct}%</b></div><div class="param">Recurring dip<b>${p.recurring_drop_pct}%</b></div>
            <div class="param">First / recurring<b>${p.first_buy_pct}% / ${p.recurring_buy_pct}%</b></div><div class="param">Margin target<b>${p.take_profit_margin_pct}%${bot.reduced_target ? ' reduced' : ''}</b></div>
          </div>
          <div class="metrics">
            <div class="metric"><div class="label">Portfolio</div><div class="value">${money(bot.portfolio_equity)}</div></div>
            <div class="metric"><div class="label">Total P&amp;L</div><div class="value ${color(bot.profit)}">${signed(bot.profit)}<br><small>${bot.profit_pct >= 0 ? '+' : ''}${bot.profit_pct.toFixed(2)}%</small></div></div>
            <div class="metric"><div class="label">Current position P&amp;L</div><div class="value ${color(bot.position_profit || 0)}">${bot.position_profit == null ? '—' : signed(bot.position_profit)}<br><small>${bot.position_profit_pct == null ? '—' : `${bot.position_profit_pct >= 0 ? '+' : ''}${bot.position_profit_pct.toFixed(2)}% of margin`}</small></div></div>
            <div class="metric"><div class="label">Margin used</div><div class="value">${money(bot.margin_used)}</div></div>
            <div class="metric"><div class="label">${bot.leverage.toFixed(1)}x position</div><div class="value">${money(bot.notional)}</div></div>
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
            <div class="control-row manual-control-row">
              <select id="${actionId}" data-control-input onchange="updateManualControl('${bot.bot_id}')" aria-label="${bot.name} manual action">
                <option value="buy">Buy</option>
                <option value="sell" ${isOpen ? '' : 'disabled'}>Sell</option>
                <option value="add_margin" ${isOpen ? '' : 'disabled'}>Add margin</option>
              </select>
              <input id="${amountId}" data-control-input type="number" min="0.01" step="0.01" placeholder="Buy USD position value" aria-label="Buy USD position value">
              <button onclick="controlBot(event, '${bot.bot_id}', document.getElementById('${actionId}').value, '${amountId}')">Submit</button>
            </div>
            <div class="control-unit" id="${bot.bot_id}-amount-unit">All amounts in USD</div>
            <div class="control-help">Buy and sell use USD position value. New buys use the cycle's displayed leverage; add margin is USD collateral.</div>
            <div class="control-message" id="message-${bot.bot_id}">${escapeHtml(controlMessages[bot.bot_id] || '')}</div>
          </div>
          <details class="history" data-history-id="${historyId}">
            <summary>Trade history (${history.length}/50)</summary>
            <div class="history-scroll"><table>
              <thead><tr><th>Time</th><th>Event</th><th>Price</th><th>Margin</th><th>Leveraged value</th><th>HYPE</th><th>Realized P&amp;L</th><th>Liquidation after</th></tr></thead>
              <tbody>${historyRows || '<tr><td colspan="8" class="history-empty">No trades recorded yet.</td></tr>'}</tbody>
            </table></div>
          </details>
          ${bot.last_error ? `<div class="error">Market-data issue: ${bot.last_error}</div>` : ''}
        </article>`;
      }).join('');
      for (const [id, value] of Object.entries(savedInputValues)) {
        const input = document.getElementById(id);
        if (input) input.value = value;
      }
      for (const bot of data.bots) updateManualControl(bot.bot_id);
      for (const botId of Object.keys(pendingControls)) {
        if (pendingControls[botId]) document.querySelectorAll(`[onclick*="${botId}"]`).forEach(control => control.disabled = true);
      }
      for (const details of document.querySelectorAll('details[data-history-id]')) {
        details.open = openHistoryIds.has(details.dataset.historyId);
      }
      if (focusedId) {
        const focusedElement = document.getElementById(focusedId);
        focusedElement?.focus({preventScroll:true});
        if (focusedElement instanceof HTMLInputElement && selectionStart != null && selectionEnd != null) {
          try { focusedElement.setSelectionRange(selectionStart, selectionEnd); } catch (error) {}
        }
      }
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
            except OSError as error:
                response = json.dumps({"error": f"Manual action was applied in memory, but state could not be saved: {error}"}).encode("utf-8")
                self._respond(HTTPStatus.INTERNAL_SERVER_ERROR, "application/json; charset=utf-8", response)

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
    parser.add_argument(
        "--leverage",
        type=float,
        default=DEFAULT_BOT_LEVERAGE,
        help=f"Leverage for future position cycles. Defaults to {DEFAULT_BOT_LEVERAGE:g}x.",
    )
    parser.add_argument(
        "--dip-lookback-hours",
        "--lookback-hours",
        dest="dip_lookback_hours",
        type=int,
        default=LOOKBACK_HOURS,
        help=f"Completed 1-hour candles used for future initial-dip triggers. Defaults to {LOOKBACK_HOURS}.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Start the dashboard without opening a browser tab.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be greater than zero.")
    if not 1 < args.leverage <= HYPE_MAX_LEVERAGE:
        raise ValueError(f"--leverage must be greater than 1 and no more than HYPE's {HYPE_MAX_LEVERAGE:g}x limit.")
    if args.dip_lookback_hours < 1:
        raise ValueError("--dip-lookback-hours must be at least 1.")

    stop_event = threading.Event()
    market_data = MarketDataCache(args.poll_seconds, args.dip_lookback_hours)
    bots = [
        PaperBot(definition, market_data, args.poll_seconds, stop_event, args.leverage, args.dip_lookback_hours)
        for definition in BOT_DEFINITIONS
    ]
    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(DashboardState(bots, time.time(), args.leverage, args.dip_lookback_hours)),
    )
    for bot in bots:
        bot.start()
    print(f"Started {len(bots)} HYPE paper-trading worker threads.")
    print(f"Future cycles: {args.leverage:g}x leverage, {args.dip_lookback_hours}h initial-dip lookback.")
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
