#!/usr/bin/env python3
"""Run five concurrent HYPE paper-trading bots with a local monitoring dashboard.

This program intentionally never sends an order or reads exchange credentials. It
uses Hyperliquid's public market-data API to simulate the configured strategies
against the observed HYPE mid price.
"""

from __future__ import annotations

import argparse
import json
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

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                market = self.market_data.get_snapshot()
                with self._lock:
                    self.last_market = market
                    self.last_error = market.error
                    if market.price is not None and market.rolling_high is not None:
                        self._process_price(market.price, market.rolling_high)
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
        if not self.position.is_open:
            initial_trigger = rolling_high * (1.0 - params.initial_drop_pct / 100.0)
            self.cycle_start_equity = self.cash
            first_margin = self.cycle_start_equity * params.first_buy_fraction
            if price <= initial_trigger and self.cash >= first_margin:
                self._buy(first_margin, price, f"Initial paper buy at ${price:,.4f}.")
                self._trim_to_margin_cap(price)

        liquidated_this_tick = False
        while self.position.is_open and self.position.buy_count - 1 < MAX_ADDITIONAL_BUYS:
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

        if self.position.is_open and not liquidated_this_tick:
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

        current_equity = equity(self.cash, self.position, price)
        self.peak_equity = max(self.peak_equity, current_equity)
        if self.peak_equity > 0:
            self.max_drawdown_pct = max(
                self.max_drawdown_pct,
                (self.peak_equity - current_equity) / self.peak_equity * 100.0,
            )

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
                "status": "In position" if self.position.is_open else "Waiting for dip",
                "price": price,
                "rolling_high": rolling_high,
                "initial_trigger": initial_trigger,
                "portfolio_equity": current_equity,
                "profit": profit,
                "profit_pct": profit / BOT_STARTING_CASH * 100.0,
                "starting_cash": BOT_STARTING_CASH,
                "cash": self.cash,
                "margin_used": self.position.margin_used,
                "notional": self.position.margin_used * LEVERAGE,
                "quantity": self.position.quantity,
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
    .error { color:var(--down); border-top:1px solid var(--line); padding:10px 14px; font-size:11px; line-height:1.4; }
    @media (max-width:1300px) { .market { grid-template-columns:repeat(3, minmax(0, 1fr)); } .market-item:nth-child(3n) { border-right:0; } .market-item:nth-child(-n+3) { border-bottom:1px solid var(--line); } .grid { grid-template-columns:repeat(3, minmax(250px, 1fr)); } }
    @media (max-width:850px) { header { align-items:flex-start; flex-direction:column; } .updated { text-align:left; } .market { grid-template-columns:1fr 1fr; } .market-item:nth-child(2n) { border-right:0; } .market-item:nth-child(-n+4) { border-bottom:1px solid var(--line); } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>HYPE Paper Bot Monitor</h1><div class="subtitle">Five independent 5x leveraged simulations. No exchange orders are submitted.</div></div>
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
    const level = (label, value) => `<div class="level"><span>${label}</span><span>${value}</span></div>`;
    function render(data) {
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
        const isOpen = bot.status === 'In position';
        return `<article class="bot">
          <div class="bot-header"><div class="bot-name">${bot.name}</div><div class="badge ${isOpen ? 'open' : ''}">${bot.status}</div></div>
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
            ${level('Average entry', price(bot.average_entry))}
            ${level('Take-profit', price(bot.target_price))}
            ${level('Liquidation', price(bot.liquidation_price))}
            ${level('Initial trigger', price(bot.initial_trigger))}
            ${level('Buys / cycles', `${bot.buys} / ${bot.closed_cycles}`)}
            ${level('Max drawdown', bot.max_drawdown_pct.toFixed(2) + '%')}
          </div>
          <div class="action"><div class="label">Latest action</div><div>${bot.last_action}</div><div class="action-time">${time(bot.last_action_at)}</div></div>
          ${bot.last_error ? `<div class="error">Market-data issue: ${bot.last_error}</div>` : ''}
        </article>`;
      }).join('');
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
