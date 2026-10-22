#!/usr/bin/env python3
"""Render one parameterized scalping backtest as an interactive trade chart."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from optimize_scalping_strategy import (  # noqa: E402
    LEVERAGE,
    STARTING_CASH,
    StrategyParams,
    TradeEvent,
    load_candles,
    simulate_strategy,
)


EVENT_STYLES = {
    "initial_buy": {"name": "Initial buy", "color": "#089981", "symbol": "triangle-up"},
    "recurring_buy": {"name": "Average-down buy", "color": "#4c8bf5", "symbol": "triangle-up"},
    "take_profit": {"name": "Take-profit sell", "color": "#f0b90b", "symbol": "triangle-down"},
    "margin_trim": {"name": "Margin-cap trim", "color": "#ff9800", "symbol": "triangle-down"},
    "liquidation": {"name": "Liquidation", "color": "#f23645", "symbol": "x"},
}


def parse_fraction(value: float) -> float:
    """Accept either a decimal fraction or a human-friendly percentage."""
    return value / 100.0 if value > 1 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show every simulated HYPE scalping fill on an interactive 1-hour candlestick chart."
    )
    parser.add_argument("--starting-cash", type=float, default=STARTING_CASH)
    parser.add_argument("--leverage", type=float, default=LEVERAGE)
    parser.add_argument("--initial-drop-pct", type=float, required=True)
    parser.add_argument("--recurring-drop-pct", type=float, required=True)
    parser.add_argument(
        "--first-buy-fraction",
        type=float,
        required=True,
        help="First-buy margin allocation as a percent (for example 5) or decimal (0.05).",
    )
    parser.add_argument(
        "--recurring-buy-fraction",
        type=float,
        required=True,
        help="Average-down margin allocation as a percent (for example 5) or decimal (0.05).",
    )
    parser.add_argument("--take-profit-margin-pct", type=float, required=True)
    parser.add_argument("--max-cycle-margin-pct", type=float, default=12.0)
    parser.add_argument(
        "--reduced-target-margin-threshold-pct",
        type=float,
        default=12.0,
        help="Use half target when allocated margin reaches this percent of current portfolio equity.",
    )
    parser.add_argument(
        "--reduced-target-multiplier",
        type=float,
        default=0.5,
        help="Multiplier for the take-profit margin target after the exposure threshold is reached.",
    )
    parser.add_argument("--dip-lookback-hours", type=int, default=168)
    parser.add_argument(
        "--max-additional-buys",
        type=int,
        default=15,
        help="Maximum average-down buys after the initial purchase (15 by default).",
    )
    parser.add_argument("--start", help="Optional inclusive ISO timestamp, for example 2026-01-01.")
    parser.add_argument("--end", help="Optional inclusive ISO timestamp, for example 2026-03-31.")
    parser.add_argument("--no-update", action="store_true", help="Use the CSV already on disk.")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "output" / "hype_1h_scalping_strategy_chart.html",
        help="Destination HTML file for the interactive chart.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Create the HTML file without opening it.")
    return parser.parse_args()


def filter_candles(candles: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if not start and not end:
        return candles

    if "open_time_iso" in candles.columns:
        timestamps = pd.to_datetime(candles["open_time_iso"], utc=True)
    else:
        timestamps = pd.to_datetime(candles["open_time"], unit="ms", utc=True)

    filtered = candles.copy()
    if start:
        filtered = filtered[timestamps >= pd.Timestamp(start, tz="UTC")]
        timestamps = timestamps[timestamps >= pd.Timestamp(start, tz="UTC")]
    if end:
        end_timestamp = pd.Timestamp(end, tz="UTC")
        if len(end) == 10:
            end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        filtered = filtered[timestamps <= end_timestamp]
    if filtered.empty:
        raise ValueError("No candles remain after applying --start/--end.")
    return filtered.reset_index(drop=True)


def candle_timestamps(candles: pd.DataFrame) -> pd.Series:
    if "open_time_iso" in candles.columns:
        return pd.to_datetime(candles["open_time_iso"], utc=True)
    return pd.to_datetime(candles["open_time"], unit="ms", utc=True)


def event_trace(events: list[TradeEvent], event_type: str) -> go.Scatter:
    style = EVENT_STYLES[event_type]
    matching_events = [event for event in events if event.event_type == event_type]
    customdata = [
        [event.quantity, event.margin_amount, event.detail]
        for event in matching_events
    ]
    return go.Scatter(
        x=[event.timestamp for event in matching_events],
        y=[event.price for event in matching_events],
        mode="markers",
        name=style["name"],
        marker={"color": style["color"], "size": 9, "symbol": style["symbol"], "line": {"width": 1, "color": "#131722"}},
        customdata=customdata,
        hovertemplate=(
            f"<b>{style['name']}</b><br>"
            "Time: %{x}<br>"
            "Fill price: $%{y:,.4f}<br>"
            "Quantity: %{customdata[0]:,.4f}<br>"
            "Margin: $%{customdata[1]:,.2f}<br>"
            "%{customdata[2]}<extra></extra>"
        ),
    )


def build_chart(candles: pd.DataFrame, events: list[TradeEvent], title: str) -> go.Figure:
    timestamps = candle_timestamps(candles)
    volume_colors = [
        "#089981" if close >= open_price else "#f23645"
        for open_price, close in zip(candles["open"], candles["close"])
    ]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.84, 0.16],
    )
    figure.add_trace(
        go.Candlestick(
            x=timestamps,
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="HYPE 1h",
            increasing_line_color="#089981",
            increasing_fillcolor="#089981",
            decreasing_line_color="#f23645",
            decreasing_fillcolor="#f23645",
        ),
        row=1,
        col=1,
    )
    for event_type in EVENT_STYLES:
        figure.add_trace(event_trace(events, event_type), row=1, col=1)

    figure.add_trace(
        go.Bar(
            x=timestamps,
            y=candles["volume"],
            name="Volume",
            marker={"color": volume_colors, "opacity": 0.55},
            hovertemplate="Time: %{x}<br>Volume: %{y:,.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16, "color": "#f1f3f6"}},
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font={"family": "Arial, sans-serif", "color": "#d1d4dc", "size": 12},
        height=960,
        hovermode="x unified",
        dragmode="pan",
        hoverlabel={"bgcolor": "#1e222d", "bordercolor": "#363c4e", "font": {"color": "#d1d4dc"}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
        margin={"l": 18, "r": 76, "t": 90, "b": 30},
        bargap=0.05,
    )
    figure.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.055,
        showgrid=True,
        gridcolor="#242832",
        gridwidth=1,
        showline=True,
        linecolor="#363c4e",
        tickfont={"color": "#9598a1"},
        showspikes=True,
        spikecolor="#6a6f7b",
        spikethickness=1,
        spikemode="across",
        spikesnap="cursor",
        row=2,
        col=1,
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor="#242832",
        gridwidth=1,
        showline=True,
        linecolor="#363c4e",
        showspikes=True,
        spikecolor="#6a6f7b",
        spikethickness=1,
        spikemode="across",
        spikesnap="cursor",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="#242832",
        gridwidth=1,
        showline=True,
        linecolor="#363c4e",
        tickprefix="$",
        tickfont={"color": "#9598a1"},
        fixedrange=False,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="#242832",
        gridwidth=1,
        showline=True,
        linecolor="#363c4e",
        tickfont={"color": "#9598a1"},
        fixedrange=False,
        row=2,
        col=1,
    )
    return figure


def main() -> None:
    args = parse_args()
    candles = filter_candles(load_candles(update_csv=not args.no_update), args.start, args.end)
    params = StrategyParams(
        initial_drop_pct=args.initial_drop_pct,
        recurring_drop_pct=args.recurring_drop_pct,
        first_buy_fraction=parse_fraction(args.first_buy_fraction),
        recurring_buy_fraction=parse_fraction(args.recurring_buy_fraction),
        take_profit_margin_pct=args.take_profit_margin_pct,
    )
    run = simulate_strategy(
        candles,
        params,
        starting_cash=args.starting_cash,
        leverage=args.leverage,
        max_cycle_margin_pct=args.max_cycle_margin_pct,
        reduced_target_margin_threshold_pct=args.reduced_target_margin_threshold_pct,
        reduced_target_multiplier=args.reduced_target_multiplier,
        dip_lookback_hours=args.dip_lookback_hours,
        max_additional_buys=args.max_additional_buys,
        capture_events=True,
    )
    result = run.result
    title = (
        "HYPE scalping backtest | "
        f"initial dip {params.initial_drop_pct:g}% | recurring dip {params.recurring_drop_pct:g}% | "
        f"TP {params.take_profit_margin_pct:g}% on margin | profit ${result.profit:,.2f}"
    )
    figure = build_chart(candles, run.events, title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output,
        include_plotlyjs=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    event_summary = ", ".join(
        f"{EVENT_STYLES[event_type]['name']}={sum(event.event_type == event_type for event in run.events)}"
        for event_type in EVENT_STYLES
    )
    print(f"Chart written to {args.output.resolve()}")
    print(f"Profit: ${result.profit:,.2f} ({result.profit_pct:.2f}%), final equity: ${result.final_equity:,.2f}")
    print(event_summary)

    if not args.no_browser:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
