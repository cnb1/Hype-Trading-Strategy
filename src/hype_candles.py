#!/usr/bin/env python3
"""
Fetches HYPE candlestick history from the Hyperliquid API and stores it in CSV files.

- If the CSV doesn't exist: pulls the entire available price history.
- If the CSV exists: pulls only new candles since the last stored timestamp and appends them.

Adds two derived columns:
- direction: 1 if close > open, 0 if close == open, -1 if close < open
- pct_change: percentage change from open to close, i.e. (close - open) / open * 100

Requires: requests, pandas
    pip install requests pandas
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

API_URL = "https://api.hyperliquid.xyz/info"
COIN = "HYPE"
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
HYPE_1H_INTERVAL = "1h"
HYPE_4H_INTERVAL = "4h"
HYPE_1H_CSV_FILE = os.path.join(RESOURCES_DIR, "hype_1h_candles.csv")
HYPE_4H_CSV_FILE = os.path.join(RESOURCES_DIR, "hype_4h_candles.csv")
HYPE_1H_INTERVAL_MS = 1 * 60 * 60 * 1000
HYPE_4H_INTERVAL_MS = 4 * 60 * 60 * 1000
# Hyperliquid caps each candleSnapshot response at 5000 candles
MAX_CANDLES_PER_REQUEST = 5000


def fetch_candles(start_time_ms: int, end_time_ms: int, interval: str) -> list:
    """Fetch raw candle data from Hyperliquid between start_time_ms and end_time_ms (inclusive-ish)."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": COIN,
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        },
    }
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_candles(start_time_ms: int, end_time_ms: int, interval: str, interval_ms: int) -> list:
    """
    Paginate through the Hyperliquid API to grab every candle between start and end,
    since a single request can be capped in how many candles it returns.
    """
    all_candles = []
    cursor = start_time_ms

    while cursor < end_time_ms:
        batch = fetch_candles(cursor, end_time_ms, interval)
        if not batch:
            break

        all_candles.extend(batch)

        last_open_time = batch[-1]["t"]
        next_cursor = last_open_time + interval_ms

        # No progress or fewer than a full page returned -> we've reached the end
        if next_cursor <= cursor or len(batch) < MAX_CANDLES_PER_REQUEST:
            cursor = next_cursor
            if len(batch) < MAX_CANDLES_PER_REQUEST:
                break
        else:
            cursor = next_cursor

        time.sleep(0.2)  # be polite to the API

    return all_candles


def candles_to_dataframe(raw_candles: list) -> pd.DataFrame:
    """Convert raw Hyperliquid candle dicts into a clean DataFrame with derived columns."""
    if not raw_candles:
        return pd.DataFrame(
            columns=[
                "open_time", "close_time", "open", "high", "low", "close",
                "volume", "num_trades", "direction", "pct_change",
            ]
        )

    df = pd.DataFrame(raw_candles)
    df = df.rename(columns={
        "t": "open_time",
        "T": "close_time",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "num_trades",
    })

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")

    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)

    df["direction"] = 0
    df.loc[df["close"] > df["open"], "direction"] = 1
    df.loc[df["close"] < df["open"], "direction"] = -1

    df["pct_change"] = (df["close"] - df["open"]) / df["open"] * 100

    df["open_time_iso"] = df["open_time"].apply(
        lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    )
    df["close_time_iso"] = df["close_time"].apply(
        lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    )

    cols = [
        "open_time", "open_time_iso", "close_time", "close_time_iso",
        "open", "high", "low", "close", "volume", "num_trades",
        "direction", "pct_change",
    ]
    return df[cols]


def update_hype_csv_file(csv_file: str, interval: str, interval_ms: int) -> None:
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    now_ms = int(time.time() * 1000)

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        if existing_df.empty:
            start_time_ms = 0
        else:
            last_open_time = int(existing_df["open_time"].max())
            start_time_ms = last_open_time + interval_ms
            if start_time_ms >= now_ms:
                print("CSV is already up to date. No new candles to fetch.")
                return
        print(f"Existing {interval} CSV found. Fetching new candles from {start_time_ms} to {now_ms}...")
    else:
        existing_df = pd.DataFrame()
        start_time_ms = 0  # Hyperliquid will just return data from listing onward
        print(f"No {interval} CSV found. Fetching full HYPE price history...")

    raw_candles = fetch_all_candles(start_time_ms, now_ms, interval, interval_ms)

    if not raw_candles:
        print("No new candle data returned.")
        return

    new_df = candles_to_dataframe(raw_candles)

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    else:
        combined_df = new_df

    combined_df.to_csv(csv_file, index=False)
    print(f"Saved {len(combined_df)} total candles ({len(new_df)} new) to {csv_file}")


def update_hype_csv() -> None:
    """Update the original 1-hour CSV for existing callers."""
    update_hype_csv_file(HYPE_1H_CSV_FILE, HYPE_1H_INTERVAL, HYPE_1H_INTERVAL_MS)


def update_hype_4h_csv() -> None:
    """Update the 4-hour CSV used by the strategy simulator and viewer."""
    update_hype_csv_file(HYPE_4H_CSV_FILE, HYPE_4H_INTERVAL, HYPE_4H_INTERVAL_MS)
