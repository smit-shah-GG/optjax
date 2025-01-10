import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta


def fetch_perp_data(
    symbol="BTC/USDT:USDT",  # Perpetual futures pair
    timeframe="1h",  # Timeframe
    start_date=None,  # e.g., '2020-01-01'
    limit=25000,  # Matches your current shape
):
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},  # Important: specify futures market
        }
    )

    # Convert start_date to timestamp if provided
    since = int(pd.Timestamp(start_date).timestamp() * 1000) if start_date else None

    ohlcv = []
    while len(ohlcv) < limit:
        temp = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=1000,  # Binance limit per request
        )
        if not temp:
            break

        ohlcv.extend(temp)
        since = temp[-1][0] + 1  # Next timestamp
        print(f"Got data for {since}")

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return df


df_data = fetch_perp_data(start_date="2020-01-01")
df_data.to_csv(f"btcustd_price.csv")
