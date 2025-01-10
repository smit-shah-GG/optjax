import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Tuple, Optional
import asyncio
import aiohttp
import time
import pandas_ta as ta
import os
from pathlib import Path


class BinanceDataFetcher:
    def __init__(
        self, symbol: str = "BTC/USDT:USDT", data_dir: str = "ohlcvf_plus_indicators"
    ):
        self.exchange = ccxt.binance(
            {"enableRateLimit": True, "options": {"defaultType": "future"}}
        )
        self.symbol = symbol
        self.binance_symbol = symbol.split(":")[0].replace("/", "").lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    async def fetch_funding_rates(self, start_time: int, end_time: int) -> pd.DataFrame:
        """Fetch funding rate history using Binance API"""
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        async with aiohttp.ClientSession() as session:
            funding_rates = []
            current_time = start_time

            while current_time < end_time:
                params = {
                    "symbol": self.binance_symbol.upper(),
                    "startTime": current_time,
                    "endTime": min(current_time + 1000 * 480 * 60000, end_time),
                    "limit": 1000,
                }

                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        await asyncio.sleep(60)
                        continue

                    data = await response.json()
                    if not data:
                        break

                    funding_rates.extend(data)
                    current_time = data[-1]["fundingTime"] + 1
                    await asyncio.sleep(0.1)

            df_funding = pd.DataFrame(funding_rates)
            if not df_funding.empty:
                df_funding["fundingTime"] = pd.to_datetime(
                    df_funding["fundingTime"], unit="ms"
                )
                df_funding["fundingRate"] = df_funding["fundingRate"].astype(float)
                df_funding.set_index("fundingTime", inplace=True)
            return df_funding

    def fetch_ohlcv(
        self,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        limit: int = 25000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        since = int(pd.Timestamp(start_date).timestamp() * 1000) if start_date else None
        ohlcv_data = []

        while len(ohlcv_data) < limit:
            try:
                temp = self.exchange.fetch_ohlcv(
                    symbol=self.symbol, timeframe=timeframe, since=since, limit=1000
                )
                if not temp:
                    break

                ohlcv_data.extend(temp)
                since = temp[-1][0] + 1
                time.sleep(0.1)
                print(f"Got data for timestamp: {since}")

            except ccxt.RequestTimeout:
                time.sleep(30)
                continue
            except Exception as e:
                print(f"Error fetching OHLCV: {e}")
                break

        df = pd.DataFrame(
            ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate each indicator separately and return as dict of DataFrames"""
        indicators = {}
        
        # Save original index
        original_index = df.index
        
        # RSI
        try:
            rsi = ta.rsi(df['close'])
            indicators['rsi'] = pd.DataFrame({'rsi': rsi}, index=original_index)
            print("RSI calculated")
        except Exception as e:
            print(f"Error calculating RSI: {e}")

        # ATR
        try:
            atr = ta.atr(high=df['high'], low=df['low'], close=df['close'])
            indicators['atr'] = pd.DataFrame({'atr': atr}, index=original_index)
            print("ATR calculated")
        except Exception as e:
            print(f"Error calculating ATR: {e}")

        # EMAs and Volume MAs
        for period in [7, 25, 99]:
            try:
                ema = ta.ema(df['close'], length=period)
                vol_ma = df['volume'].rolling(window=period).mean()
                
                indicators[f'ema_{period}'] = pd.DataFrame(
                    {f'ema_{period}': ema}, index=original_index
                )
                indicators[f'volume_ma_{period}'] = pd.DataFrame(
                    {f'volume_ma_{period}': vol_ma}, index=original_index
                )
                print(f"EMA and Volume MA {period} calculated")
            except Exception as e:
                print(f"Error calculating EMA/Volume MA {period}: {e}")

        # Bollinger Bands
        try:
            bb = ta.bbands(df['close'], length=20, std=2)
            bb.index = original_index
            indicators['bbands'] = bb
            print("Bollinger Bands calculated")
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")

        # MACD
        try:
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            macd.index = original_index
            indicators['macd'] = macd
            print("MACD calculated")
        except Exception as e:
            print(f"Error calculating MACD: {e}")

        # Momentum
        try:
            mom = ta.mom(df['close'], length=14)
            roc = ta.roc(df['close'], length=14)
            indicators['momentum'] = pd.DataFrame({
                'mom': mom,
                'roc': roc
            }, index=original_index)
            print("Momentum indicators calculated")
        except Exception as e:
            print(f"Error calculating Momentum indicators: {e}")

        # Volatility
        try:
            vol = df['close'].pct_change().rolling(window=30).std() * np.sqrt(365)
            indicators['volatility'] = pd.DataFrame({
                'historical_volatility': vol
            }, index=original_index)
            print("Volatility calculated")
        except Exception as e:
            print(f"Error calculating Volatility: {e}")

        return indicators

    async def fetch_and_save_all(
        self, timeframe: str = "1h", start_date: str = "2020-01-01", limit: int = 25000
    ) -> None:
        """Fetch all data and save individual components"""
        # Fetch and save OHLCV
        print("Fetching OHLCV data...")
        df_ohlcv = self.fetch_ohlcv(timeframe, start_date, limit)
        df_ohlcv.to_csv(self.data_dir / "ohlcv.csv")
        print("OHLCV data saved")

        # Fetch and save funding rates
        print("Fetching funding rates...")
        start_time = int(df_ohlcv.index[0].timestamp() * 1000)
        end_time = int(df_ohlcv.index[-1].timestamp() * 1000)
        df_funding = await self.fetch_funding_rates(start_time, end_time)
        df_funding.to_csv(self.data_dir / "funding.csv")
        print("Funding rates saved")

        # Calculate and save indicators
        print("Calculating indicators...")
        indicators = self.calculate_indicators(df_ohlcv)
        for name, df in indicators.items():
            df.to_csv(self.data_dir / f"{name}.csv")
            print(f"{name} indicator saved")


def merge_and_save_parquet(
    data_dir: str = "ohlcvf_plus_indicators", output_file: str = "dataset.parquet"
):
    """Merge all CSV files and save as parquet"""
    data_dir = Path(data_dir)

    # Load OHLCV as base
    print("Loading OHLCV data...")
    df = pd.read_csv(data_dir / "ohlcv.csv")
    print(f"OHLCV shape: {df.shape}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Add funding rates
    print("Adding funding rates...")
    try:
        funding = pd.read_csv(data_dir / "funding.csv")
        funding["fundingTime"] = pd.to_datetime(funding["fundingTime"])
        funding.set_index("fundingTime", inplace=True)
        # Resample funding rates to match OHLCV timeframe
        funding = funding.resample("1H").ffill()
        df = df.join(funding)
        print(f"After adding funding rates shape: {df.shape}")
    except Exception as e:
        print(f"Error adding funding rates: {e}")

    # Add all indicators
    for file in data_dir.glob("*.csv"):
        if file.stem not in ["ohlcv", "funding"]:
            print(f"Adding {file.stem}...")
            try:
                temp_df = pd.read_csv(file)
                print(f"{file.stem} initial shape: {temp_df.shape}")

                # Remove index column if it exists
                if "Unnamed: 0" in temp_df.columns:
                    temp_df = temp_df.drop("Unnamed: 0", axis=1)

                # Create new index matching the base dataframe
                temp_df.index = df.index[: len(temp_df)]

                # Join with base dataframe
                df = df.join(temp_df)
                print(f"After adding {file.stem} shape: {df.shape}")
            except Exception as e:
                print(f"Error adding {file.stem}: {e}")

    # Save as parquet
    print(f"Saving to {output_file}...")
    pq.write_table(pa.Table.from_pandas(df), output_file, compression="snappy")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Print some basic statistics
    print("\nData ranges:")
    print(f"Start date: {df.index.min()}")
    print(f"End date: {df.index.max()}")
    print(f"Total hours: {len(df)}")


async def main():
    # Create fetcher
    fetcher = BinanceDataFetcher(
        symbol="BTC/USDT:USDT", data_dir="ohlcvf_plus_indicators"
    )

    # Fetch and save all individual components
    await fetcher.fetch_and_save_all(timeframe="1h", start_date="2020-01-01")

    # Merge and save as parquet
    merge_and_save_parquet(
        data_dir="ohlcvf_plus_indicators", output_file="ohlcvf_plus_indicators/btc_futures_data.parquet"
    )


if __name__ == "__main__":
    asyncio.run(main())
