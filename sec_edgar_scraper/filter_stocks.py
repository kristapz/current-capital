import os
import re
import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging
import time
from typing import List, Dict, Optional
from ratelimit import limits, sleep_and_retry
from collections import deque
from datetime import datetime, timedelta

# =======================
# 1. Logging Configuration
# =======================
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce log verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("investing_criteria_async.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =======================
# 2. Configuration
# =======================
# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Google Cloud Project and Dataset information
project_id = ''
dataset_id = 'stock_datasets'
source_table_id = 'stocksbio'
destination_table_id = 'stocksenhancedjan2'
source_full_table_id = f"{project_id}.{dataset_id}.{source_table_id}"
destination_full_table_id = f"{project_id}.{dataset_id}.{destination_table_id}"

# Initialize BigQuery client
client = bigquery.Client(project=project_id)

# Rate limiting configuration
CALLS_PER_MINUTE = 2500  # Adjust based on API limits
RATE_LIMIT_PERIOD = 15  # seconds

# Batch size
BATCH_SIZE = 100

# Number of concurrent requests
CONCURRENT_REQUESTS = 5

# Cache duration in seconds
CACHE_DURATION = 3600  # 1 hour


# =======================
# 3. Caching Mechanism
# =======================
class StockDataCache:
    def __init__(self):
        self.cache = {}
        self.sp500_data = None
        self.sp500_timestamp = None

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < CACHE_DURATION:
                return data
            del self.cache[key]
        return None

    def set(self, key: str, value: Dict):
        self.cache[key] = (value, datetime.now().timestamp())


# Initialize cache
stock_cache = StockDataCache()


# =======================
# 4. Rate Limiting Decorator
# =======================
@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=RATE_LIMIT_PERIOD)
async def limited_fetch(session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
    async with session.get(url) as response:
        if response.status == 200:
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                logger.error(f"Invalid JSON response for URL: {url}")
                return None
        elif response.status == 429:
            logger.warning(f"Rate limit exceeded for URL: {url}")
            raise Exception("Too Many Requests")
        else:
            logger.error(f"Failed to fetch URL: {url} with status {response.status}")
            return None


# =======================
# 5. Utility Functions
# =======================

def convert_to_serializable(data: Dict) -> Dict:
    """Convert numpy data types to native Python types for JSON serialization."""
    for key, value in data.items():
        if isinstance(value, np.generic):
            data[key] = value.item()
    return data


def calculate_relative_strength(stock_hist: pd.DataFrame, sp500_hist: pd.DataFrame) -> Optional[float]:
    """Calculates the Relative Strength (RS) of the stock compared to the S&P 500."""
    if len(stock_hist) < 14 or len(sp500_hist) < 14:
        return None
    stock_returns = stock_hist['Close'].pct_change().fillna(0).iloc[-14:]
    sp500_returns = sp500_hist['Close'].pct_change().fillna(0).iloc[-14:]
    stock_cum = (1 + stock_returns).prod()
    sp500_cum = (1 + sp500_returns).prod()
    return stock_cum / sp500_cum


def analyze_volume(stock_hist: pd.DataFrame) -> Optional[tuple]:
    """Analyzes the latest trading volume compared to the average over the last 50 days."""
    if 'Volume' not in stock_hist.columns or len(stock_hist) < 51:
        return None, None, None
    latest_volume = stock_hist['Volume'].iloc[-1]
    avg_volume = stock_hist['Volume'].iloc[:-1].rolling(window=50).mean().iloc[-1]
    percent_increase = ((latest_volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else None
    return latest_volume, avg_volume, percent_increase


def calculate_rsi(stock_hist: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calculates the Relative Strength Index (RSI) for a given stock."""
    delta = stock_hist['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else None


def calculate_relative_stock_index(stock_hist: pd.DataFrame, benchmark_hist: pd.DataFrame, days: int = 3) -> Optional[
    float]:
    """Calculates the Relative Stock Index over the specified number of days."""
    if len(stock_hist) < days or len(benchmark_hist) < days:
        return None
    stock_returns = stock_hist['Close'].pct_change().fillna(0).iloc[-days:]
    benchmark_returns = benchmark_hist['Close'].pct_change().fillna(0).iloc[-days:]
    stock_avg_return = stock_returns.mean()
    benchmark_avg_return = benchmark_returns.mean()
    if benchmark_avg_return == 0:
        return None
    return stock_avg_return / benchmark_avg_return


def process_stock_data(ticker: str, stock_data: Dict, sp500_hist: pd.DataFrame) -> Optional[Dict]:
    """Processes individual stock data and prepares it for BigQuery insertion."""
    try:
        hist_data = stock_data['history']
        info = stock_data['info']

        if hist_data.empty:
            logger.error(f"{ticker}: No historical data found.")
            return None

        rs = calculate_relative_strength(hist_data, sp500_hist)
        rsi = calculate_rsi(hist_data)
        latest_volume, avg_volume, percent_volume_increase = analyze_volume(hist_data)
        relative_stock_index = calculate_relative_stock_index(hist_data, sp500_hist, days=3)
        current_price = hist_data['Close'].iloc[-1] if 'Close' in hist_data.columns else None

        data = {
            "name": info.get('shortName'),
            "ticker": ticker,
            "sector": info.get('sector'),
            "long_business_summary": info.get('longBusinessSummary'),
            "embeddings": None,
            "embeddings_large": None,
            "openai_embeddings": None,
            "embeddings_large_instruct": None,
            "relative_strength": rs,
            "rsi_14_days": rsi,
            "eps": info.get('trailingEps'),
            "shares_outstanding": info.get('sharesOutstanding'),
            "institutional_ownership": info.get('heldPercentInstitutions') * 100 if info.get(
                'heldPercentInstitutions') is not None else None,
            "latest_volume": latest_volume,
            "average_volume_50_days": avg_volume,
            "percentage_volume_increase": percent_volume_increase,
            "pe_ratio": info.get('trailingPE'),
            "relative_stock_index_3_days": relative_stock_index,
            "current_price": current_price
        }

        # Filter out stocks with insufficient data
        if data["shares_outstanding"] is None or data["institutional_ownership"] is None:
            logger.info(f"{ticker}: Insufficient data for shares_outstanding or institutional_ownership.")
            return None

        # Ensure at least 5 fields are non-null
        non_null_count = sum(1 for key in [
            'relative_strength', 'rsi_14_days', 'eps', 'shares_outstanding',
            'institutional_ownership', 'latest_volume', 'average_volume_50_days',
            'percentage_volume_increase', 'pe_ratio', 'relative_stock_index_3_days',
            'current_price'
        ] if data.get(key) is not None)

        if non_null_count >= 5:
            serializable_data = convert_to_serializable(data)
            return serializable_data
        else:
            logger.info(f"{ticker}: Not enough non-null fields.")
            return None

    except Exception as e:
        logger.error(f"Error processing data for {ticker}: {e}")
        return None


# =======================
# 6. BigQuery Functions
# =======================

def insert_to_bigquery(data: Dict):
    """Inserts processed stock data into the BigQuery table."""
    try:
        table_ref = client.dataset(dataset_id).table(destination_table_id)
        errors = client.insert_rows_json(table_ref, [data])
        if errors:
            logger.error(f"Error inserting data into BigQuery for {data['ticker']}: {errors}")
            print(f"Error inserting data into BigQuery for {data['ticker']}: {errors}")
        else:
            logger.info(f"Inserted {data['ticker']} into BigQuery successfully.")
            print(f"Inserted {data['ticker']}")
    except Exception as e:
        logger.error(f"Error inserting into BigQuery for {data['ticker']}: {e}")
        print(f"Error inserting into BigQuery for {data['ticker']}: {e}")


def check_and_create_table():
    """Checks if the destination table exists, and creates it if it does not."""
    try:
        table_ref = client.dataset(dataset_id).table(destination_table_id)
        client.get_table(table_ref)
        logger.info(f"Table {destination_full_table_id} already exists.")
    except NotFound:
        schema = [
            bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("long_business_summary", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embeddings", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embeddings_large", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("openai_embeddings", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embeddings_large_instruct", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("relative_strength", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("rsi_14_days", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("eps", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("shares_outstanding", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("institutional_ownership", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("latest_volume", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("average_volume_50_days", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("percentage_volume_increase", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("pe_ratio", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("relative_stock_index_3_days", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("current_price", "FLOAT", mode="NULLABLE")
        ]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        logger.info(f"Created table {destination_full_table_id}.")


# =======================
# 7. Asynchronous Processing
# =======================

async def fetch_sp500_data(session: aiohttp.ClientSession) -> Optional[pd.DataFrame]:
    """Fetches S&P 500 historical data."""
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = await asyncio.to_thread(sp500.history, period='1mo')
        if hist.empty:
            logger.warning("S&P 500 (^GSPC) data is empty. Trying SPY.")
            spy = yf.Ticker("SPY")
            hist = await asyncio.to_thread(spy.history, period='1mo')
            if hist.empty:
                logger.error("Insufficient data for both S&P 500 (^GSPC) and SPY.")
                return None
        return hist
    except Exception as e:
        logger.error(f"Error fetching S&P 500 data: {e}")
        return None


async def fetch_stock_data(session: aiohttp.ClientSession, ticker: str) -> Optional[Dict]:
    """Fetches stock data for a given ticker."""
    # Check cache first
    cached_data = stock_cache.get(ticker)
    if cached_data:
        return cached_data

    try:
        stock = yf.Ticker(ticker)
        hist = await asyncio.to_thread(stock.history, period='3mo')
        info = await asyncio.to_thread(lambda: stock.info)

        if hist.empty:
            logger.error(f"{ticker}: possibly delisted; no price data found (period=3mo)")
            return None

        data = {
            'history': hist,
            'info': info
        }

        # Cache the result
        stock_cache.set(ticker, data)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


async def process_and_insert(session: aiohttp.ClientSession, ticker: str, sp500_hist: pd.DataFrame):
    """Processes a single stock and inserts it into BigQuery."""
    data = await fetch_stock_data(session, ticker)
    if data:
        processed = process_stock_data(ticker, data, sp500_hist)
        if processed:
            insert_to_bigquery(processed)


async def process_batch(session: aiohttp.ClientSession, tickers: List[str], sp500_hist: pd.DataFrame):
    """Processes a batch of tickers concurrently."""
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def sem_task(ticker):
        async with semaphore:
            await process_and_insert(session, ticker, sp500_hist)

    for ticker in tickers:
        tasks.append(asyncio.create_task(sem_task(ticker)))

    await asyncio.gather(*tasks, return_exceptions=True)


async def fetch_and_process_stocks_async():
    """Main async function to fetch and process stocks."""
    check_and_create_table()

    async with aiohttp.ClientSession() as session:
        # Fetch S&P 500 data once
        sp500_hist = await fetch_sp500_data(session)
        if sp500_hist is None:
            logger.error("Failed to fetch benchmark data. Exiting.")
            return
        logger.info("Successfully fetched S&P 500 data.")

        offset = 0
        while True:
            # Fetch batch of tickers from BigQuery
            query = f"""
            SELECT ticker FROM `{source_full_table_id}`
            ORDER BY ticker
            LIMIT {BATCH_SIZE} OFFSET {offset}
            """
            try:
                query_job = client.query(query)
                results = await asyncio.to_thread(query_job.result)
                tickers = [row['ticker'] for row in results]
            except Exception as e:
                logger.error(f"Error fetching tickers from BigQuery: {e}")
                break

            if not tickers:
                logger.info("No more stocks to process.")
                break

            # Filter valid tickers
            valid_tickers = []
            excluded_tickers = []
            for ticker in tickers:
                cleaned_ticker = ticker.strip().upper()
                if re.match(r'^[A-Z0-9.]+$', cleaned_ticker):
                    valid_tickers.append(cleaned_ticker)
                else:
                    excluded_tickers.append(ticker)

            if excluded_tickers:
                logger.info(f"Excluded tickers due to special characters: {excluded_tickers}")

            if not valid_tickers:
                offset += BATCH_SIZE
                continue

            # Process the batch
            await process_batch(session, valid_tickers, sp500_hist)

            logger.info(f"Completed processing batch starting at offset {offset}.")
            offset += BATCH_SIZE

            # Optional: Introduce a longer pause after each batch to ensure rate limits are respected
            logger.info(f"Pausing for 60 seconds before processing the next batch.")
            print(f"Pausing for 60 seconds before processing the next batch.")
            await asyncio.sleep(30)

    logger.info("All batches have been processed.")
    print("All batches have been processed.")


# =======================
# 6. Main Execution
# =======================

def main():
    asyncio.run(fetch_and_process_stocks_async())


if __name__ == "__main__":
    main()
