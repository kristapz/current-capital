import os
import yfinance as yf
from google.cloud import bigquery
from datetime import datetime, timedelta
import logging
import pytz
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'backwards_testing'
table_id = 'main'
temp_table_id = 'temp_predictions'
client_bq = bigquery.Client(project=project_id)

def get_pacific_time():
    utc_time = datetime.utcnow().replace(tzinfo=pytz.utc)
    pacific = pytz.timezone('America/Los_Angeles')
    return utc_time.astimezone(pacific)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        google_exceptions.ServerError,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        requests.exceptions.RequestException
    ))
)
def fetch_articles_for_update():
    buffer_time = get_pacific_time() - timedelta(hours=3)  # Adjust this time as needed
    query = f"""
    SELECT id, stock_prediction,
        SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) AS parsed_date
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE 'true' IN UNNEST(updated) OR 'truer' IN UNNEST(updated)
      AND SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) < DATETIME(TIMESTAMP('{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}'))
      AND DATETIME_DIFF(CURRENT_DATETIME(), SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date), MINUTE) > 90
    ORDER BY parsed_date DESC
    LIMIT 1
    """
    try:
        results = list(client_bq.query(query).result())
        logging.info(f"Fetched {len(results)} oldest articles with updated=true or truer, older than 90 minutes, and outside the streaming buffer")
        return results
    except Exception as e:
        logging.error(f"Error fetching updated articles: {e}")
        raise

def fetch_hourly_stock_prices(ticker, start_date):
    logging.info(f"Fetching hourly stock prices for {ticker}")
    stock = yf.Ticker(ticker)
    end_date = datetime.now(pytz.UTC)
    try:
        hist = stock.history(start=start_date, end=end_date, interval="1h")
        if hist.empty:
            logging.warning(f"No historical data available for ticker {ticker}")
            return None
        return hist['Close'].tolist()
    except Exception as e:
        logging.error(f"Error fetching stock prices for {ticker}: {e}")
        return None

def get_price_24_hours_ago(ticker):
    logging.info(f"Fetching price 24 hours ago for {ticker}")
    end_time = datetime.now(pytz.UTC)
    time_24hrs_ago = end_time - timedelta(hours=24)
    stock = yf.Ticker(ticker)
    hist_daily = stock.history(start=time_24hrs_ago - timedelta(days=1), end=end_time, interval="1d")

    if not hist_daily.empty:
        target_date = time_24hrs_ago.date()
        price_24hrs_ago = hist_daily.loc[hist_daily.index.date == target_date, 'Close']
        if not price_24hrs_ago.empty:
            return price_24hrs_ago.iloc[0]
        else:
            logging.warning(f"No data available for {ticker} on {target_date}")
            return None
    else:
        logging.warning(f"No historical data available for ticker {ticker}")
        return None

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        google_exceptions.ServerError,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        requests.exceptions.RequestException
    ))
)
def update_stock_prices(article_id, stock_predictions, article_date):
    logging.info(f"Updating stock prices for article {article_id}")
    fetch_query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE id = @article_id AND ('true' IN UNNEST(updated) OR 'truer' IN UNNEST(updated))
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("article_id", "INT64", article_id)])
    try:
        existing_rows = list(client_bq.query(fetch_query, job_config=job_config).result())
    except Exception as e:
        logging.error(f"Error querying article {article_id}: {e}")
        return

    if not existing_rows:
        logging.warning(f"No row found with id {article_id} and updated = true or truer")
        return

    existing_row = dict(existing_rows[0])
    updated_stock_predictions = []

    for sp in existing_row['stock_prediction']:
        # Fetch the hourly prices
        hourly_prices = fetch_hourly_stock_prices(sp['ticker'], article_date)
        # Fetch the 24-hour price
        price_24hrs_ago = get_price_24_hours_ago(sp['ticker'])

        if hourly_prices:
            sp.update({
                'stock_price_1hr': hourly_prices[0] if len(hourly_prices) > 0 else None,
                'stock_price_2hrs': hourly_prices[1] if len(hourly_prices) > 1 else None,
                'stock_price_3hrs': hourly_prices[2] if len(hourly_prices) > 2 else None,
                'stock_price_5hrs': hourly_prices[4] if len(hourly_prices) > 4 else None,
                'stock_price_10hrs': hourly_prices[9] if len(hourly_prices) > 9 else None,
                'stock_price_24hrs': price_24hrs_ago
            })
        updated_stock_predictions.append(sp)

    existing_row['stock_prediction'] = updated_stock_predictions

    # Check if all 24-hour prices are filled
    all_filled = all(sp.get('stock_price_24hrs') is not None for sp in updated_stock_predictions)
    if all_filled:
        existing_row['updated'] = ["truest"]
    else:
        existing_row['updated'] = ["true"]

    # Use REPLACE instead of INSERT
    job_config = bigquery.LoadJobConfig(
        schema=client_bq.get_table(temp_table_ref).schema,
        write_disposition="WRITE_TRUNCATE"
    )
    try:
        job = client_bq.load_table_from_json([existing_row], temp_table_ref, job_config=job_config)
        job.result()  # Wait for the job to complete

        if job.errors:
            logging.error(f"Errors occurred while replacing rows: {job.errors}")
        else:
            logging.info(f"Updated row replaced in temp table for article {article_id}")
    except Exception as e:
        logging.error(f"Error loading data to temp table for article {article_id}: {e}")

    # Verify that the predictions were updated correctly
    try:
        verification_query = f"""
        SELECT stock_prediction
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE id = @article_id
        """
        verification_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("article_id", "INT64", article_id)])
        verification_rows = list(client_bq.query(verification_query, job_config=verification_config).result())

        if not verification_rows:
            logging.warning(f"No row found with id {article_id} during verification")
        else:
            verification_row = dict(verification_rows[0])
            logging.info(f"Verified stock predictions for article {article_id}: {verification_row['stock_prediction']}")
    except Exception as e:
        logging.error(f"Error verifying stock predictions for article {article_id}: {e}")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((google_exceptions.ServerError, google_exceptions.TooManyRequests, google_exceptions.ServiceUnavailable, requests.exceptions.RequestException))
)
def setup_temp_table():
    global temp_table_ref
    temp_table_ref = client_bq.dataset(dataset_id).table(temp_table_id)
    try:
        temp_table = bigquery.Table(temp_table_ref, schema=client_bq.get_table(f"{project_id}.{dataset_id}.{table_id}").schema)
        client_bq.create_table(temp_table, exists_ok=True)
        logging.info(f"Temporary table {temp_table_id} set up")
    except google_exceptions.NotFound:
        temp_table = bigquery.Table(temp_table_ref, schema=client_bq.get_table(f"{project_id}.{dataset_id}.{table_id}").schema)
        client_bq.create_table(temp_table)
        logging.info(f"Temporary table {temp_table_id} created")
    except Exception as e:
        logging.error(f"Error setting up temporary table: {e}")
        raise

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        google_exceptions.ServerError,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        requests.exceptions.RequestException
    ))
)
def merge_temp_table():
    buffer_time = get_pacific_time() - timedelta(hours=3)  # Adjust this time as needed
    merge_query = f"""
    MERGE `{project_id}.{dataset_id}.{table_id}` T
    USING `{project_id}.{dataset_id}.{temp_table_id}` S
    ON T.id = S.id 
    AND ('true' IN UNNEST(T.updated) OR 'truer' IN UNNEST(T.updated))
    AND ('true' IN UNNEST(S.updated) OR 'truest' IN UNNEST(S.updated))
    AND SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', T.date) < DATETIME(TIMESTAMP('{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}'))
    WHEN MATCHED THEN
      UPDATE SET 
        T.stock_prediction = S.stock_prediction,
        T.updated = S.updated
    """
    try:
        client_bq.query(merge_query).result()
        logging.info("Merged temporary table into main table")
    except Exception as e:
        logging.error(f"Error merging temporary table: {e}")

def delete_temp_table():
    try:
        client_bq.delete_table(temp_table_ref, not_found_ok=True)
        logging.info(f"Deleted temporary table {temp_table_id}")
    except Exception as e:
        logging.error(f"Error deleting temporary table: {e}")

def update_all_stock_prices():
    try:
        setup_temp_table()
        articles = fetch_articles_for_update()
        for article in articles:
            if article.stock_prediction:
                update_stock_prices(article.id, article.stock_prediction, article.parsed_date)
                merge_temp_table()
                delete_temp_table()
                setup_temp_table()
            else:
                logging.warning(f"No stock predictions found for article {article.id}")

        logging.info(f"Processed {len(articles)} articles")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        delete_temp_table()

if __name__ == "__main__":
    update_all_stock_prices()


