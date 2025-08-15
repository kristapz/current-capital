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
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'backwards_testing'
table_id = 'calls_together'
temp_table_id = 'temp_predictions'
client_bq = bigquery.Client(project=project_id)

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
def fetch_updated_articles():
    buffer_time = datetime.utcnow() - timedelta(hours=3)  # Adjust this time as needed
    query = f"""
    SELECT id, stock_prediction,
        CASE
            WHEN SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) IS NOT NULL THEN SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date)
            WHEN SAFE.PARSE_DATETIME('%Y-%m-%d', date) IS NOT NULL THEN SAFE.PARSE_DATETIME('%Y-%m-%d', date)
            ELSE NULL
        END AS parsed_date
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE 'true' IN UNNEST(updated)
      AND (
          (SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) IS NOT NULL AND SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) < DATETIME(TIMESTAMP('{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}')))
          OR
          (SAFE.PARSE_DATETIME('%Y-%m-%d', date) IS NOT NULL AND SAFE.PARSE_DATETIME('%Y-%m-%d', date) < DATETIME(TIMESTAMP('{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}')))
      )
      AND DATETIME_DIFF(CURRENT_DATETIME(), 
          CASE
              WHEN SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) IS NOT NULL THEN SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date)
              WHEN SAFE.PARSE_DATETIME('%Y-%m-%d', date) IS NOT NULL THEN SAFE.PARSE_DATETIME('%Y-%m-%d', date)
              ELSE NULL
          END, MINUTE) > 90
    ORDER BY parsed_date ASC
    LIMIT 30
    """
    try:
        results = list(client_bq.query(query).result())
        logging.info(f"Fetched {len(results)} oldest articles with updated=true, older than 90 minutes, and outside the streaming buffer")
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
    WHERE id = @article_id AND 'true' IN UNNEST(updated)
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("article_id", "INT64", article_id)])
    try:
        existing_rows = list(client_bq.query(fetch_query, job_config=job_config).result())
    except Exception as e:
        logging.error(f"Error querying article {article_id}: {e}")
        return

    if not existing_rows:
        logging.warning(f"No row found with id {article_id} and updated = true")
        return

    existing_row = dict(existing_rows[0])
    updated_stock_predictions = []

    for sp in existing_row['stock_prediction']:
        hourly_prices = fetch_hourly_stock_prices(sp['ticker'], article_date)
        if hourly_prices:
            sp.update({
                'stock_price_1hr': hourly_prices[0] if len(hourly_prices) > 0 else None,
                'stock_price_2hrs': hourly_prices[1] if len(hourly_prices) > 1 else None,
                'stock_price_3hrs': hourly_prices[2] if len(hourly_prices) > 2 else None,
                'stock_price_5hrs': hourly_prices[4] if len(hourly_prices) > 4 else None,
                'stock_price_10hrs': hourly_prices[9] if len(hourly_prices) > 9 else None,
                'stock_price_24hrs': hourly_prices[23] if len(hourly_prices) > 23 else None,
            })
        else:
            logging.warning(f"Skipping stock {sp['ticker']} due to no historical data.")
        updated_stock_predictions.append(sp)

    existing_row['stock_prediction'] = updated_stock_predictions
    existing_row['updated'] = ["truer"]

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

def setup_temp_table():
    global temp_table_ref
    temp_table_ref = client_bq.dataset(dataset_id).table(temp_table_id)
    try:
        temp_table = bigquery.Table(temp_table_ref, schema=client_bq.get_table(f"{project_id}.{dataset_id}.{table_id}").schema)
        client_bq.create_table(temp_table, exists_ok=True)
        logging.info(f"Temporary table {temp_table_id} set up")
    except Exception as e:
        logging.error(f"Error setting up temporary table: {e}")

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
    buffer_time = datetime.utcnow() - timedelta(hours=3)  # Adjust this time as needed
    merge_query = f"""
    MERGE `{project_id}.{dataset_id}.{table_id}` T
    USING `{project_id}.{dataset_id}.{temp_table_id}` S
    ON T.id = S.id 
    AND 'true' IN UNNEST(T.updated) 
    AND 'truer' IN UNNEST(S.updated)
    AND (
        (SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', T.date) IS NOT NULL AND SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', T.date) < DATETIME(TIMESTAMP('{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}')))
        OR
        (SAFE.PARSE_DATETIME('%Y-%m-%d', T.date) IS NOT NULL AND SAFE.PARSE_DATETIME('%Y-%m-%d', T.date) < DATETIME(TIMESTAMP('{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}')))
    )
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

def update_stock_prices_main():
    try:
        setup_temp_table()
        articles = fetch_updated_articles()
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

def main():
    max_retries = 5
    retry_count = 0

    while True:
        try:
            # Run the stock price update process
            update_stock_prices_main()

            logging.info("Sleeping for 5 minutes before next cycle...")
            time.sleep(300)  # Sleep for 5 minutes
            retry_count = 0  # Reset retry count after successful iteration

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Max retries ({max_retries}) reached. Exiting.")
                break
            wait_time = 60 * retry_count  # Increase wait time with each retry
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

if __name__ == "__main__":
    main()
