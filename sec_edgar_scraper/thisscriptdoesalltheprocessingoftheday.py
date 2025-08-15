# iterates through the filings in the filtered dataset from investingcriteria.py and runs 8k and 10k analysis on the stocks
import os
import subprocess
import sys
import logging
import argparse
from datetime import datetime, timezone
from google.cloud import bigquery

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def run_scrapy_spider(spider_name, ticker, max_documents=None, arg_name='tickers'):
    """
    Runs a Scrapy spider with the given name and ticker, optionally limiting to max_documents.

    Args:
        spider_name (str): The name of the Scrapy spider to run.
        ticker (str): The stock ticker symbol.
        max_documents (int, optional): Maximum number of filings to process.
        arg_name (str): The name of the argument to pass to the spider ('tickers' or 'ticker').
    """
    try:
        # Construct the Scrapy command with dynamic argument name
        cmd = [
            'scrapy',
            'crawl',
            spider_name,
            '-a',
            f'{arg_name}={ticker}'
        ]

        if max_documents is not None:
            cmd.extend(['-a', f'max_documents={max_documents}'])

        logging.info(f"Starting Scrapy spider '{spider_name}' for ticker: {ticker} with args: {cmd}")

        # Determine the project directory (assumes this script is in the project root)
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Execute the Scrapy spider
        subprocess.check_call(cmd, cwd=project_dir)

        logging.info(f"Scrapy spider '{spider_name}' completed successfully for ticker: {ticker}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Scrapy spider '{spider_name}' failed for ticker {ticker}. Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Scrapy is not installed or not found in the PATH.")
        sys.exit(1)


def run_openai_processing(script_relative_path, ticker):
    """
    Runs an OpenAI processing script with the given relative path and ticker.

    Args:
        script_relative_path (str): The relative path to the OpenAI processing script.
        ticker (str): The stock ticker symbol.
    """
    try:
        logging.info(f"Starting OpenAI processing script '{script_relative_path}' for ticker: {ticker}")

        # Determine the absolute path to the script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_relative_path)

        if not os.path.isfile(script_path):
            logging.error(f"OpenAI processing script '{script_relative_path}' not found at '{script_path}'.")
            sys.exit(1)

        # Execute the OpenAI processing script with the ticker as an argument
        subprocess.check_call(['python', script_path, '-ticker', ticker])

        logging.info(f"OpenAI processing script '{script_relative_path}' completed successfully for ticker: {ticker}")

    except subprocess.CalledProcessError as e:
        logging.error(f"OpenAI processing script '{script_relative_path}' failed for ticker {ticker}. Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Python is not installed or not found in the PATH.")
        sys.exit(1)


def query_tickers_from_bigquery():
    """
    Queries the 'filtered_stocks' BigQuery table, returning only tickers from
    specific US exchanges (yfinance style) AND skipping any stock whose
    stock_pick.added_date is within the last 24 hours.

    Returns:
        List of tickers retrieved from the table.
    """
    client = bigquery.Client()

    # Print the condition so you can see what is being used
    logging.info("Condition used in BigQuery:")
    logging.info("  1) exchange IN ('NYQ', 'NMS', 'ASE')")
    logging.info("  2) stock_pick.added_date IS NULL OR stock_pick.added_date < CURRENT_TIMESTAMP() - 24 hours")

    query = """
        SELECT
            ticker,
            stock_pick.added_date AS added_date
        FROM `.stock_datasets.filtered_stocks`
        WHERE exchange IN ('NYQ', 'NMS', 'ASE', 'NCM')
          AND (
               stock_pick.added_date IS NULL
               OR stock_pick.added_date > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
          )
    """

    try:
        query_job = client.query(query)
        results = query_job.result()

        tickers = []
        now_utc = datetime.now(timezone.utc)  # We'll use this for a more explicit time difference

        for row in results:
            row_ticker = row["ticker"]
            added_date = row["added_date"]

            # If added_date is None, it definitely passes
            if not added_date:
                logging.debug(
                    f"Ticker {row_ticker} included because added_date is NULL (never added or no record)."
                )
            else:
                # For clarity, let's print out how old it is
                age_in_hours = (now_utc - added_date).total_seconds() / 3600.0
                logging.debug(
                    f"Ticker {row_ticker} included because its added_date = {added_date} "
                    f"which is ~{age_in_hours:.2f} hours ago (> 24 needed to be included)."
                )

            tickers.append(row_ticker)

        logging.info(f"Retrieved {len(tickers)} tickers from US exchanges in BigQuery.")
        return tickers

    except Exception as e:
        logging.error(f"Error querying BigQuery: {e}")
        sys.exit(1)


def create_combined_table(client, dataset_id, table_id):
    """
    Creates a new BigQuery table with the specified schema to store combined filings.

    Args:
        client (bigquery.Client): The BigQuery client.
        dataset_id (str): The dataset ID where the table will be created.
        table_id (str): The name of the new table.
    """
    table_ref = client.dataset(dataset_id).table(table_id)
    table = bigquery.Table(table_ref)

    # Define the schema
    schema = [
        bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("cik", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("filing_date", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("item_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("processed_text", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("effect", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("date", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("prediction_text", "STRING", mode="NULLABLE"),
    ]

    table.schema = schema

    try:
        table = client.create_table(table)  # Make an API request.
        logging.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
    except Exception as e:
        if "Already Exists" in str(e):
            logging.info(f"Table {table.project}.{table.dataset_id}.{table.table_id} already exists.")
        else:
            logging.error(f"Failed to create table {table_id}. Error: {e}")
            sys.exit(1)


def insert_into_combined_table(client, dataset_id, table_id, row):
    """
    Inserts a single row into the combined filings table.

    Args:
        client (bigquery.Client): The BigQuery client.
        dataset_id (str): The dataset ID where the table resides.
        table_id (str): The name of the table.
        row (dict): The row data to insert.
    """
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, [row])
    if errors:
        logging.error(f"Failed to insert row into {table_id}: {errors}")
    else:
        logging.debug(f"Inserted row into {table_id}: {row}")


def aggregate_processed_data(client, tickers, dataset_id, combined_table_id):
    """
    Aggregates processed data from 'processed_eight_k_filings_new' and 'processed_ten_k_filings_new'
    and inserts into the combined filings table.

    Args:
        client (bigquery.Client): The BigQuery client.
        tickers (list): List of tickers to query.
        dataset_id (str): The dataset ID where tables reside.
        combined_table_id (str): The name of the combined table.
    """
    # Query processed_eight_k_filings
    query_8k = f"""
        SELECT
            ticker,
            filing_date,
            processed_items
        FROM `{client.project}.{dataset_id}.processed_eight_k_filings_new`
        WHERE ticker IN UNNEST(@tickers)
    """

    job_config_8k = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers)
        ]
    )

    try:
        query_job_8k = client.query(query_8k, job_config=job_config_8k)
        results_8k = query_job_8k.result()
        logging.info("Fetched data from 'processed_eight_k_filings_new'.")
    except Exception as e:
        logging.error(f"Error querying 'processed_eight_k_filings_new': {e}")
        sys.exit(1)

    # Process and insert 8k filings
    for row in results_8k:
        ticker = row['ticker']
        filing_date = row['filing_date']
        processed_items = row.get('processed_items', [])

        for item in processed_items:
            item_name = item.get('item_name')
            processed_text = item.get('processed_text')
            processed_at = item.get('processed_at')
            effect = item.get('effect')
            date = item.get('date')
            stock_prediction = item.get('stock_prediction', {})
            prediction_text = stock_prediction.get('prediction_text')

            combined_row = {
                "source": "8k",
                "ticker": ticker,
                "cik": None,
                "filing_date": filing_date.isoformat() if filing_date else None,
                "item_name": item_name,
                "processed_text": processed_text,
                "processed_at": processed_at.isoformat() if processed_at else None,
                "effect": effect,
                "date": date.isoformat() if date else None,
                "prediction_text": prediction_text
            }

            insert_into_combined_table(client, dataset_id, combined_table_id, combined_row)

    # Query processed_ten_k_filings
    query_10k = f"""
        SELECT
            cik,
            filing_date,
            processed_items
        FROM `{client.project}.{dataset_id}.processed_ten_k_filings_new`
        WHERE cik IN (
            SELECT cik FROM `test1-427219.stock_datasets.filtered_stocks` WHERE ticker IN UNNEST(@tickers)
        )
    """

    job_config_10k = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers)
        ]
    )

    try:
        query_job_10k = client.query(query_10k, job_config=job_config_10k)
        results_10k = query_job_10k.result()
        logging.info("Fetched data from 'processed_ten_k_filings_new'.")
    except Exception as e:
        logging.error(f"Error querying 'processed_ten_k_filings_new': {e}")
        sys.exit(1)

    # Process and insert 10k filings
    for row in results_10k:
        cik = row['cik']
        filing_date = row['filing_date']
        processed_items = row.get('processed_items', [])

        for item in processed_items:
            item_name = item.get('item_name')
            processed_text = item.get('processed_text')
            processed_at = item.get('processed_at')

            combined_row = {
                "source": "10k",
                "ticker": None,
                "cik": cik,
                "filing_date": filing_date.isoformat() if filing_date else None,
                "item_name": item_name,
                "processed_text": processed_text,
                "processed_at": processed_at.isoformat() if processed_at else None,
                "effect": None,
                "date": None,
                "prediction_text": None
            }

            insert_into_combined_table(client, dataset_id, combined_table_id, combined_row)


def main():
    """
    Main function to orchestrate the scraping and processing workflows for each ticker in BigQuery.
    After processing, aggregates the data into a new combined table.
    """
    # Fetch tickers from BigQuery (filtered to exclude those added in last 24h)
    tickers = query_tickers_from_bigquery()

    # Define spider and processing script configurations for 8-K
    spider_8k = 'edgar8k'
    processing_script_8k = os.path.join('process10k', 'process8k_openai.py')  # Ensure this path is correct
    max_documents_8k = 5  # Set the maximum number of filings to process for 8-K

    # Define spider and processing script configurations for additional processing
    spider_cisk = 'edgarcisk'
    processing_script_cisk = os.path.join('process10k', 'process10k_openai.py')  # Ensure this path is correct

    # Process each ticker
    for ticker in tickers:
        ticker = ticker.upper()  # Ensure ticker is uppercase
        logging.info(f"Processing ticker: {ticker}")

        # Run the 8-K Scrapy spider with 'tickers' argument
        run_scrapy_spider(spider_8k, ticker, max_documents=max_documents_8k, arg_name='tickers')

        # Run the corresponding OpenAI processing script for 8-K
        run_openai_processing(processing_script_8k, ticker)

        # Run the additional Scrapy spider with 'ticker' argument
        run_scrapy_spider(spider_cisk, ticker, arg_name='ticker')

        # Run the corresponding OpenAI processing script for additional data
        run_openai_processing(processing_script_cisk, ticker)

        logging.info(f"Completed processing for ticker: {ticker}")

    # After processing all tickers, aggregate the data
    logging.info("Starting data aggregation into the combined filings table.")

    client = bigquery.Client()

    # Define dataset and table IDs
    dataset_id = 'backwards_testing'
    today_str = datetime.utcnow().strftime('%Y%m%d')
    combined_table_id = f"combined_filings_{today_str}"

    # Create the combined table
    create_combined_table(client, dataset_id, combined_table_id)

    # Aggregate data from processed filings
    aggregate_processed_data(client, tickers, dataset_id, combined_table_id)

    logging.info(f"Data aggregation completed. Combined data stored in table '{combined_table_id}'.")


if __name__ == "__main__":
    main()
