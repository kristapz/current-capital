import os
import json
import time
import yfinance as yf
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import pandas as pd

# -------------------- Configuration --------------------

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# BigQuery configuration
PROJECT_ID = ''
DATASET_ID = 'stock_datasets'
SOURCE_TABLE_ID = 'aistocktickers'  # Table with tickers in string_field_0
DESTINATION_TABLE_ID = 'stocksai'  # Target table to populate (Change if needed)

# Schema for the destination table
DESTINATION_SCHEMA = [
    bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ticker", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("long_business_summary", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embeddings", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embeddings_large", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("openai_embeddings", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("embeddings_large_instruct", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("row", "INTEGER", mode="NULLABLE"),
]

# Maximum number of retries for fetching data
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


# -------------------------------------------------------

def initialize_bigquery_client():
    """Initializes and returns a BigQuery client."""
    client = bigquery.Client(project=PROJECT_ID)
    return client


def fetch_tickers(client):
    """
    Fetches tickers from the source table's string_field_0.

    Returns:
        pandas.DataFrame: DataFrame containing a single column 'ticker'.
    """
    query = f"""
        SELECT string_field_0 AS ticker
        FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE_ID}`
    """
    try:
        df = client.query(query).to_dataframe()
        print(f"Fetched {len(df)} tickers from {SOURCE_TABLE_ID}.")
        return df
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return pd.DataFrame(columns=['ticker'])


def fetch_stock_info(ticker):
    """
    Fetches sector and long business summary for a given ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Dictionary containing 'sector' and 'long_business_summary'.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', None)
            long_business_summary = info.get('longBusinessSummary', None)
            if not long_business_summary:
                # Attempt to fetch from description or other fields if available
                long_business_summary = info.get('description', '')
            print(f"Successfully fetched data for {ticker}.")
            return {
                'sector': sector,
                'long_business_summary': long_business_summary
            }
        except Exception as e:
            print(f"Error fetching data for {ticker} on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_DELAY * attempt  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to fetch data for {ticker} after {MAX_RETRIES} attempts.")
                return {
                    'sector': None,
                    'long_business_summary': None
                }


def prepare_rows(df):
    """
    Prepares rows with fetched stock information for insertion into BigQuery.

    Args:
        df (pandas.DataFrame): DataFrame containing 'ticker'.

    Returns:
        list of dict: List containing dictionaries of row data.
    """
    rows_to_insert = []
    for index, row in df.iterrows():
        ticker = row['ticker']
        print(f"Processing ticker: {ticker}")
        stock_info = fetch_stock_info(ticker)
        if stock_info['sector'] is None and stock_info['long_business_summary'] is None:
            print(f"Skipping ticker {ticker} due to missing data.")
            continue  # Skip if essential data is missing
        row_data = {
            "name": None,  # We no longer have a 'name' from string_field_1
            "ticker": ticker,
            "sector": stock_info['sector'],
            "long_business_summary": stock_info['long_business_summary'],
            "embeddings": None,
            "embeddings_large": None,
            "openai_embeddings": None,
            "embeddings_large_instruct": None,
            "row": index + 1  # Assuming row numbers start at 1
        }
        rows_to_insert.append(row_data)
    print(f"Prepared {len(rows_to_insert)} rows for insertion.")
    return rows_to_insert


def insert_rows(client, rows):
    """
    Inserts rows into the destination BigQuery table.

    Args:
        client (bigquery.Client): BigQuery client.
        rows (list of dict): Rows to insert.
    """
    table_ref = client.dataset(DATASET_ID).table(DESTINATION_TABLE_ID)
    table = bigquery.Table(table_ref, schema=DESTINATION_SCHEMA)

    # Create the table if it doesn't exist
    try:
        client.get_table(table_ref)
        print(f"Table {DESTINATION_TABLE_ID} already exists.")
    except NotFound:
        table = client.create_table(table)
        print(f"Created table {DESTINATION_TABLE_ID} with the specified schema.")

    # Insert rows
    try:
        errors = client.insert_rows_json(table, rows)
        if not errors:
            print(f"Successfully inserted {len(rows)} rows into {DESTINATION_TABLE_ID}.")
        else:
            print(f"Encountered errors while inserting rows: {errors}")
    except Exception as e:
        print(f"An error occurred while inserting rows: {e}")


def main():
    client = initialize_bigquery_client()
    tickers_df = fetch_tickers(client)

    if tickers_df.empty:
        print("No tickers found to process.")
        return

    rows_to_insert = prepare_rows(tickers_df)

    if not rows_to_insert:
        print("No valid rows to insert.")
        return

    insert_rows(client, rows_to_insert)


if __name__ == "__main__":
    main()
