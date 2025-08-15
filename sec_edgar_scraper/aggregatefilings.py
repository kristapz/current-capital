#this script aggregates the filings from the 10k and 8k reports that were processed in
#"thisscriptdoesalltheprocessingoftheday.py"

import os
import sys
import logging
from datetime import datetime
from google.cloud import bigquery
from google.api_core.exceptions import NotFound  # Correctly import NotFound

# Import the StockMapper class from sec_cik_mapper
from sec_cik_mapper import StockMapper

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def create_combined_table(client, dataset_id, table_id):
    """
    Creates a new BigQuery table with the specified schema to store combined filings.
    If the table already exists, it verifies the schema.

    Args:
        client (bigquery.Client): The BigQuery client.
        dataset_id (str): The dataset ID where the table will be created.
        table_id (str): The name of the new table.
    """
    table_ref = client.dataset(dataset_id).table(table_id)
    desired_schema = [
        bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("cik", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("filing_date", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("processed_items", "RECORD", mode="REPEATED", fields=[
            bigquery.SchemaField("item_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processed_text", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("effect", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("date", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("prediction_text", "STRING", mode="NULLABLE"),
        ]),
    ]

    try:
        table = client.get_table(table_ref)
        existing_schema = table.schema
        # Compare schemas
        if len(existing_schema) != len(desired_schema):
            logging.error(f"Schema mismatch for table `{dataset_id}.{table_id}`.")
            logging.error(f"Expected {len(desired_schema)} fields, found {len(existing_schema)} fields.")
            logging.error("Please delete the existing table and rerun the aggregation script.")
            sys.exit(1)
        for desired_field, existing_field in zip(desired_schema, existing_schema):
            if desired_field.name != existing_field.name or \
                    desired_field.field_type != existing_field.field_type or \
                    desired_field.mode != existing_field.mode:
                logging.error(f"Schema mismatch in field `{desired_field.name}`.")
                logging.error(f"Expected: {desired_field.field_type}, {desired_field.mode}")
                logging.error(f"Found: {existing_field.field_type}, {existing_field.mode}")
                logging.error("Please delete the existing table and rerun the aggregation script.")
                sys.exit(1)
        logging.info(f"Table `{dataset_id}.{table_id}` already exists with the correct schema.")
    except NotFound:
        # Define the schema
        table = bigquery.Table(table_ref, schema=desired_schema)
        try:
            table = client.create_table(table)
            logging.info(f"Created table `{dataset_id}.{table_id}` with the correct schema.")
        except Exception as e:
            logging.error(f"Failed to create table `{dataset_id}.{table_id}`. Error: {e}")
            sys.exit(1)


def insert_into_combined_table(client, dataset_id, table_id, rows):
    """
    Inserts multiple rows into the combined filings table.

    Args:
        client (bigquery.Client): The BigQuery client.
        dataset_id (str): The dataset ID where the table resides.
        table_id (str): The name of the table.
        rows (list of dict): The rows data to insert.
    """
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        logging.error(f"Failed to insert rows into `{table_id}`: {errors}")
    else:
        logging.debug(f"Inserted {len(rows)} rows into `{table_id}`.")


def query_tickers(client):
    """
    Queries the 'stocks_filtered23' table to retrieve tickers.

    Returns:
        List of tickers retrieved from the table.
    """
    query = """
        SELECT ticker
        FROM `....stock_datasets.stocks_filteredoct21`
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        tickers = [row['ticker'] for row in results]
        logging.info(f"Retrieved {len(tickers)} tickers from 'stocks_filtered23'.")
        return tickers
    except Exception as e:
        logging.error(f"Error querying 'stocks_filtered23': {e}")
        sys.exit(1)


def map_ticker_to_cik(tickers):
    """
    Maps each ticker to its corresponding CIK using the StockMapper.

    Args:
        tickers (list): List of ticker symbols.

    Returns:
        Dictionary mapping each ticker to its CIK.
    """
    mapper = StockMapper()
    ticker_cik_map = {}
    for ticker in tickers:
        cik = mapper.ticker_to_cik.get(ticker.upper())
        if cik:
            ticker_cik_map[ticker.upper()] = cik
            logging.debug(f"Mapped ticker `{ticker.upper()}` to CIK `{cik}`.")
        else:
            logging.warning(f"No CIK found for ticker `{ticker.upper()}`.")
            ticker_cik_map[ticker.upper()] = None  # Explicitly set to None if not found
    return ticker_cik_map


def map_cik_to_ticker(ticker_cik_map):
    """
    Creates a reverse mapping from CIK to ticker.

    Args:
        ticker_cik_map (dict): Mapping of tickers to CIKs.

    Returns:
        Dictionary mapping each CIK to its ticker.
    """
    cik_to_ticker_map = {}
    for ticker, cik in ticker_cik_map.items():
        if cik:
            cik_to_ticker_map[cik] = ticker
    return cik_to_ticker_map


def aggregate_processed_data(client, tickers, ticker_cik_map, cik_to_ticker_map, dataset_id, combined_table_id):
    """
    Aggregates processed data from 'processed_eight_k_filings' and 'processed_ten_k_filings'
    and inserts into the combined filings table.

    Args:
        client (bigquery.Client): The BigQuery client.
        tickers (list): List of tickers to query.
        ticker_cik_map (dict): Mapping of tickers to CIKs.
        cik_to_ticker_map (dict): Reverse mapping of CIKs to tickers.
        dataset_id (str): The dataset ID where tables reside.
        combined_table_id (str): The name of the combined table.
    """
    # --------------- Process 8-K Filings ---------------
    # Query processed_eight_k_filings
    query_8k = """
        SELECT
            ticker,
            filing_date,
            processed_items
        FROM `....backwards_testing.processed_eight_k_filings_new`
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
        logging.info("Fetched data from 'processed_eight_k_filings'.")
    except Exception as e:
        logging.error(f"Error querying 'processed_eight_k_filings': {e}")
        sys.exit(1)

    # Process and prepare 8k filings
    filings_8k = []
    for row in results_8k:
        ticker = row['ticker']
        filing_date = row['filing_date']
        processed_items = row.get('processed_items', [])

        processed_items_list = []
        for item in processed_items:
            processed_item = {
                "item_name": item.get('item_name'),
                "processed_text": item.get('processed_text'),
                "processed_at": item.get('processed_at').isoformat() if item.get('processed_at') else None,
                "effect": item.get('effect'),
                "date": item.get('date').isoformat() if item.get('date') else None,
                "prediction_text": item.get('stock_prediction', {}).get('prediction_text')
            }
            processed_items_list.append(processed_item)

        combined_row = {
            "source": "8k",
            "ticker": ticker,
            "cik": ticker_cik_map.get(ticker.upper()),  # Include CIK if available
            "filing_date": filing_date.isoformat() if filing_date else None,
            "processed_items": processed_items_list
        }

        filings_8k.append(combined_row)

    # Insert 8k filings in batches
    if filings_8k:
        insert_into_combined_table(client, dataset_id, combined_table_id, filings_8k)
        logging.info(f"Inserted {len(filings_8k)} 8-K filings into `{combined_table_id}`.")

    # --------------- Process 10-K Filings ---------------
    # Extract CIKs (excluding None)
    ciks = [cik for cik in ticker_cik_map.values() if cik]

    if not ciks:
        logging.warning("No valid CIKs found. Skipping 'processed_ten_k_filings' aggregation.")
    else:
        # Query processed_ten_k_filings
        query_10k = """
            SELECT
                cik,
                filing_date,
                processed_items
            FROM `test1-427219.backwards_testing.processed_ten_k_filings_new`
            WHERE cik IN UNNEST(@ciks)
        """

        job_config_10k = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ciks", "STRING", ciks)
            ]
        )

        try:
            query_job_10k = client.query(query_10k, job_config=job_config_10k)
            results_10k = query_job_10k.result()
            logging.info("Fetched data from 'processed_ten_k_filings'.")
        except Exception as e:
            logging.error(f"Error querying 'processed_ten_k_filings': {e}")
            sys.exit(1)

        # Process and prepare 10k filings
        filings_10k = []
        for row in results_10k:
            cik = row['cik']
            filing_date = row['filing_date']
            processed_items = row.get('processed_items', [])

            # Retrieve ticker using cik_to_ticker_map
            ticker = cik_to_ticker_map.get(cik)

            processed_items_list = []
            for item in processed_items:
                processed_item = {
                    "item_name": item.get('item_name'),
                    "processed_text": item.get('processed_text'),
                    "processed_at": item.get('processed_at').isoformat() if item.get('processed_at') else None,
                    "effect": None,  # Not applicable for 10-K
                    "date": None,  # Not applicable for 10-K
                    "prediction_text": None  # Not applicable for 10-K
                }
                processed_items_list.append(processed_item)

            combined_row = {
                "source": "10k",
                "ticker": ticker,  # Populate ticker if available
                "cik": cik,
                "filing_date": filing_date.isoformat() if filing_date else None,
                "processed_items": processed_items_list
            }

            filings_10k.append(combined_row)

        # Insert 10k filings in batches
        if filings_10k:
            insert_into_combined_table(client, dataset_id, combined_table_id, filings_10k)
            logging.info(f"Inserted {len(filings_10k)} 10-K filings into `{combined_table_id}`.")


def main():
    """
    Main function to aggregate processed data from 8-K and 10-K filings into a combined BigQuery table.
    """
    client = bigquery.Client()

    # Define dataset and table IDs
    dataset_id = 'backwards_testing'
    today_str = datetime.utcnow().strftime('%Y%m%d')
    combined_table_id = f"combined_filings_{today_str}"

    # Create the combined table if it does not exist or verify its schema
    create_combined_table(client, dataset_id, combined_table_id)

    # Fetch tickers from 'stocks_filtered23'
    tickers = query_tickers(client)

    if not tickers:
        logging.info("No tickers found in 'stocks_filtered23'. Exiting.")
        sys.exit(0)

    # Map tickers to CIKs using StockMapper
    ticker_cik_map = map_ticker_to_cik(tickers)

    # Create reverse mapping from CIK to ticker
    cik_to_ticker_map = map_cik_to_ticker(ticker_cik_map)

    # Aggregate data from processed filings
    aggregate_processed_data(client, tickers, ticker_cik_map, cik_to_ticker_map, dataset_id, combined_table_id)

    logging.info(f"Data aggregation completed. Combined data stored in table `{combined_table_id}`.")


if __name__ == "__main__":
    main()
