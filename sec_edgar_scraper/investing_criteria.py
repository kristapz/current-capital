import os
import json
import logging
import time
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, ServiceUnavailable, BadRequest
import yfinance as yf

# =======================
# 1. Logging Configuration
# =======================
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("investing_criteria.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =======================
# 2. Configuration
# =======================
# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Define the BigQuery table information
PROJECT_ID = ''
DATASET_ID = 'stock_datasets'
SOURCE_TABLE_ID = 'stocksenhanced_20250124'  # Existing table name
DESTINATION_TABLE_ID = 'filtered_stocks'  # Destination table name
TEMPORARY_TABLE_ID = 'stocks_filteredoct28_temp'  # Temporary staging table

SOURCE_FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE_ID}"
DESTINATION_FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{DESTINATION_TABLE_ID}"
TEMPORARY_FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TEMPORARY_TABLE_ID}"

# Initialize BigQuery client with specified location
DATASET_LOCATION = 'US'  # Replace with your dataset's location if different
client = bigquery.Client(project=PROJECT_ID, location=DATASET_LOCATION)

# Define the schema for the destination and temporary tables
SCHEMA = [
    # Top-level fields
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("exchange", "STRING", mode="NULLABLE"),  # Moved exchange to top-level

    # Nested stock_info RECORD
    bigquery.SchemaField("stock_info", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("long_business_summary", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("relative_strength", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rsi_14_days", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("eps", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("shares_outstanding", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("institutional_ownership", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("latest_volume", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("average_volume_50_days", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("percentage_volume_increase", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("pe_ratio", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("current_price", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("company_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("relative_stock_index_3_days", "FLOAT", mode="NULLABLE"),
    ]),

    # Nested stock_pick RECORD
    bigquery.SchemaField("stock_pick", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("added_date", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("added_price", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("recent_added_date_n", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("recent_added_price_n", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("addition_count", "FLOAT", mode="NULLABLE"),
    ]),
]


# =======================
# 3. Utility Functions
# =======================

def table_exists(table_id: str) -> bool:
    """
    Checks if a BigQuery table exists.

    Args:
        table_id (str): Full table ID in the format 'project.dataset.table'.

    Returns:
        bool: True if table exists, False otherwise.
    """
    try:
        client.get_table(table_id)
        logger.debug(f"Verified existence of table: {table_id}")
        return True
    except NotFound:
        logger.debug(f"Table {table_id} does not exist.")
        return False
    except Exception as e:
        logger.error(f"Error checking existence of table '{table_id}': {e}")
        return False


def create_table(table_id: str) -> bool:
    """
    Creates a BigQuery table with the specified schema if it does not already exist.

    Args:
        table_id (str): Full table ID in the format 'project.dataset.table'.

    Returns:
        bool: True if table is created or already exists, False otherwise.
    """
    if table_exists(table_id):
        logger.info(f"Table '{table_id.split('.')[-1]}' already exists in dataset '{DATASET_ID}'.")
        return True
    else:
        table = bigquery.Table(table_id, schema=SCHEMA)
        try:
            client.create_table(table)
            logger.info(f"Table '{table_id.split('.')[-1]}' created successfully in dataset '{DATASET_ID}'.")
            print(f"Table '{table_id.split('.')[-1]}' created successfully in dataset '{DATASET_ID}'.")
            return True
        except Exception as e:
            logger.error(f"Error creating table '{table_id.split('.')[-1]}': {e}")
            print(f"Error creating table '{table_id.split('.')[-1]}': {e}")
            return False


def verify_destination_schema() -> bool:
    """
    Verifies that the destination table has the 'stock_pick' field.

    Returns:
        bool: True if 'stock_pick' exists, False otherwise.
    """
    try:
        table = client.get_table(DESTINATION_FULL_TABLE_ID)
        fields = [field.name for field in table.schema]
        logger.debug(f"Destination table schema fields: {fields}")
        if 'stock_pick' in fields:
            logger.info(f"'stock_pick' field exists in destination table '{DESTINATION_TABLE_ID}'.")
            return True
        else:
            logger.warning(f"'stock_pick' field is missing in destination table '{DESTINATION_TABLE_ID}'.")
            return False
    except Exception as e:
        logger.error(f"Error verifying schema for destination table '{DESTINATION_TABLE_ID}': {e}")
        print(f"Error verifying schema for destination table '{DESTINATION_TABLE_ID}': {e}")
        return False


def recreate_destination_table() -> bool:
    """
    Recreates the destination table with the correct schema.
    **WARNING:** This will delete existing data in the destination table.

    Returns:
        bool: True if table is recreated successfully, False otherwise.
    """
    logger.info(f"Recreating destination table '{DESTINATION_TABLE_ID}' with updated schema.")
    print(f"Recreating destination table '{DESTINATION_TABLE_ID}' with updated schema.")
    try:
        # Backup existing data if necessary
        # (Implement backup logic here if required)

        # Delete the existing table
        client.delete_table(DESTINATION_FULL_TABLE_ID)
        logger.info(f"Deleted existing destination table '{DESTINATION_TABLE_ID}'.")
        print(f"Deleted existing destination table '{DESTINATION_TABLE_ID}'.")

        # Create the table with the correct schema
        table = bigquery.Table(DESTINATION_FULL_TABLE_ID, schema=SCHEMA)
        client.create_table(table)
        logger.info(f"Recreated destination table '{DESTINATION_TABLE_ID}' with updated schema.")
        print(f"Recreated destination table '{DESTINATION_TABLE_ID}' with updated schema.")
        return True
    except Exception as e:
        logger.error(f"Error recreating destination table '{DESTINATION_TABLE_ID}': {e}")
        print(f"Error recreating destination table '{DESTINATION_TABLE_ID}': {e}")
        return False


def fetch_stock_details(ticker: str):
    """
    Fetches the company name, exchange, country, and current price for the given ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        tuple: (company_name, exchange, country, current_price)
    """
    logger.debug(f"Fetching stock details for ticker: {ticker}")
    print(f"Fetching stock details for ticker: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get('longName') or info.get('shortName')
        exchange = info.get('exchange')
        country = info.get('country')

        # Fetch the latest closing price to avoid stale data
        stock_history = stock.history(period="1d")
        if not stock_history.empty:
            current_price = stock_history['Close'].iloc[-1]
            logger.debug(f"Fetched latest closing price for {ticker}: {current_price}")
        else:
            current_price = None
            logger.warning(f"Unable to fetch latest closing price for ticker: {ticker}")

        # If the current price from .history() is not available, fall back on .info
        if current_price is None:
            current_price = (
                    info.get('regularMarketPrice') or
                    info.get('ask') or
                    info.get('bid') or
                    info.get('previousClose')
            )
            logger.debug(f"Fetched fallback current price for {ticker} from info: {current_price}")

        if current_price is None:
            logger.warning(f"Current price not found for ticker: {ticker}")
            print(f"Current price not found for ticker: {ticker}")
        else:
            logger.debug(f"Fetched current price for ticker {ticker}: {current_price}")
            print(f"Fetched current price for ticker {ticker}: {current_price}")

        return company_name, exchange, country, current_price
    except Exception as e:
        logger.error(f"Error fetching stock details for {ticker}: {e}")
        print(f"Error fetching stock details for {ticker}: {e}")
        return None, None, None, None


def format_timestamp(dt: datetime) -> str:
    """
    Converts a datetime object to an ISO 8601 formatted string with UTC indicator.

    Args:
        dt (datetime): Datetime object.

    Returns:
        str: ISO 8601 formatted string.
    """
    if dt:
        formatted = dt.isoformat() + 'Z'
        logger.debug(f"Formatted datetime '{dt}' to '{formatted}'")
        print(f"Formatted datetime '{dt}' to '{formatted}'")
        return formatted
    return None


def filter_and_prepare_data() -> list:
    """
    Queries the source table, filters data based on criteria, fetches stock details,
    and prepares data for upsert.

    Returns:
        list: List of dictionaries representing rows to upsert.
    """
    logger.info("Executing query to filter stocks based on criteria...")
    print("Executing query to filter stocks based on criteria...")

    # Modify the query to select unique tickers using ROW_NUMBER()
    query = f"""
    WITH RankedStocks AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY relative_strength DESC) AS rn
        FROM `{SOURCE_FULL_TABLE_ID}`
        WHERE relative_strength > 1
          AND rsi_14_days BETWEEN 55 AND 80
          AND percentage_volume_increase IS NOT NULL AND percentage_volume_increase > 50
    )
    SELECT *
    FROM RankedStocks
    WHERE rn = 1
    """
    # This query partitions the data by 'ticker' and assigns a row number.
    # Only the first row (rn = 1) for each ticker is selected, effectively removing duplicates.
    #         AND pe_ratio IS NOT NULL AND pe_ratio < 0

    try:
        query_job = client.query(query)
        results = query_job.result()
        logger.info(f"Query completed. Number of unique tickers retrieved: {results.total_rows}")
        print(f"Query completed. Number of unique tickers retrieved: {results.total_rows}")
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        print(f"Error executing query: {e}")
        return []

    rows_to_upsert = []
    current_timestamp = datetime.utcnow()

    for row in results:
        ticker = row.ticker
        logger.info(f"Processing ticker: {ticker}")
        print(f"Processing ticker: {ticker}")

        company_name, exchange, country, current_price = fetch_stock_details(ticker)

        if not all([ticker, exchange, current_price]):
            logger.warning(f"Skipping ticker '{ticker}' due to missing information.")
            print(f"Skipping ticker '{ticker}' due to missing information.")
            continue

        logger.debug(f"Ticker: {ticker}, Company: {company_name}, Exchange: {exchange}, "
                     f"Country: {country}, Current Price: {current_price}")
        print(
            f"Ticker: {ticker}, Company: {company_name}, Exchange: {exchange}, Country: {country}, Current Price: {current_price}")

        stock_info = {
            "name": row.name,
            "sector": row.sector,
            "long_business_summary": row.long_business_summary,
            "relative_strength": row.relative_strength,
            "rsi_14_days": row.rsi_14_days,
            "eps": row.eps,
            "shares_outstanding": row.shares_outstanding,
            "institutional_ownership": row.institutional_ownership,
            "latest_volume": row.latest_volume,
            "average_volume_50_days": row.average_volume_50_days,
            "percentage_volume_increase": row.percentage_volume_increase,
            "pe_ratio": row.pe_ratio,
            "current_price": current_price,
            "company_name": company_name,
            "country": country,
            "relative_stock_index_3_days": row.relative_stock_index_3_days,
        }

        stock_pick = {
            "added_date": format_timestamp(current_timestamp),
            "added_price": current_price,
            "recent_added_date_n": format_timestamp(current_timestamp),
            "recent_added_price_n": current_price,
            "addition_count": 1,
        }

        row_dict = {
            "ticker": ticker,
            "exchange": exchange,
            "stock_info": stock_info,
            "stock_pick": stock_pick,
        }

        logger.debug(f"Prepared row for upsert: {json.dumps(row_dict, indent=2)}")
        print(f"Prepared row for upsert: {json.dumps(row_dict, indent=2)}")

        rows_to_upsert.append(row_dict)

    logger.info(f"Total unique rows prepared for upsert: {len(rows_to_upsert)}")
    print(f"Total unique rows prepared for upsert: {len(rows_to_upsert)}")
    return rows_to_upsert


def create_temporary_table(rows_to_upsert: list) -> bool:
    """
    Creates a temporary staging table to hold the upsert data and inserts the data.

    Args:
        rows_to_upsert (list): List of dictionaries representing rows to upsert.

    Returns:
        bool: True if successful, False otherwise.
    """
    temporary_table_ref = client.dataset(DATASET_ID).table(TEMPORARY_TABLE_ID)

    # Delete the temporary table if it exists
    if table_exists(TEMPORARY_FULL_TABLE_ID):
        try:
            client.delete_table(temporary_table_ref)
            logger.info(f"Temporary table '{TEMPORARY_TABLE_ID}' deleted.")
            print(f"Temporary table '{TEMPORARY_TABLE_ID}' deleted.")
        except Exception as e:
            logger.error(f"Error deleting temporary table '{TEMPORARY_TABLE_ID}': {e}")
            print(f"Error deleting temporary table '{TEMPORARY_TABLE_ID}': {e}")
            return False
    else:
        logger.info(f"Temporary table '{TEMPORARY_TABLE_ID}' does not exist. Proceeding to create a new one.")
        print(f"Temporary table '{TEMPORARY_TABLE_ID}' does not exist. Proceeding to create a new one.")

    # Create the temporary table with the same schema as destination
    try:
        client.create_table(bigquery.Table(temporary_table_ref, schema=SCHEMA))
        logger.info(f"Temporary table '{TEMPORARY_TABLE_ID}' created successfully.")
        print(f"Temporary table '{TEMPORARY_TABLE_ID}' created successfully.")
    except Exception as e:
        logger.error(f"Error creating temporary table '{TEMPORARY_TABLE_ID}': {e}")
        print(f"Error creating temporary table '{TEMPORARY_TABLE_ID}': {e}")
        return False

    # Wait briefly to ensure table creation is registered
    logger.debug("Waiting for 2 seconds to ensure table creation is registered...")
    print("Waiting for 2 seconds to ensure table creation is registered...")
    time.sleep(2)

    # Verify table creation
    if not table_exists(TEMPORARY_FULL_TABLE_ID):
        logger.error(f"Temporary table '{TEMPORARY_TABLE_ID}' was not found after creation.")
        print(f"Temporary table '{TEMPORARY_TABLE_ID}' was not found after creation.")
        return False
    else:
        logger.debug(f"Table '{TEMPORARY_TABLE_ID}' exists.")
        print(f"Table '{TEMPORARY_TABLE_ID}' exists.")

    # Insert rows into the temporary table using Load Jobs
    try:
        logger.debug(f"Starting insertion of {len(rows_to_upsert)} rows into '{TEMPORARY_TABLE_ID}'.")
        print(f"Starting insertion of {len(rows_to_upsert)} rows into '{TEMPORARY_TABLE_ID}'.")

        # Convert data to JSONL format (newline-delimited JSON)
        jsonl_data = '\n'.join([json.dumps(row) for row in rows_to_upsert])

        # Create a temporary file to upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            tmp_file.write(jsonl_data)
            tmp_file_path = tmp_file.name

        # Define the job configuration
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            schema=SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )

        # Load the data into the temporary table
        with open(tmp_file_path, "rb") as source_file:
            load_job = client.load_table_from_file(
                source_file,
                temporary_table_ref,
                job_config=job_config
            )
        logger.debug(f"Submitted load job for table '{TEMPORARY_TABLE_ID}'. Job ID: {load_job.job_id}")
        print(f"Submitted load job for table '{TEMPORARY_TABLE_ID}'. Job ID: {load_job.job_id}")

        load_job.result()  # Wait for the job to complete
        logger.info(f"Loaded {load_job.output_rows} rows into '{TEMPORARY_TABLE_ID}' successfully.")
        print(f"Loaded {load_job.output_rows} rows into '{TEMPORARY_TABLE_ID}' successfully.")

        # Cleanup the temporary file
        os.remove(tmp_file_path)
        logger.debug(f"Temporary file '{tmp_file_path}' deleted.")
        print(f"Temporary file '{tmp_file_path}' deleted.")

        return True
    except Exception as e:
        logger.error(f"Error inserting into temporary table '{TEMPORARY_TABLE_ID}': {e}")
        print(f"Error inserting into temporary table '{TEMPORARY_TABLE_ID}': {e}")
        return False


def perform_merge_operation() -> bool:
    """
    Performs a MERGE operation to upsert data from the temporary table into the destination table.

    Returns:
        bool: True if MERGE is successful, False otherwise.
    """
    merge_query = f"""
    MERGE `{DESTINATION_FULL_TABLE_ID}` T
    USING `{TEMPORARY_FULL_TABLE_ID}` S
    ON T.ticker = S.ticker
    WHEN MATCHED THEN
      UPDATE SET
        T.stock_info = S.stock_info,
        T.stock_pick.recent_added_date_n = S.stock_pick.recent_added_date_n,
        T.stock_pick.recent_added_price_n = S.stock_pick.recent_added_price_n,
        T.stock_pick.addition_count = T.stock_pick.addition_count + 1
    WHEN NOT MATCHED THEN
      INSERT (ticker, exchange, stock_info, stock_pick)
      VALUES (S.ticker, S.exchange, S.stock_info, S.stock_pick)
    """

    logger.debug(f"MERGE Query:\n{merge_query}")
    print(f"MERGE Query:\n{merge_query}")

    try:
        query_job = client.query(merge_query)
        query_job.result()  # Wait for the job to complete
        logger.info("MERGE operation completed successfully.")
        print("MERGE operation completed successfully.")
        return True
    except BadRequest as e:
        logger.error(f"BadRequest Error performing MERGE operation: {e}")
        print(f"BadRequest Error performing MERGE operation: {e}")
        return False
    except Exception as e:
        logger.error(f"Error performing MERGE operation: {e}")
        print(f"Error performing MERGE operation: {e}")
        return False


def cleanup_temporary_table():
    """
    Deletes the temporary staging table after the MERGE operation.
    """
    temporary_table_ref = client.dataset(DATASET_ID).table(TEMPORARY_TABLE_ID)
    if table_exists(TEMPORARY_FULL_TABLE_ID):
        try:
            client.delete_table(temporary_table_ref)
            logger.info(f"Temporary table '{TEMPORARY_TABLE_ID}' deleted successfully.")
            print(f"Temporary table '{TEMPORARY_TABLE_ID}' deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting temporary table '{TEMPORARY_TABLE_ID}': {e}")
            print(f"Error deleting temporary table '{TEMPORARY_TABLE_ID}': {e}")
    else:
        logger.warning(f"Temporary table '{TEMPORARY_TABLE_ID}' does not exist for deletion.")
        print(f"Temporary table '{TEMPORARY_TABLE_ID}' does not exist for deletion.")


def verify_and_update_destination_schema() -> bool:
    """
    Verifies that the destination table has the 'stock_pick' field.
    If not, recreates the table with the correct schema.

    Returns:
        bool: True if verification (and update if needed) is successful, False otherwise.
    """
    if verify_destination_schema():
        return True
    else:
        # Recreate the destination table with the correct schema
        return recreate_destination_table()


def filter_and_upsert_data():
    """
    Main function to filter data and perform upsert operations.
    """
    # Step 1: Filter and prepare data
    rows_to_upsert = filter_and_prepare_data()
    if not rows_to_upsert:
        logger.info("No data to upsert.")
        print("No data to upsert.")
        return

    # Step 2: Verify and update destination schema if necessary
    if not verify_and_update_destination_schema():
        logger.error("Failed to verify or update destination table schema. Aborting upsert.")
        print("Failed to verify or update destination table schema. Aborting upsert.")
        return

    # Step 3: Create temporary table and insert data
    if not create_temporary_table(rows_to_upsert):
        logger.error("Failed to create temporary table. Aborting upsert.")
        print("Failed to create temporary table. Aborting upsert.")
        return

    # Step 4: Perform MERGE operation with retries
    max_merge_retries = 3
    for attempt in range(1, max_merge_retries + 1):
        logger.info(f"Attempting MERGE operation (Attempt {attempt}/{max_merge_retries})...")
        print(f"Attempting MERGE operation (Attempt {attempt}/{max_merge_retries})...")
        if perform_merge_operation():
            break
        else:
            logger.warning(f"MERGE attempt {attempt} failed.")
            print(f"MERGE attempt {attempt} failed.")
            if attempt < max_merge_retries:
                delay = 5  # seconds
                logger.info(f"Retrying MERGE operation after {delay} seconds...")
                print(f"Retrying MERGE operation after {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("All MERGE attempts failed.")
                print("All MERGE attempts failed.")
                cleanup_temporary_table()
                return

    # Step 5: Clean up temporary table
    cleanup_temporary_table()
    logger.info("Upsert process completed successfully.")
    print("Upsert process completed successfully.")


def main():
    """
    Entry point for the script.
    """
    # Step 0: Ensure the destination table exists with the correct schema
    if not create_table(DESTINATION_FULL_TABLE_ID):
        logger.error(f"Failed to ensure destination table '{DESTINATION_TABLE_ID}' exists. Exiting.")
        print(f"Failed to ensure destination table '{DESTINATION_TABLE_ID}' exists. Exiting.")
        return

    # Step 1-5: Filter and upsert the data
    filter_and_upsert_data()


if __name__ == "__main__":
    main()
