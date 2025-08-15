import os
import time
import random
import logging
import anthropic
import google
from google.cloud import bigquery
from anthropic import AnthropicVertex
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from requests.exceptions import SSLError
from urllib3.exceptions import SSLError as URLLib3SSLError
from sec_cik_mapper import StockMapper
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'backwards_testing'
source_table_id = 'ten_k_filings'
processed_table_id = 'processed_ten_k_filings'  # New table for storing processed items
full_source_table_id = f"{project_id}.{dataset_id}.{source_table_id}"
full_processed_table_id = f"{project_id}.{dataset_id}.{processed_table_id}"
client_bq = bigquery.Client(project=project_id)
stock_mapper = StockMapper()

# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id=project_id)

# Directory with prompts
prompt_directory = 'sec_edgar_scraper/process10k/prompts/'

# Create the new processed table in BigQuery if it doesn't exist
def create_processed_table():
    # Define the schema for the table
    schema = [
        bigquery.SchemaField("cik", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("filing_date", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField(
            "processed_items",
            "RECORD",
            mode="REPEATED",
            fields=[
                bigquery.SchemaField("item_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("processed_text", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
            ],
        ),
    ]

    table = bigquery.Table(full_processed_table_id, schema=schema)
    try:
        client_bq.get_table(full_processed_table_id)  # Check if table exists
        logging.info(f"Table {full_processed_table_id} already exists.")
    except google_exceptions.NotFound:
        client_bq.create_table(table)
        logging.info(f"Created table {full_processed_table_id} in BigQuery.")

# Retry settings for Anthropic API
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((anthropic.RateLimitError, SSLError, URLLib3SSLError))
)
def retry_anthropic_call(func, *args, **kwargs):
    try:
        start_time = time.time()
        response = anthropic_call_with_timeout(func, timeout=60, *args, **kwargs)
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds")
        return response
    except (anthropic.RateLimitError, SSLError, URLLib3SSLError) as e:
        logging.warning(f"API call failed due to retryable error. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 5))  # Add a random delay before retry
        raise
    except Exception as e:
        logging.error(f"Unexpected error in API call: {e}")
        raise

# Anthropic API call with timeout handling
class AnthropicTimeoutError(Exception):
    pass

def anthropic_call_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise AnthropicTimeoutError(f"Anthropic API call timed out after {timeout} seconds")

# Fetch items from the BigQuery source table for a specific CIK
def fetch_items_from_bigquery(cik):
    query = f"SELECT * FROM `{full_source_table_id}` WHERE cik = @cik LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cik", "STRING", cik),
        ]
    )
    query_job = client_bq.query(query, job_config=job_config)
    rows = [row for row in query_job]
    logging.info(f"Fetched {len(rows)} rows for CIK {cik} from BigQuery table {full_source_table_id}")
    return rows

# Updated read_prompt function
def read_prompt(file_name):
    try:
        # Dynamically create the correct path for the prompt file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        prompt_dir = os.path.join(current_dir, 'prompts')  # Adjust this as needed based on the actual location
        prompt_path = os.path.join(prompt_dir, file_name)

        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
            logging.info(f"Successfully read prompt from {file_name}")
            return prompt
    except Exception as e:
        logging.error(f"Failed to read prompt from {file_name}: {e}")
        return ""

def upsert_processed_item(cik, filing_date, item_name, processed_text):
    processed_text_escaped = processed_text.replace("'", "''")

    # SQL to merge processed items in a nested format
    merge_query = f"""
    MERGE `{full_processed_table_id}` T
    USING (SELECT @cik AS cik) S
    ON T.cik = S.cik
    WHEN MATCHED THEN
      UPDATE SET
        processed_items = ARRAY_CONCAT(T.processed_items, [STRUCT(@item_name AS item_name, @processed_text AS processed_text, CURRENT_TIMESTAMP() AS processed_at)])
    WHEN NOT MATCHED THEN
      INSERT (cik, filing_date, processed_items)
      VALUES (@cik,  @filing_date, [STRUCT(@item_name AS item_name, @processed_text AS processed_text, CURRENT_TIMESTAMP() AS processed_at)]);
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cik", "STRING", cik),
            bigquery.ScalarQueryParameter("filing_date", "TIMESTAMP", filing_date),
            bigquery.ScalarQueryParameter("item_name", "STRING", item_name),
            bigquery.ScalarQueryParameter("processed_text", "STRING", processed_text_escaped)
        ]
    )

    query_job = client_bq.query(merge_query, job_config=job_config)
    query_job.result()  # Wait for the query to complete
    logging.info(f"Upserted processed item {item_name} for CIK {cik} into {full_processed_table_id}")

# Process each item using Anthropic API
def process_item_with_anthropic(cik, item_text, prompt_file_name):
    prompt = read_prompt(prompt_file_name)
    if not prompt:
        logging.warning(f"No prompt available for {prompt_file_name}, skipping processing.")
        return None

    full_prompt = f"{prompt}\n\n{item_text}"
    logging.info(f"Sending prompt for CIK {cik} to Anthropic API")

    try:
        response = retry_anthropic_call(
            client_anthropic.messages.create,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=8192,
            model="claude-3-5-sonnet@20240620"
        )
        logging.info(f"Received response for CIK {cik} from Anthropic API")
        return response.content[0].text  # Return only the text part of the response
    except Exception as e:
        logging.error(f"Error processing item with Anthropic API: {e}")
        return None

# Process items in parallel and upsert one by one
def process_row_items_parallel(cik, filing_date, row, item_prompt_mapping):
    logging.info(f"Processing items for CIK: {cik}")

    # Define a function to process a single item
    def process_single_item(item_name, item_text):
        return process_item_with_anthropic(cik, item_text, item_prompt_mapping[item_name])

    # Process all items concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare futures for concurrent processing
        futures = {
            executor.submit(process_single_item, item_name, row[item_name]): item_name
            for item_name in item_prompt_mapping.keys() if row.get(item_name) is not None
        }

        # Collect the responses and upsert each one individually
        for future in concurrent.futures.as_completed(futures):
            item_name = futures[future]
            try:
                response_text = future.result()
                if response_text:
                    upsert_processed_item(cik, filing_date, item_name, response_text)
                    logging.info(f"Processed item '{item_name}' for CIK {cik} successfully.")
                else:
                    logging.warning(f"Failed to process item '{item_name}' for CIK {cik}.")
            except Exception as e:
                logging.error(f"Error processing item '{item_name}' for CIK {cik}: {e}")

# Main function to process a specific CIK
def main(cik):
    # Create the new processed table in BigQuery if it doesn't exist
    create_processed_table()

    # Fetch the filing for the specific CIK from the BigQuery source table
    items = fetch_items_from_bigquery(cik)

    if not items:
        logging.info(f"No filings found in the source table for CIK {cik}.")
        return

    # Define the mapping of items to prompt files in the correct order
    global item_prompt_mapping
    item_prompt_mapping = {
        "item1_business": "item1.txt",
        "item1a_risk_factors": "item1a.txt",
        "item1b_unresolved_staff_comments": "item1b.txt",
        "item1c_cybersecurity": "item1c.txt",
        "item2_properties": "item2.txt",
        "item3_legal_proceedings": "item3.txt",
        "item5_market_for_registrants_common_equity_related_stockholder_matters_and_issuer_purchases_of_equity_securities": "item5.txt",
        "item6_selected_financial_data": "item6.txt",
        "item7_management_discussion_and_analysis": "item7.txt",
        "item7a_quantitative_and_qualitative_disclosures_about_market_risk": "item7a.txt",
        "item8_financial_statements_and_supplementary_data": "item8.txt",
        "item9_changes_in_and_disagreements_with_accountants_on_accounting_and_financial_disclosure": "item9.txt",
        "item9a_controls_and_procedures": "item9a.txt",
        "item9b_other_information": "item9b.txt",
        "item9c_disclosure_regarding_foreign_jurisdictions_that_prevent_inspections": "item9c.txt",
        "item10_directors_executive_officers_and_corporate_governance": "item10.txt",
        "item11_executive_compensation": "item11.txt",
        "item12_security_ownership_of_certain_beneficial_owners_and_management_and_related_stockholder_matters": "item12.txt",
        "item13_certain_relationships_and_related_transactions_and_director_independence": "item13.txt",
        "item14_principal_accountant_fees_and_services": "item14.txt",
        "item15_exhibits_financial_statement_schedules": "item15.txt",
        "item16_form_10k_summary": "item16.txt"
    }

    # Process the first filing's items
    item = items[0]
    filing_date = item['filing_date']
    process_row_items_parallel(cik, filing_date, item, item_prompt_mapping)

if __name__ == "__main__":
    # Example usage: process a specific CIK
    ticker = "JPM"
    specific_cik = stock_mapper.ticker_to_cik.get(ticker.upper())
    main(specific_cik)




