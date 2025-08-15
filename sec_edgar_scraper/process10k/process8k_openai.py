import os
import time
import random
import logging
import openai
from openai import OpenAI
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sec_cik_mapper import StockMapper
from datetime import datetime
import argparse
import re
from requests.exceptions import RequestException
import json  # Ensure this is imported

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment and BigQuery configuration
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Update the path accordingly
project_id = ''
dataset_id = 'backwards_testing'
source_table_id = 'eight_k_filings'
processed_table_id_new = 'processed_eight_k_filings_new'  # Updated table name for the new table
full_source_table_id = f"{project_id}.{dataset_id}.{source_table_id}"
full_processed_table_id_new = f"{project_id}.{dataset_id}.{processed_table_id_new}"  # Fully qualified name for the new table
client_bq = bigquery.Client(project=project_id)
stock_mapper = StockMapper()

# Initialize OpenAI client
openai_api_key = ""  # Ensure this API key is correct and secure
client = OpenAI(api_key=openai_api_key)

# Directory with prompts
prompt_directory = '/Users/kristapszilgalvis/Desktop/current-capital/sec_edgar_scraper/process10k/prompts/prompt8k'
prompt_file_name = '8kprompt'  # The exact name of your prompt file

# Define the schema for processed_eight_k_filings_new
processed_eight_k_filings_new_schema = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
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
    bigquery.SchemaField("effect", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("date", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField(
        "stock_prediction",
        "RECORD",
        mode="NULLABLE",
        fields=[
            bigquery.SchemaField("stock_ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embeddings", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("prediction_text", "STRING", mode="NULLABLE"),
        ],
    ),
    bigquery.SchemaField("filing_url", "STRING", mode="REQUIRED"),  # New field added to store the filing URL
]

# Create the new processed table in BigQuery if it doesn't exist
def create_processed_table_new():
    table = bigquery.Table(full_processed_table_id_new, schema=processed_eight_k_filings_new_schema)
    try:
        client_bq.get_table(full_processed_table_id_new)  # Check if table exists
        logging.info(f"Table {full_processed_table_id_new} already exists.")
    except google_exceptions.NotFound:
        client_bq.create_table(table)
        logging.info(f"Created table {full_processed_table_id_new} in BigQuery.")

# Retry settings for OpenAI API
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RequestException, Exception))
)
def retry_openai_call(func, *args, **kwargs):
    try:
        start_time = time.time()
        response = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds")
        return response
    except Exception as e:
        logging.warning(f"API call failed due to error. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 5))  # Add a random delay before retry
        raise

# Fetch items from the BigQuery source table for a specific CIK
def fetch_items_from_bigquery(cik):
    query = f"""
    SELECT *
    FROM `{full_source_table_id}`
    WHERE cik = @cik
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cik", "STRING", cik),
        ]
    )
    query_job = client_bq.query(query, job_config=job_config)
    rows = [row for row in query_job]
    logging.info(f"Fetched {len(rows)} rows for CIK {cik} from BigQuery table {full_source_table_id}")
    return rows

# Read the prompt from the specified file
def read_prompt(file_name):
    try:
        prompt_path = os.path.join(prompt_directory, file_name)
        with open(prompt_path, 'r') as file:
            prompt = file.read().strip()
            logging.info(f"Successfully read prompt from {file_name}")
            return prompt
    except Exception as e:
        logging.error(f"Failed to read prompt from {file_name}: {e}")
        return ""

# Parse the OpenAI response using regex
def parse_openai_response(response_text):
    try:
        normalized_response = response_text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        logging.debug(f"Normalized OpenAI response:\n{normalized_response}")

        pattern_with_key = r'\{\{effect:\s*"([\w\s]+)"\}\},\{\{(?:\"?prediction(?:[\s_]?and[\s_]?reasoning)?\"?\s*:\s*)?"?(.*?)"?\}\}'
        pattern_without_key = r'\{\{effect:\s*"([\w\s]+)"\}\},\{\{(.*?)\}\}'

        match = re.match(pattern_with_key, normalized_response, re.IGNORECASE)
        if match:
            effect = match.group(1).strip().lower()
            prediction_text = match.group(2).strip('"').strip()
            logging.debug(f"Matched with key - Effect: {effect}, Prediction Text: {prediction_text}")
            return effect, prediction_text

        match = re.match(pattern_without_key, normalized_response, re.IGNORECASE)
        if match:
            effect = match.group(1).strip().lower()
            prediction_text = match.group(2).strip('"').strip()
            logging.debug(f"Matched without key - Effect: {effect}, Prediction Text: {prediction_text}")
            return effect, prediction_text

        logging.warning(f"Response does not match expected formats: {normalized_response}")
        return "none", ""

    except Exception as e:
        logging.error(f"Error parsing OpenAI response: {e}")
        return "none", ""

# Generate OpenAI embeddings
def generate_openai_embeddings(text):
    """Generate embeddings using OpenAI."""
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding for OpenAI.")
        return ""

    max_characters = 8000  # text-embedding-3-large can handle up to 8191 tokens
    cleaned_text = ' '.join(text[:max_characters].split())
    try:
        response = client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        embedding = response['data'][0]['embedding']
        # Convert embedding list to JSON string
        return json.dumps(embedding)
    except Exception as e:
        logging.error(f"Error generating OpenAI embeddings: {e}")
        return ""

# Function to check if a filing_url already exists in the destination table
def filing_url_exists(filing_url):
    query = f"""
    SELECT COUNT(*) as count
    FROM `{full_processed_table_id_new}`
    WHERE filing_url = @filing_url
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("filing_url", "STRING", filing_url),
        ]
    )
    query_job = client_bq.query(query, job_config=job_config)
    result = query_job.result()
    count = 0
    for row in result:
        count = row.count
    return count > 0

# Insert processed filing into BigQuery (new table)
def insert_processed_filing_new(ticker, filing_date, processed_items, effect, date_related, stock_prediction, filing_url):
    """
    Inserts the processed 8-K filing into the processed_eight_k_filings_new table.

    Args:
        ticker (str): Stock ticker symbol.
        filing_date (date or datetime): Date of the 8-K filing.
        processed_items (list of dict): List of processed items.
        effect (str): Overall effect of the filing.
        date_related (datetime): Specific date related to the filing or event.
        stock_prediction (dict): Stock prediction details, including embeddings.
        filing_url (str): URL to the 8-K filing on SEC EDGAR.
    """
    # Convert filing_date from date to datetime if necessary
    if isinstance(filing_date, datetime):
        filing_datetime = filing_date
    else:
        filing_datetime = datetime.combine(filing_date, datetime.min.time())

    # Prepare the row to insert
    row = {
        "ticker": ticker,
        "filing_date": filing_datetime.isoformat("T") + "Z",
        "processed_items": [
            {
                "item_name": item['item_name'],
                "processed_text": item['processed_text'],
                "processed_at": item['processed_at'].isoformat("T") + "Z",
            }
            for item in processed_items
        ],
        "effect": effect,  # Correctly mapped to the 'effect' field
        "date": date_related.isoformat("T") + "Z",
        "stock_prediction": stock_prediction,  # Includes embeddings
        "filing_url": filing_url  # New field added
    }

    # Insert the row into BigQuery
    try:
        errors = client_bq.insert_rows_json(full_processed_table_id_new, [row])
        if errors:
            logging.error(f"Errors occurred while inserting rows: {errors}")
        else:
            logging.info(f"Inserted processed 8-K filing for {ticker} on {filing_datetime.date()}")
    except Exception as e:
        logging.error(f"Exception occurred while inserting into BigQuery: {e}")

# Process a single item using OpenAI API
def process_item_with_openai(ticker, item_text):
    """
    Processes a single item with OpenAI and returns the parsed response along with embeddings.

    Args:
        ticker (str): Stock ticker symbol.
        item_text (str): Text content of the item.

    Returns:
        tuple: (effect, prediction_text, embeddings)
    """
    prompt = read_prompt(prompt_file_name)
    if not prompt:
        logging.warning(f"No prompt available for processing. Skipping item.")
        return None, None, None

    full_prompt = f"{prompt}\n\n{item_text}"
    logging.info(f"Sending prompt for ticker {ticker} to OpenAI API")
    logging.debug(f"Prompt sent to OpenAI for ticker {ticker}:\n{full_prompt}")

    try:
        response = retry_openai_call(
            client.chat.completions.create,
            model="o1-mini",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
        )
        logging.info(f"Received response for ticker {ticker} from OpenAI API")
        response_text = response.choices[0].message.content
        logging.debug(f"Raw OpenAI response for ticker {ticker}:\n{response_text}")

        effect, prediction_text = parse_openai_response(response_text)
        logging.debug(f"Parsed Effect: {effect}, Parsed Prediction Text: {prediction_text}")

        if prediction_text:
            embeddings = generate_openai_embeddings(prediction_text)
            logging.debug(f"Generated Embeddings for ticker {ticker}: {embeddings}")
        else:
            embeddings = ""
            logging.warning(f"No prediction text to generate embeddings for ticker {ticker}.")

        return effect, prediction_text, embeddings
    except Exception as e:
        logging.error(f"Error processing item with OpenAI API: {e}")
        return None, None, None

def main(ticker):
    # Create the new processed table in BigQuery if it doesn't exist
    create_processed_table_new()

    # Fetch the filing(s) for the specific ticker from the BigQuery source table
    specific_cik = stock_mapper.ticker_to_cik.get(ticker)
    if not specific_cik:
        logging.error(f"CIK not found for ticker {ticker}")
        return

    filings = fetch_items_from_bigquery(specific_cik)

    if not filings:
        logging.info(f"No filings found in the source table for CIK {specific_cik}.")
        return

    for filing in filings:
        filing_date = filing['filing_date']
        accession_number = filing['accession_number']
        url = filing['url']  # Extract the URL from the source filing

        # Check if the filing_url already exists in the destination table
        if filing_url_exists(url):
            print(f"Filing with URL {url} already exists. Skipping.")
            continue  # Skip processing this filing

        # Initialize lists to hold processed items and effects
        processed_items = []
        effects = []

        # Iterate through all possible item fields
        for field in filing.keys():
            # Skip metadata fields
            if field in ['cik', 'company', 'filing_date', 'accession_number', 'url']:
                continue

            item_text = filing.get(field)
            if item_text:
                logging.info(f"Processing item '{field}' for ticker {ticker}")
                effect, prediction_text, embeddings = process_item_with_openai(ticker, item_text)

                if effect is not None and prediction_text:
                    # Append the processed item to the list
                    processed_items.append({
                        "item_name": field,
                        "processed_text": prediction_text,
                        "processed_at": datetime.utcnow(),
                    })
                    # Collect the effect for aggregation
                    effects.append(effect)
                    logging.info(f"Successfully processed item '{field}' for ticker {ticker}")
                else:
                    logging.warning(f"Failed to process item '{field}' for ticker {ticker}")

        if not processed_items:
            logging.info(f"No items processed for filing {accession_number} of ticker {ticker}. Skipping insertion.")
            continue

        # Aggregate 'effect' based on collected effects
        # Define a hierarchy for effect levels
        effect_hierarchy = {"none": 0, "low": 1, "moderate": 2, "high": 3, "very high": 4}
        aggregated_effect_level = 0
        for effect_text in effects:
            effect_text = effect_text.lower()
            if effect_text in effect_hierarchy:
                aggregated_effect_level = max(aggregated_effect_level, effect_hierarchy[effect_text])
            else:
                logging.warning(
                    f"Invalid effect '{effect_text}' found. Ignoring for aggregation."
                )

        aggregated_effect = (
            [k for k, v in effect_hierarchy.items() if v == aggregated_effect_level][0]
            if aggregated_effect_level > 0
            else "none"
        )

        # Aggregate 'prediction_text' by concatenating all predictions
        aggregated_prediction_text = " ".join(
            [item['processed_text'] for item in processed_items if item['processed_text']]
        )

        # Generate embeddings for the aggregated_prediction_text
        aggregated_embeddings = generate_openai_embeddings(aggregated_prediction_text)
        logging.debug(
            f"Generated embeddings for aggregated prediction text of ticker {ticker}: {aggregated_embeddings}"
        )

        # Create 'stock_prediction' field
        stock_prediction = {
            "stock_ticker": ticker,
            "embeddings": aggregated_embeddings,  # Embeddings as JSON string
            "prediction_text": aggregated_prediction_text
        }

        # Use current UTC time for 'date'
        date_related = datetime.utcnow()

        # Insert the aggregated data into BigQuery (new table)
        insert_processed_filing_new(
            ticker=ticker,
            filing_date=filing_date,
            processed_items=processed_items,
            effect=aggregated_effect,
            date_related=date_related,
            stock_prediction=stock_prediction,
            filing_url=url  # Pass the URL to be inserted
        )

    logging.info(f"Completed processing for ticker {ticker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process 8-K filings with OpenAI.')
    parser.add_argument('-ticker', '--ticker', type=str, required=True, help='Ticker symbol to process (e.g., AAPL)')
    args = parser.parse_args()

    ticker = args.ticker.upper()
    main(ticker)
