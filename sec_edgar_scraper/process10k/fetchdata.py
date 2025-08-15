import os
import time
import random
import logging
import anthropic
from google.cloud import bigquery
from anthropic import AnthropicVertex
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from requests.exceptions import SSLError
from urllib3.exceptions import SSLError as URLLib3SSLError
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'backwards_testing'
source_table_id = 'processed_ten_k_filings'
full_source_table_id = f"{project_id}.{dataset_id}.{source_table_id}"
client_bq = bigquery.Client(project=project_id)

# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id=project_id)

# Directory with prompts
prompt_directory = 'sec_edgar_scraper/process10k/prompts/'

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

# Read the prompt from the specified file
def read_prompt(file_name):
    try:
        # Construct the path for the prompt file
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

# Process CIK using the Anthropic API and print the response
def process_cik_with_anthropic(cik):
    # Read the prompt from cik.txt
    prompt_file_name = "cik.txt"
    prompt = read_prompt(prompt_file_name)
    if not prompt:
        logging.warning(f"No prompt available for {prompt_file_name}, skipping processing.")
        return None

    # Print the prompt to verify its content
    logging.info(f"Prompt content being sent to API:\n{prompt}")

    # Fetch items for the specific CIK from BigQuery
    items = fetch_items_from_bigquery(cik)
    if not items:
        logging.info(f"No filings found in the source table for CIK {cik}.")
        return

    # Print the item text to verify the input being used
    item_text = str(items[0])
    logging.info(f"Item text being processed:\n{item_text}")

    # Combine the prompt and item text
    full_prompt = f"{prompt}\n\n{item_text}"
    logging.info(f"Sending prompt for CIK {cik} to Anthropic API")

    # Make the API call to Anthropic
    try:
        response = retry_anthropic_call(
            client_anthropic.messages.create,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=8192,
            model="claude-3-5-sonnet@20240620"
        )
        logging.info(f"Received response for CIK {cik} from Anthropic API")
        print(response.content[0].text)  # Print only the text part of the response
    except Exception as e:
        logging.error(f"Error processing CIK {cik} with Anthropic API: {e}")

# Main function
def main(cik):
    process_cik_with_anthropic(cik)

if __name__ == "__main__":
    # Example usage: process a specific CIK
    specific_cik = "0000065312"  # Replace with the CIK you want to process
    main(specific_cik)
