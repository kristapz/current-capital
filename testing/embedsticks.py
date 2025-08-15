import os
import json
import openai
from google.cloud import bigquery
from google.cloud import aiplatform
from google.api_core.exceptions import NotFound
import time

# -------------------- Configuration --------------------

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Google Cloud Project and Dataset
PROJECT_ID = ''
DATASET_ID = 'stock_datasets'
TABLE_ID = 'stocksai'  # Updated table name
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize BigQuery Client and Vertex AI
bigquery_client = bigquery.Client(project=PROJECT_ID)
aiplatform.init(project=PROJECT_ID, location='us-east1')

# Vertex AI Endpoint
VERTEX_ENDPOINT_NAME = ""
vertex_endpoint = aiplatform.Endpoint(endpoint_name=VERTEX_ENDPOINT_NAME)

# OpenAI API Key (Recommended to use environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY", "")  # Ensure this API key is correct and secure

# OpenAI Client for Embeddings (Migration)
client = openai  # Using 'openai_client' to avoid confusion with 'bigquery_client'

# Destination Schema
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

# Maximum number of retries for operations
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# -------------------------------------------------------

def generate_vertex_embeddings(text):
    """
    Generates embeddings using Google Cloud's Vertex AI.

    Args:
        text (str): The input text to generate embeddings for.

    Returns:
        list: The generated embedding vector or an empty list on failure.
    """
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        print("Empty or NaN text encountered, returning empty embedding for Vertex AI.")
        return []

    max_characters = 1450
    cleaned_text = ' '.join(text[:max_characters].split())
    instances = [{"inputs": cleaned_text}]
    try:
        response = vertex_endpoint.predict(instances=instances)
        return response.predictions[0]
    except Exception as e:
        print(f"Error generating Vertex AI embeddings: {e}")
        return []

def generate_openai_embeddings(text):
    """
    Generates embeddings using OpenAI's Embedding API.

    Args:
        text (str): The input text to generate embeddings for.

    Returns:
        list: The generated embedding vector or an empty list on failure.
    """
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        print("Empty or NaN text encountered, returning empty embedding for OpenAI.")
        return []

    max_characters = 8000  # text-embedding-3-large can handle up to 8191 tokens
    cleaned_text = ' '.join(text[:max_characters].split())
    try:
        response = client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating OpenAI embeddings: {e}")
        return []


def update_embeddings_one_by_one(max_retries=3):
    """
    Iterates through each row in the stocksbio table and updates embeddings.

    Args:
        max_retries (int): Maximum number of retry attempts per row.
    """
    query = f"""
    SELECT name, ticker, sector, long_business_summary, row
    FROM `{FULL_TABLE_ID}`
    WHERE embeddings_large_instruct IS NULL OR openai_embeddings IS NULL
    """

    query_job = bigquery_client.query(query)
    rows = query_job.result()

    for row in rows:
        success = False
        for attempt in range(1, max_retries + 1):
            print(f"Processing ticker: {row['ticker']} - {row['name']} (Attempt {attempt})")
            vertex_embeddings = generate_vertex_embeddings(row['long_business_summary'])
            openai_embeddings = generate_openai_embeddings(row['long_business_summary'])
            update_row(row['ticker'], vertex_embeddings, openai_embeddings)
            if verify_row(row['ticker']):
                success = True
                break
            else:
                print(f"Verification failed for ticker {row['ticker']} on attempt {attempt}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)

        if not success:
            print(f"Failed to update ticker {row['ticker']} after {max_retries} attempts.")

def update_row(ticker, vertex_embeddings, openai_embeddings):
    """
    Updates a single row in the stocksbio table with the generated embeddings.

    Args:
        ticker (str): The stock ticker symbol.
        vertex_embeddings (list): Embedding vector from Vertex AI.
        openai_embeddings (list): Embedding vector from OpenAI.
    """
    # Serialize embeddings to JSON strings
    vertex_embeddings_json = json.dumps(vertex_embeddings)
    openai_embeddings_json = json.dumps(openai_embeddings)

    update_query = f"""
    UPDATE `{FULL_TABLE_ID}`
    SET 
        embeddings_large_instruct = @vertex_embeddings,
        openai_embeddings = @openai_embeddings
    WHERE ticker = @ticker
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("vertex_embeddings", "STRING", vertex_embeddings_json),
            bigquery.ScalarQueryParameter("openai_embeddings", "STRING", openai_embeddings_json),
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ]
    )

    try:
        query_job = bigquery_client.query(update_query, job_config=job_config)
        query_job.result()  # Wait for the job to complete
        print(f"Ticker {ticker} updated with Vertex AI and OpenAI embeddings.")
    except Exception as e:
        print(f"Error updating ticker {ticker}: {e}")

def verify_row(ticker):
    """
    Verifies that the embeddings have been successfully updated for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        bool: True if both embeddings are updated, False otherwise.
    """
    verify_query = f"""
    SELECT ticker, 
           embeddings_large_instruct,
           openai_embeddings,
           CASE WHEN embeddings_large_instruct IS NOT NULL THEN 1 ELSE 0 END as vertex_updated,
           CASE WHEN openai_embeddings IS NOT NULL THEN 1 ELSE 0 END as openai_updated
    FROM `{FULL_TABLE_ID}`
    WHERE ticker = @ticker
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ]
    )

    try:
        query_job = bigquery_client.query(verify_query, job_config=job_config)
        result = next(iter(query_job.result()), None)

        if result and result.vertex_updated and result.openai_updated:
            print(f"Verification successful for ticker {ticker}.")
            return True
        else:
            print(f"Verification failed for ticker {ticker}.")
            return False
    except Exception as e:
        print(f"Error verifying ticker {ticker}: {e}")
        return False

def wait_for_buffer_to_clear(client, project_id, dataset_id, table_id, timeout=300):
    """
    Waits until the streaming buffer of a BigQuery table is empty.

    Args:
        client (bigquery.Client): BigQuery client.
        project_id (str): Google Cloud project ID.
        dataset_id (str): BigQuery dataset ID.
        table_id (str): BigQuery table ID.
        timeout (int): Maximum time to wait in seconds.

    Returns:
        bool: True if the buffer cleared within the timeout, False otherwise.
    """
    table_ref = client.dataset(dataset_id).table(table_id)
    start_time = time.time()
    while time.time() - start_time < timeout:
        table = client.get_table(table_ref)
        if not table.streaming_buffer:
            print("Streaming buffer is clear.")
            return True
        else:
            print("Streaming buffer is not clear yet. Waiting...")
            time.sleep(10)  # Wait for 10 seconds before checking again
    print("Timeout waiting for streaming buffer to clear.")
    return False

def main():
    """
    Main function to initiate the embedding update process.
    """
    # Wait for streaming buffer to clear
   # buffer_cleared = wait_for_buffer_to_clear(bigquery_client, PROJECT_ID, DATASET_ID, TABLE_ID)
  #  if not buffer_cleared:
 #       print("Cannot proceed with updates as the streaming buffer is still active.")
  #      return

    # Start updating embeddings
    update_embeddings_one_by_one()

if __name__ == "__main__":
    main()
