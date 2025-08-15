import os
import time
import random
import logging
import json
from google.cloud import bigquery
from anthropic import AnthropicVertex
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from requests.exceptions import SSLError
from urllib3.exceptions import SSLError as URLLib3SSLError
from datetime import datetime
from openai import OpenAI
from google.cloud import aiplatform
from typing import List
from sec_cik_mapper import StockMapper


# ================================
# Configuration and Setup
# ================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Define your project ID and dataset ID
project_id = ''
dataset_id = 'backwards_testing'
source_table_id = 'processed_ten_k_filings'  # Updated source table
insights_table_id = 'insights_with_embeddings'  # New table for storing insights and embeddings

# Full table identifiers
full_source_table_id = f"{project_id}.{dataset_id}.{source_table_id}"
full_insights_table_id = f"{project_id}.{dataset_id}.{insights_table_id}"
stock_mapper = StockMapper()

# Initialize BigQuery client
client_bq = bigquery.Client(project=project_id)
cik = "0001452857"
# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id=project_id)

# Directory with prompts
prompt_directory = 'sec_edgar_scraper/process10k/prompts/'

# ================================
# Initialize Embedding Clients
# ================================

# Initialize OpenAI client
openai_api_key = ""  # Replace with your actual OpenAI API key
openai_client = OpenAI(api_key=openai_api_key)

# Initialize Vertex AI
aiplatform.init(project=project_id, location='us-east1')

# Vertex AI Endpoint
vertex_endpoint_name = ""  # Replace with your actual endpoint
vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_name)

# ================================
# Define BigQuery Schema
# ================================

insights_schema = [
    bigquery.SchemaField("cik", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("filing_date", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("company_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("industry", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("tier", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("overallInsights", "RECORD", mode="REQUIRED", fields=[
        bigquery.SchemaField("businessStrategyAndVision", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("competitiveAdvantageAndDifferentiators", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("riskExposureAndMitigationStrategies", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("managementQualityAndLeadershipEffectiveness", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("corporateGovernancePractices", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("regulatoryAndLegalEnvironment", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("sustainabilityAndESGInitiatives", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("intellectualPropertyAndInnovationPipeline", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("innovationInsight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("innovationInsight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("innovationInsight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("marketPositionAndShare", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("potentialCompetitor1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("potentialCompetitor2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("potentialCompetitor3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("customerAndSupplierRelationships", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("strategicAlliancesAndPartnerships", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("operationalEfficiencyAndSupplyChainManagement", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("technologicalInfrastructureAndCybersecurityMeasures", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("productAndServiceDiversification", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("environmentalImpactAndResourceManagement", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("litigationAndComplianceIssues", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("specificLawsuit1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("specificLawsuit2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("specificLawsuit3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("internationalOperationsAndGeopoliticalRisks", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
        bigquery.SchemaField("marketResearchAndDescriptionOfMarket", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("insight1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight2", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight3", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight4", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("insight5", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]),
    ])
]

# ================================
# Function Definitions
# ================================

def create_insights_table():
    """Create the insights_with_embeddings table in BigQuery if it doesn't exist."""
    table_ref = client_bq.dataset(dataset_id).table(insights_table_id)
    try:
        client_bq.get_table(table_ref)  # Check if table exists
        logging.info(f"Table {full_insights_table_id} already exists.")
    except google_exceptions.NotFound:
        table = bigquery.Table(table_ref, schema=insights_schema)
        client_bq.create_table(table)
        logging.info(f"Created table {full_insights_table_id} in BigQuery.")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
)
def retry_anthropic_call(func, *args, **kwargs):
    """Retry mechanism for Anthropic API calls."""
    try:
        start_time = time.time()
        response = anthropic_call_with_timeout(func, timeout=60, *args, **kwargs)
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds")
        return response
    except (AnthropicVertex.RateLimitError, SSLError, URLLib3SSLError) as e:
        logging.warning(f"API call failed due to retryable error. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 5))  # Add a random delay before retry
        raise
    except Exception as e:
        logging.error(f"Unexpected error in API call: {e}")
        raise

class AnthropicTimeoutError(Exception):
    """Custom exception for Anthropic API call timeouts."""
    pass

def anthropic_call_with_timeout(func, timeout, *args, **kwargs):
    """Handle Anthropic API call with timeout."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise AnthropicTimeoutError(f"Anthropic API call timed out after {timeout} seconds")

def fetch_items_from_bigquery(cik):
    """Fetch processed 10-K filings for a specific CIK from BigQuery."""
    query = f"""
        SELECT * FROM `{full_source_table_id}`
        WHERE cik = "0000320193"
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cik", "STRING", cik),
        ]
    )
    query_job = client_bq.query(query, job_config=job_config)
    rows = list(query_job)
    logging.info(f"Fetched {len(rows)} rows for CIK {cik} from BigQuery table {full_source_table_id}")
    return rows

def read_prompt(file_name):
    """Read the prompt from the specified file."""
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

def generate_vertex_embeddings(text):
    """Generate embeddings using Vertex AI."""
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding for Vertex AI.")
        return ""

    max_characters = 1450
    cleaned_text = ' '.join(text[:max_characters].split())
    instances = [{"inputs": cleaned_text}]
    try:
        response = vertex_endpoint.predict(instances=instances)
        embedding = response.predictions[0]
        # Convert embedding list to JSON string
        return json.dumps(embedding)
    except Exception as e:
        logging.error(f"Error generating Vertex AI embeddings: {e}")
        return ""

def generate_openai_embeddings(text):
    """Generate embeddings using OpenAI."""
    text = str(text).strip()
    if text == '' or text.lower() == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding for OpenAI.")
        return ""

    max_characters = 8000  # text-embedding-3-large can handle up to 8191 tokens
    cleaned_text = ' '.join(text[:max_characters].split())
    try:
        response = openai_client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        # Convert embedding list to JSON string
        return json.dumps(embedding)
    except Exception as e:
        logging.error(f"Error generating OpenAI embeddings: {e}")
        return ""

def insert_insights_into_bigquery(row_data):
    """Insert the insights data into BigQuery."""
    try:
        # Define the table reference
        table_ref = client_bq.dataset(dataset_id).table(insights_table_id)
        table = client_bq.get_table(table_ref)

        # Insert the row
        errors = client_bq.insert_rows_json(table, [row_data])
        if errors:
            logging.error(f"Encountered errors while inserting rows: {errors}")
        else:
            logging.info(f"Successfully inserted insights for CIK {row_data['cik']} into {full_insights_table_id}")
    except Exception as e:
        logging.error(f"Failed to insert insights into BigQuery: {e}")

def process_cik_with_anthropic(cik):
    """Process a specific CIK: extract insights, generate embeddings, and insert into BigQuery."""
    # Read the prompt from cik.txt
    prompt_file_name = "cik.txt"
    prompt = read_prompt(prompt_file_name)
    if not prompt:
        logging.warning(f"No prompt available for {prompt_file_name}, skipping processing.")
        return None

    # Log the prompt content for debugging
    logging.info(f"Prompt content being sent to API:\n{prompt}")

    # Fetch items for the specific CIK from BigQuery
    items = fetch_items_from_bigquery(cik)
    if not items:
        logging.info(f"No filings found in the source table for CIK {cik}.")
        return

    # Extract necessary fields from the fetched row
    fetched_row = items[0]
    cik = fetched_row['cik']
    filing_date = fetched_row['filing_date']
    company_name = fetched_row.get('company_name', 'Unknown')  # Adjust field names as per your table
    industry = fetched_row.get('industry', 'Unknown')
    tier = fetched_row.get('tier', 'Unknown')

    # Extract and concatenate all processed_text from processed_items
    processed_items = fetched_row.get('processed_items', [])
    if not processed_items:
        logging.warning(f"No processed_items found for CIK {cik}. Skipping.")
        return

    # Concatenate processed_text with item_name for context
    item_text_parts = []
    for item in processed_items:
        item_name = item.get('item_name', 'Unknown Item')
        processed_text = item.get('processed_text', '')
        if processed_text:
            item_text_parts.append(f"{item_name}: {processed_text}")

    if not item_text_parts:
        logging.warning(f"No processed_text found in processed_items for CIK {cik}. Skipping.")
        return

    item_text = "\n".join(item_text_parts)
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
        response_text = response.content[0].text.strip()  # Get the text part of the response

        # Print the response text (optional)
        print(response_text)

        # Parse the JSON response with sanitization
        try:
            insights = json.loads(response_text)
            logging.info("Successfully parsed JSON response.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            # Attempt to fix common issues by replacing double single quotes with single quotes
            response_text_fixed = response_text.replace("''", "'")
            try:
                insights = json.loads(response_text_fixed)
                logging.info("Successfully parsed fixed JSON response.")
            except json.JSONDecodeError as e_fixed:
                logging.error(f"Failed to parse fixed JSON response: {e_fixed}")
                logging.debug(f"Original Response Text: {response_text}")
                logging.debug(f"Fixed Response Text: {response_text_fixed}")
                return

        # Prepare the row data for BigQuery
        row_data = {
            "cik": cik,
            "filing_date": filing_date.isoformat() if isinstance(filing_date, datetime) else filing_date,
            "company_name": company_name,
            "industry": industry,
            "tier": tier,
            "overallInsights": {}
        }

        # Iterate over each key in overallInsights and map to the schema
        overall_insights = insights.get("overallInsights", {})
        for key, value in overall_insights.items():
            # Initialize the nested dict
            row_data["overallInsights"][key] = {}
            # Concatenate all insights into a single string for embedding
            concatenated_insights = " ".join([
                str(v) for k, v in value.items()
                if "insight" in k or "innovationInsight" in k or "potentialCompetitor" in k or "specificLawsuit" in k
            ])

            if not concatenated_insights.strip():
                logging.warning(f"No insights found for section '{key}'. Skipping embedding generation.")
                embedding_model_1 = None
                embedding_model_2 = None
            else:
                # Generate embeddings using Vertex AI and OpenAI
                logging.info(f"Generating embeddings for section '{key}'.")
                embedding_model_1 = generate_vertex_embeddings(concatenated_insights)
                embedding_model_2 = generate_openai_embeddings(concatenated_insights)
                logging.info(f"Embeddings generated for section '{key}'.")

            # Assign insights
            for sub_key, sub_value in value.items():
                if "embedding" not in sub_key:
                    # Ensure that the value is a string or set to None
                    row_data["overallInsights"][key][sub_key] = sub_value if sub_value else None

            # Assign embeddings
            row_data["overallInsights"][key]["embedding_model_1"] = embedding_model_1 if embedding_model_1 else None
            row_data["overallInsights"][key]["embedding_model_2"] = embedding_model_2 if embedding_model_2 else None

        # Insert the row into BigQuery
        insert_insights_into_bigquery(row_data)

    except Exception as e:
        logging.error(f"Error processing CIK {cik} with Anthropic API: {e}")

# ================================
# Main Function
# ================================

def main(cik):
    # Create the insights_with_embeddings table in BigQuery if it doesn't exist
    create_insights_table()

    # Process the CIK
    process_cik_with_anthropic(cik)

# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    # Example usage: process a specific CIK
    ticker = "HYSR"
    specific_cik = stock_mapper.ticker_to_cik.get(ticker.upper())
    if specific_cik:
        main(specific_cik)
    else:
        logging.error(f"CIK not found for ticker {ticker}")
