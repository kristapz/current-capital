import os
import time
import random
import logging
import json
from google.cloud import bigquery
import openai
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from requests.exceptions import RequestException
from datetime import datetime
from google.cloud import aiplatform
from typing import List
from sec_cik_mapper import StockMapper
import re  # For regex JSON extraction
from openai import OpenAI

# ================================
# Configuration and Setup
# ================================

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'  # Update the path accordingly

# Define your project ID and dataset ID
project_id = 'test1-427219'
dataset_id = 'consciousness'
source_table_id = 'papers'
insights_table_id = 'papers'

# Full table identifiers
full_source_table_id = f"{project_id}.{dataset_id}.{source_table_id}"
full_insights_table_id = f"{project_id}.{dataset_id}.{insights_table_id}"
stock_mapper = StockMapper()

# Initialize BigQuery client
client_bq = bigquery.Client(project=project_id)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY", "sk-None-WB2bX4Z2lFWuobMohrUuT3BlbkFJZZsmV3aoiqnN1leZJxW9")  # Ensure this API key is correct and secure
openai.api_key = openai_api_key
client = OpenAI(api_key=openai_api_key)

# Directory with prompts
prompt_directory = 'prompts'  # Adjust this as needed based on the actual location

# ================================
# Initialize Embedding Clients
# ================================

vertex_endpoint_name = os.getenv(
    "VERTEX_ENDPOINT_NAME",
    "projects/test1-427219/locations/us-east1/endpoints/7398750082746548224"
)  # Replace with your actual endpoint or set as environment variable
aiplatform.init(project=project_id, location='us-east1')
vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_name)

# ================================
# Define BigQuery Schema
# ================================
# Updated schema to include 'row_number', 'link', and 'field'
insights_schema = [
    bigquery.SchemaField("article", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("facts", "RECORD", mode="REQUIRED", fields=[
        bigquery.SchemaField(f"fact{n}", "RECORD", mode="NULLABLE", fields=[
            bigquery.SchemaField("Statement", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("Evidence", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("Context", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_1", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model_2", "STRING", mode="NULLABLE"),
        ]) for n in range(1, 21)
    ]),
    bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("row_number", "INTEGER", mode="NULLABLE"),  # New column
    bigquery.SchemaField("link", "STRING", mode="NULLABLE"),          # New column
    bigquery.SchemaField("field", "STRING", mode="NULLABLE")          # New column
]

# ================================
# Function Definitions
# ================================

def create_insights_table():
    """Create the papers table in BigQuery if it doesn't exist (includes new columns)."""
    table_ref = client_bq.dataset(dataset_id).table(insights_table_id)
    try:
        client_bq.get_table(table_ref)  # Check if table exists
        logging.info(f"Table {full_insights_table_id} already exists.")
    except google_exceptions.NotFound:
        table = bigquery.Table(table_ref, schema=insights_schema)
        client_bq.create_table(table)
        logging.info(f"Created table {full_insights_table_id} in BigQuery with updated schema.")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RequestException, Exception))
)
def retry_openai_call(func, *args, **kwargs):
    """Retry mechanism for OpenAI API calls."""
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


def read_prompt(file_name):
    """Read the prompt (or text/link) from the specified file in the prompts directory."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, prompt_directory, file_name)
        with open(prompt_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            logging.info(f"Successfully read content from {file_name}")
            return content
    except Exception as e:
        logging.error(f"Failed to read content from {file_name}: {e}")
        return ""


def extract_json(response_text):
    """Extract JSON object from a string using regex."""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json_str
        else:
            logging.error("No JSON object found in the response.")
            return None
    except Exception as e:
        logging.error(f"Error extracting JSON: {e}")
        return None


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
        return json.dumps(embedding)  # Convert list to JSON string
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
        response = client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        return json.dumps(embedding)  # Convert list to JSON string
    except Exception as e:
        logging.error(f"Error generating OpenAI embeddings: {e}")
        return ""


def insert_insights_into_bigquery(row_data):
    """Insert the insights data into BigQuery."""
    try:
        table_ref = client_bq.dataset(dataset_id).table(insights_table_id)
        table = client_bq.get_table(table_ref)

        errors = client_bq.insert_rows_json(table, [row_data])
        if errors:
            logging.error(f"Encountered errors while inserting rows: {errors}")
        else:
            logging.info(f"Successfully inserted row into {full_insights_table_id}")
    except Exception as e:
        logging.error(f"Failed to insert insights into BigQuery: {e}")


def validate_row_data(row_data):
    """Validate the row data before insertion."""
    if not isinstance(row_data.get("article"), str):
        logging.error("Invalid data type for 'article'. Expected string.")
        return False
    if not isinstance(row_data.get("facts"), dict):
        logging.error("Invalid data type for 'facts'. Expected dict.")
        return False
    # Check if at least one fact is present
    if not any(row_data["facts"].values()):
        logging.error("No facts present in the row data.")
        return False
    return True


def get_next_row_number():
    """
    Retrieve the current max row_number from the table and return the next one.
    If no existing rows or row_number is NULL, start at 17 (because there are already 16 rows).
    """
    try:
        query = f"SELECT MAX(row_number) as max_row_number FROM `{full_insights_table_id}`"
        query_job = client_bq.query(query)
        results = query_job.result()
        max_row_num = None
        for row in results:
            max_row_num = row.max_row_number

        if max_row_num is None:
            max_row_num = 16  # Start numbering at 17 if no rows or no row_number found

        return max_row_num + 1
    except Exception as e:
        logging.error(f"Error retrieving max row_number: {e}")
        # Fallback to 17 if query fails for some reason
        return 17


def classify_field(article):
    """Classify the article into a specific field using OpenAI's o1-mini model and field.txt prompt."""
    # Read the classification prompt from field.txt
    field_prompt_file = "field.txt"
    field_prompt = read_prompt(field_prompt_file)
    if not field_prompt:
        logging.error(f"No classification prompt available in {field_prompt_file}.")
        return ""

    # Combine the classification prompt with the article text
    classification_prompt = f"{field_prompt}\n\n{article}"
    logging.info("Sending classification prompt to OpenAI API for field categorization...")

    try:
        response = retry_openai_call(
            client.chat.completions.create,
            model="o1-mini",  # Retain the exact model name
            messages=[
                {"role": "user", "content": classification_prompt}
            ],
        )
        logging.info(f"Received response from OpenAI API for field classification.")
        response_text = response.choices[0].message.content.strip()

        # Log the raw response
        logging.debug(f"Raw classification response:\n{response_text}")

        # Extract JSON from the response_text using regex
        json_str = extract_json(response_text)
        if not json_str:
            logging.error("No JSON object found in the field classification response.")
            return ""

        # Parse the JSON response
        try:
            classification_data = json.loads(json_str)
            logging.info("Successfully parsed JSON response for field classification.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response for field classification: {e}")
            logging.debug(f"JSON String: {json_str}")
            return ""

        # Extract the 'field' value
        classified_field = classification_data.get("field", "")
        if not classified_field:
            logging.error("No 'field' key found in the classification JSON.")
            return ""

        # Validate the classified field against allowed categories
        allowed_fields = [
            "Philosophy of Mind",
            "Neurobiology",
            "Cognitive Neuroscience",
            "Psychology and Experimental Research",
            "Artificial Intelligence and Computational Models",
            "Psychedelics and Altered States of Consciousness",
            "Quantum Theories of Consciousness",
            "Cultural, Spiritual, and Anthropological Perspectives"
        ]

        if classified_field not in allowed_fields:
            logging.warning(f"Classified field '{classified_field}' is not in the allowed categories.")
            # Optionally, handle this case as needed
            return classified_field  # Or return an empty string or default category

        return classified_field

    except Exception as e:
        logging.error(f"Error during field classification: {e}")
        return ""


def process_article_with_openai(article, link):
    """Process the input article, incorporate the link, generate embeddings, classify field, and insert into BigQuery."""
    # Read the prompt from prompt.txt
    prompt_file_name = "prompt.txt"
    prompt = read_prompt(prompt_file_name)
    if not prompt:
        logging.warning(f"No prompt available for {prompt_file_name}, skipping processing.")
        return None

    # Log the prompt content (debugging only)
    logging.info(f"Prompt content being sent to API:\n{prompt}")
    logging.info(f"Article text being processed:\n{article}")
    logging.info(f"Link text being processed:\n{link}")

    # Combine the prompt and article text
    # IMPORTANT: Per your instructions, we do NOT alter the prompt or the models in any way
    full_prompt = f"{prompt}\n\n{article}"
    logging.info(f"Sending prompt to OpenAI API with model='o1-mini'")

    try:
        response = retry_openai_call(
            client.chat.completions.create,
            model="o1-mini",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
        )
        logging.info(f"Received response from OpenAI API")
        response_text = response.choices[0].message.content.strip()

        # Log the raw response
        logging.debug(f"Raw response from OpenAI API:\n{response_text}")

        # Extract JSON from the response_text using regex
        json_str = extract_json(response_text)
        if not json_str:
            logging.error("No JSON object found in the API response. Skipping insertion.")
            return

        # Parse the JSON response
        try:
            insights = json.loads(json_str)
            logging.info("Successfully parsed JSON response.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            logging.debug(f"JSON String: {json_str}")
            return

        # Classify the field using the 'field.txt' prompt and OpenAI
        classified_field = classify_field(article)
        if not classified_field:
            logging.error("Field classification failed. 'field' will be set to None.")
            classified_field = None

        # Prepare the row data for BigQuery
        row_data = {
            "article": article,
            "facts": {},
            "processed_at": datetime.utcnow().isoformat(),
            # Incorporate link and row_number here
            "link": link if link else None,
            "row_number": get_next_row_number(),
            "field": classified_field
        }

        # Adjusted to match the structure of the JSON response
        extracted_facts = insights
        if not extracted_facts:
            logging.error("No facts extracted from the response.")
            return

        fact_number = 1  # Initialize fact counter
        for key, value in extracted_facts.items():
            if fact_number > 20:
                logging.warning(f"Exceeded 20 facts. Ignoring additional facts.")
                break

            # Initialize the nested dict for the current fact
            fact_key = f"fact{fact_number}"
            row_data["facts"][fact_key] = {}

            # Concatenate fields for embedding
            concatenated_facts = " ".join([
                str(v) for k, v in value.items()
                if k in ["Statement", "Evidence", "Context"]
            ])

            if not concatenated_facts.strip():
                logging.warning(f"No relevant facts found for section '{key}'. Skipping embedding generation.")
                embedding_model_1 = None
                embedding_model_2 = None
            else:
                # Generate embeddings
                logging.info(f"Generating embeddings for {fact_key}.")
                embedding_model_1 = generate_vertex_embeddings(concatenated_facts)
                embedding_model_2 = generate_openai_embeddings(concatenated_facts)
                logging.info(f"Embeddings generated for {fact_key}.")

            # Assign facts
            for sub_key, sub_value in value.items():
                if sub_key not in ["embedding_model_1", "embedding_model_2"]:
                    row_data["facts"][fact_key][sub_key] = str(sub_value) if sub_value else None

            # Assign embeddings
            row_data["facts"][fact_key]["embedding_model_1"] = embedding_model_1 if embedding_model_1 else None
            row_data["facts"][fact_key]["embedding_model_2"] = embedding_model_2 if embedding_model_2 else None

            fact_number += 1

        # Validate row data before insertion
        if validate_row_data(row_data):
            insert_insights_into_bigquery(row_data)
        else:
            logging.error("Row data validation failed. Skipping insertion.")

    except Exception as e:
        logging.error(f"Error processing article with OpenAI API: {e}")


# ================================
# Main Function
# ================================

def main():
    # Create/Verify the papers table in BigQuery (new columns included)
    create_insights_table()

    # Read article text from article.txt
    article_file_name = "article.txt"
    article = read_prompt(article_file_name)
    if not article:
        logging.error("No article text provided. Exiting.")
        return
    logging.info(f"Article text read from {article_file_name}.")

    # Read link text from link.txt
    link_file_name = "link.txt"
    link = read_prompt(link_file_name)
    if not link:
        logging.warning("No link provided. Continuing without link (will insert empty).")
        link = ""

    # Process the article (and link)
    process_article_with_openai(article, link)


# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    main()
