import os
import logging
import json
import openai
from google.cloud import bigquery
from datetime import datetime
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re
import random
import time
from requests.exceptions import RequestException

# -------------------------------
# Configuration and Setup
# -------------------------------
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = "test1-427219"
DATASET_ID = "consciousness"
TABLE_ID = "papers"
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize BigQuery
client_bq = bigquery.Client(project=PROJECT_ID)

# Initialize OpenAI (same logic as your existing code)
openai_api_key = os.getenv("OPENAI_API_KEY", "sk-None-WB2bX4Z2lFWuobMohrUuT3BlbkFJZZsmV3aoiqnN1leZJxW9")
openai.api_key = openai_api_key
client = OpenAI(api_key=openai_api_key)

# Path to the summary prompt
PROMPT_DIRECTORY = "/Users/kristapszilgalvis/Desktop/current-capital/consciousness/prompts"
SUMMARY_PROMPT_FILE = "mainsummary.txt"

# -------------------------------
# Retry Decorator for OpenAI calls
# -------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RequestException, Exception))
)
def retry_openai_call(func, *args, **kwargs):
    """
    Retry mechanism for OpenAI API calls.
    Exactly matches your existing approach to ensure consistent behavior.
    """
    try:
        start_time = time.time()
        response = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds")
        return response
    except Exception as e:
        logging.warning(f"API call failed: {e}, retrying...")
        time.sleep(random.uniform(1, 5))
        raise

# -------------------------------
# Utility to read the summary prompt
# -------------------------------
def read_prompt_file(file_path):
    """
    Read the summary prompt from mainsummary.txt
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        logging.info(f"Loaded summary prompt from {file_path}")
        return text
    except Exception as e:
        logging.error(f"Failed to read summary prompt: {e}")
        return ""

# -------------------------------
# Extract JSON helper (optional, only if needed)
# -------------------------------
def extract_json(text):
    """
    If the model returns JSON in a longer text, we can parse it using regex.
    Not strictly necessary if you only want the text.
    But provided for consistency with your pattern.
    """
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        logging.debug("No JSON block found in text.")
        return None
    except Exception as e:
        logging.error(f"extract_json error: {e}")
        return None

# -------------------------------
# Main Summaries
# -------------------------------
def generate_summaries():
    """
    1. Query BigQuery for row_number, link, field, facts -> Statement, Evidence, Context
    2. For each row, build a text snippet + summary prompt
    3. Call the 'o1-mini' model
    4. Print the summary (and link) to the console
    """

    # 1. Query BigQuery
    query = f"""
        SELECT 
          row_number,
          link,
          field,
          facts
        FROM `{FULL_TABLE_ID}`
        ORDER BY row_number ASC
    """
    rows = []
    try:
        logging.info("Fetching data from BigQuery...")
        query_job = client_bq.query(query)
        rows = list(query_job)
        logging.info(f"Fetched {len(rows)} rows from {FULL_TABLE_ID}.")
    except Exception as e:
        logging.error(f"Error querying BigQuery: {e}")
        return

    if not rows:
        logging.warning("No data found in the table.")
        return

    # 2. Load summary prompt from mainsummary.txt
    prompt_path = os.path.join(PROMPT_DIRECTORY, SUMMARY_PROMPT_FILE)
    summary_prompt = read_prompt_file(prompt_path)
    if not summary_prompt:
        logging.warning("No summary prompt found, can't proceed.")
        return

    # 3. For each row, build the snippet of facts
    for row in rows:
        row_num = row.get("row_number")
        link_val = row.get("link", "")
        field_val = row.get("field", "")
        facts_dict = row.get("facts", {})

        if not isinstance(facts_dict, dict):
            continue

        facts_text = ""
        for fact_key, fact_value in facts_dict.items():
            if not isinstance(fact_value, dict):
                continue
            statement = fact_value.get("Statement", "")
            evidence = fact_value.get("Evidence", "")
            context = fact_value.get("Context", "")
            if any([statement, evidence, context]):
                facts_text += f"\n- Fact Key: {fact_key}\n"
                facts_text += f"  Statement: {statement}\n"
                facts_text += f"  Evidence: {evidence}\n"
                facts_text += f"  Context: {context}\n"

        if not facts_text.strip():
            continue

        full_prompt = f"{summary_prompt}\n\nField: {field_val}\nLink: {link_val}\nFacts:\n{facts_text}"

        # 4. Call 'o1-mini' model EXACTLY
        try:
            response = retry_openai_call(
                client.chat.completions.create,
                model="o1-mini",
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
            )
            response_text = response.choices[0].message.content.strip()

            # Print summary (and link) to console
            print(f"\n===== Summary for row_number={row_num} =====")
            print(f"Link: {link_val}")
            print(f"Summary:\n{response_text}\n")

        except Exception as e:
            logging.error(f"OpenAI call failed for row_number={row_num}: {e}")

# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    generate_summaries()
