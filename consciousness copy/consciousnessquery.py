import os
import argparse
import logging
import json
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import concurrent.futures
from requests.exceptions import SSLError
from urllib3.exceptions import SSLError as URLLib3SSLError
from datetime import datetime
from openai import OpenAI
from google.cloud import aiplatform
import numpy as np

# ================================
# Configuration and Setup
# ================================

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define your project ID, dataset, and table
project_id = 'test1-427219'
dataset_id = 'consciousness'
papers_table_id = 'papers'  # Table containing articles and facts with embeddings

# Full table identifier
full_papers_table_id = f"{project_id}.{dataset_id}.{papers_table_id}"

# Initialize BigQuery client
client_bq = bigquery.Client(project=project_id)

# Initialize Vertex AI
aiplatform.init(project=project_id, location='us-east1')

# Vertex AI Endpoint
vertex_endpoint_name = "projects/test1-427219/locations/us-east1/endpoints/7398750082746548224"
vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_name)

# Initialize OpenAI client (ensure API key is correct and secure)
openai_api_key = "sk-None-WB2bX4Z2lFWuobMohrUuT3BlbkFJZZsmV3aoiqnN1leZJxW9"
openai_client = OpenAI(api_key=openai_api_key)


# ================================
# Function Definitions
# ================================

def generate_vertex_embeddings(text: str) -> list:
    """
    Generate embeddings using Vertex AI.
    Returns a Python list of floats (the embedding vector).
    """
    text = text.strip()
    if not text or text.lower() == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding for Vertex AI.")
        return []

    max_characters = 1450
    cleaned_text = ' '.join(text[:max_characters].split())
    instances = [{"inputs": cleaned_text}]
    try:
        response = vertex_endpoint.predict(instances=instances)
        embedding = response.predictions[0]
        if isinstance(embedding, list):
            return embedding
        else:
            logging.error("Vertex AI returned embedding in unexpected format.")
            return []
    except Exception as e:
        logging.error(f"Error generating Vertex AI embeddings: {e}")
        return []


def generate_openai_embeddings(text: str) -> list:
    """
    Generate embeddings using OpenAI.
    Returns a Python list of floats (the embedding vector).
    """
    text = text.strip()
    if not text or text.lower() == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding for OpenAI.")
        return []

    max_characters = 8000
    cleaned_text = ' '.join(text[:max_characters].split())
    try:
        response = openai_client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Error generating OpenAI embeddings: {e}")
        return []


def fetch_stored_facts() -> list:
    """
    Fetch articles and their fact embeddings from BigQuery, including 'field' and 'link'.
    """
    query = f"""
        SELECT article, field, link, facts
        FROM `{full_papers_table_id}`
    """
    try:
        query_job = client_bq.query(query)
        rows = list(query_job)
        logging.info(f"Fetched {len(rows)} rows from BigQuery: {full_papers_table_id}")
        return rows
    except Exception as e:
        logging.error(f"Error fetching data from BigQuery: {e}")
        return []


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two NumPy vectors."""
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def compute_similarity(query_vec: list, stored_vec: list) -> float:
    """
    Compute cosine similarity between a query embedding and a stored embedding.
    Takes raw Python lists (floats) and converts them to NumPy.
    """
    if not stored_vec:
        return 0.0
    vec_q = np.array(query_vec).flatten()
    vec_s = np.array(stored_vec).flatten()
    return cosine_similarity(vec_q, vec_s)


def parse_facts(facts_record: dict) -> list:
    """
    Parse the 'facts' field, which is assumed to be a RECORD with sub-fields
    like fact1, fact2, etc. Each factN might contain 'embedding_model_1' (Vertex)
    and 'embedding_model_2' (OpenAI), as well as textual content.

    Returns a list of dicts, each containing:
      {
          'fact_key': <'fact1', 'fact2'>,
          'vertex_embedding': [...],
          'openai_embedding': [...],
          'content': {...}  # other textual fields
      }
    """
    if not facts_record:
        return []

    parsed = []
    for fact_key, fact_value in facts_record.items():
        if not isinstance(fact_value, dict):
            continue

        embed_v = fact_value.get('embedding_model_1', '')
        embed_o = fact_value.get('embedding_model_2', '')

        # Convert JSON string -> list
        if isinstance(embed_v, str):
            try:
                embed_v = json.loads(embed_v)
            except json.JSONDecodeError:
                embed_v = []
        if isinstance(embed_o, str):
            try:
                embed_o = json.loads(embed_o)
            except json.JSONDecodeError:
                embed_o = []

        content_fields = {}
        for k, v in fact_value.items():
            if k not in ('embedding_model_1', 'embedding_model_2'):
                content_fields[k] = v

        parsed.append({
            'fact_key': fact_key,
            'vertex_embedding': embed_v if isinstance(embed_v, list) else [],
            'openai_embedding': embed_o if isinstance(embed_o, list) else [],
            'content': content_fields
        })

    return parsed


def find_similar_facts(query: str, top_n: int = 15) -> list:
    """
    Generate embeddings for the query, fetch stored facts from BigQuery,
    compute similarity (Vertex + OpenAI) to each fact, and return top-N.
    Now includes 'field' and 'link' in the final results.
    """
    logging.info("Generating embeddings for the query...")
    query_vertex = generate_vertex_embeddings(query)
    query_openai = generate_openai_embeddings(query)

    if not query_vertex and not query_openai:
        logging.error("Failed to generate both Vertex AI and OpenAI embeddings for the query.")
        return []

    use_vertex = bool(query_vertex)
    use_openai = bool(query_openai)

    if use_vertex and use_openai:
        logging.info("Both Vertex AI and OpenAI embeddings generated successfully.")
    elif use_vertex:
        logging.info("Only Vertex AI embeddings generated successfully.")
    elif use_openai:
        logging.info("Only OpenAI embeddings generated successfully.")

    # Fetch stored data from BigQuery
    stored_rows = fetch_stored_facts()
    if not stored_rows:
        return []

    similarity_scores = []

    # Iterate through each row (each article)
    for row in stored_rows:
        article_text = row.get('article', '')
        row_field = row.get('field', '')
        row_link = row.get('link', '')
        facts_record = row.get('facts', {})

        # Parse each fact from the row
        parsed_facts = parse_facts(facts_record)
        for fact_data in parsed_facts:
            sim_vertex = sim_openai = 0.0

            # Compute similarities if embeddings are available
            if use_vertex:
                sim_vertex = compute_similarity(query_vertex, fact_data['vertex_embedding'])
            if use_openai:
                sim_openai = compute_similarity(query_openai, fact_data['openai_embedding'])

            if use_vertex and use_openai:
                combined_similarity = (sim_vertex + sim_openai) / 2
            elif use_vertex:
                combined_similarity = sim_vertex
            elif use_openai:
                combined_similarity = sim_openai
            else:
                combined_similarity = 0.0

            similarity_scores.append({
                'article': article_text,
                'field': row_field,
                'link': row_link,     # store full link
                'fact_key': fact_data['fact_key'],
                'similarity': combined_similarity,
                'content': fact_data['content']
            })

    if not similarity_scores:
        logging.info("No embeddings found to compare.")
        return []

    # Sort descending by similarity
    sorted_scores = sorted(similarity_scores, key=lambda x: x['similarity'], reverse=True)

    return sorted_scores[:top_n]


def display_results(results: list):
    """
    Print the entire link in the 'Link' column. Adjust the column widths
    to accommodate a longer link. We also show 'Field' and the content.
    """
    if not results:
        print("No similar facts found.")
        return

    # Headers
    print(f"{'Article (Snippet)':<40} {'Field':<20} {'Link':<60} {'FactKey':<10} {'Similarity':<10} {'Content'}")
    print("-" * 180)

    for res in results:
        article_snip = (res['article'][:35] + '...') if len(res['article']) > 35 else res['article']
        field_val = res.get('field', '')
        link_val = res.get('link', '')
        fact_key = res['fact_key']
        similarity = f"{res['similarity']:.4f}"

        # Content is a dict with textual data (Statement/Evidence/etc.)
        content_str = ' | '.join(f"{k}: {v}" for k, v in res['content'].items() if v)

        print(f"{article_snip:<40} {field_val:<20} {link_val:<60} {fact_key:<10} {similarity:<10} {content_str}")


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(description="Find similar facts based on a query.")
    parser.add_argument('--query', type=str, help='The query string to search for similar facts.', required=False)
    parser.add_argument('--top_n', type=int, help='Number of top similar facts to retrieve.', default=10)
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Enter your query: ")

    top_n = args.top_n
    logging.info(f"Processing query: {query}")

    similar_facts = find_similar_facts(query, top_n)

    display_results(similar_facts)


# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    main()
