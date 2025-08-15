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
from sec_cik_mapper import StockMapper

# ================================
# Configuration and Setup
# ================================
# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define your project ID and dataset ID
project_id = 'test1-427219'
dataset_id = 'backwards_testing'
insights_table_id = 'insights_with_embeddings'  # Table containing insights and embeddings

# Full table identifier
full_insights_table_id = f"{project_id}.{dataset_id}.{insights_table_id}"
stock_mapper = StockMapper()

# Initialize BigQuery client
client_bq = bigquery.Client(project=project_id)

# Initialize Vertex AI
aiplatform.init(project=project_id, location='us-east1')

# Vertex AI Endpoint
vertex_endpoint_name = "4"  # Replace with your actual endpoint
vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_name)

# Initialize OpenAI client
openai_api_key = ""  # Ensure this API key is correct and secure
openai_client = OpenAI(api_key=openai_api_key)


# ================================
# Function Definitions
# ================================

def generate_vertex_embeddings(text: str) -> list:
    """Generate embeddings using Vertex AI."""
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
    """Generate embeddings using OpenAI."""
    text = text.strip()
    if not text or text.lower() == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding for OpenAI.")
        return []

    max_characters = 8000  # Adjust based on the model's token limit
    cleaned_text = ' '.join(text[:max_characters].split())
    try:
        response = openai_client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        # Return the embedding list directly
        return embedding
    except Exception as e:
        logging.error(f"Error generating OpenAI embeddings: {e}")
        return []


def fetch_stored_embeddings() -> list:
    """Fetch all stored embeddings from BigQuery."""
    query = f"""
        SELECT cik, filing_date, company_name, industry, tier, overallInsights
        FROM `{full_insights_table_id}`
    """
    try:
        query_job = client_bq.query(query)
        rows = list(query_job)
        logging.info(f"Fetched {len(rows)} rows from BigQuery.")
        return rows
    except Exception as e:
        logging.error(f"Error fetching data from BigQuery: {e}")
        return []


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def compute_similarity_vertex(query_vertex: list, stored_vertex: list) -> float:
    """Compute cosine similarity between query Vertex AI embedding and stored Vertex AI embedding."""
    if not stored_vertex:
        return 0.0
    vec_q_v = np.array(query_vertex).flatten()
    vec_s_v = np.array(stored_vertex).flatten()
    return cosine_similarity(vec_q_v, vec_s_v)


def compute_similarity_openai(query_openai: list, stored_openai: list) -> float:
    """Compute cosine similarity between query OpenAI embedding and stored OpenAI embedding."""
    if not stored_openai:
        return 0.0
    vec_q_o = np.array(query_openai).flatten()
    vec_s_o = np.array(stored_openai).flatten()
    return cosine_similarity(vec_q_o, vec_s_o)


def parse_embeddings(overallInsights: dict) -> list:
    """
    Parse the embeddings and insights from the overallInsights field.
    Returns a list of embeddings per section with content.
    """
    embeddings = []
    for section, content in overallInsights.items():
        embed_v = content.get('embedding_model_1')
        embed_o = content.get('embedding_model_2')
        if embed_v and embed_o:
            try:
                embed_v = json.loads(embed_v)
                embed_o = json.loads(embed_o)
                # Extract all insights except embedding fields
                insights_content = {k: v for k, v in content.items() if not k.startswith('embedding_model_') and v}
                embeddings.append({
                    'section': section,
                    'embedding_vertex': embed_v,
                    'embedding_openai': embed_o,
                    'content': insights_content
                })
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding embeddings for section {section}: {e}")
    return embeddings


def find_similar_insights(query: str, top_n: int = 2) -> list:
    """Find top-N similar insights to the query."""
    # Generate embeddings for the query
    logging.info("Generating embeddings for the query.")
    query_vertex_embedding = generate_vertex_embeddings(query)
    query_openai_embedding = generate_openai_embeddings(query)

    # Parse query OpenAI embedding if it's a JSON string
    if isinstance(query_openai_embedding, str):
        try:
            query_openai_embedding = json.loads(query_openai_embedding)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding query OpenAI embedding: {e}")
            query_openai_embedding = []

    if not query_vertex_embedding or not query_openai_embedding:
        logging.error("Failed to generate one or both embeddings for the query.")
        return []

    # Fetch stored embeddings
    stored_rows = fetch_stored_embeddings()
    if not stored_rows:
        logging.error("No data fetched from BigQuery.")
        return []

    # Prepare a list to hold similarity scores
    similarity_scores = []

    # Iterate through stored rows and compute similarity for each section
    for row in stored_rows:
        cik = row['cik']
        filing_date = row['filing_date']
        company_name = row['company_name']
        industry = row['industry']
        tier = row['tier']
        overallInsights = row['overallInsights']

        # Parse embeddings from JSON
        embeddings = parse_embeddings(overallInsights)

        # Compute similarity for each section and collect scores
        for embed in embeddings:
            sim_v = compute_similarity_vertex(query_vertex_embedding, embed['embedding_vertex'])
            sim_o = compute_similarity_openai(query_openai_embedding, embed['embedding_openai'])
            # Combine the two similarities, e.g., by averaging
            combined_similarity = (sim_v + sim_o) / 2
            similarity_scores.append({
                'cik': cik,
                'filing_date': filing_date,
                'company_name': company_name,
                'industry': industry,
                'tier': tier,
                'section': embed['section'],
                'similarity': combined_similarity,
                'content': embed['content']
            })

    if not similarity_scores:
        logging.info("No embeddings found to compare.")
        return []

    # Sort the results based on similarity
    sorted_scores = sorted(similarity_scores, key=lambda x: x['similarity'], reverse=True)

    # Return top-N results
    return sorted_scores[:top_n]


def display_results(results: list):
    """Display the similarity results."""
    if not results:
        print("No similar insights found.")
        return
    print(
        f"{'CIK':<15} {'Filing Date':<20} {'Company Name':<30} {'Industry':<20} {'Tier':<10} {'Section':<40} {'Similarity':<10} {'Content'}")
    print("-" * 200)
    for res in results:
        cik = res['cik']
        filing_date = res['filing_date']
        company_name = res['company_name']
        industry = res['industry']
        tier = res['tier']
        section = res['section']
        similarity = res['similarity']
        content = res['content']
        # Concatenate all insight texts
        content_text = ' | '.join([f"{k}: {v}" for k, v in content.items() if v])
        print(
            f"{cik:<15} {filing_date:<20} {company_name:<30} {industry:<20} {tier:<10} {section:<40} {similarity:<10.4f} {content_text}")


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(description="Find similar insights based on a query.")
    parser.add_argument('--query', type=str, help='The query string to search for similar insights.', required=False)
    parser.add_argument('--top_n', type=int, help='Number of top similar insights to retrieve.', default=2)
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Enter your query: ")

    top_n = args.top_n

    logging.info(f"Processing query: {query}")

    similar_insights = find_similar_insights(query, top_n)

    display_results(similar_insights)


# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    main()
