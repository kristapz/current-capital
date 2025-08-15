import os
import logging
import json
import numpy as np
import pandas as pd

from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Configuration
# -------------------------------
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ID = "test1-427219"
DATASET_ID = "consciousness"
TABLE_ID = "papers"
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=PROJECT_ID)


# -------------------------------
# Data Fetching
# -------------------------------
def fetch_papers_with_embeddings():
    """
    Fetches articles, facts, links, and fields from the BigQuery table.
    """
    query = f"""
        SELECT
          article,
          facts,
          link,
          field
        FROM `{FULL_TABLE_ID}`
    """
    try:
        query_job = client_bq.query(query)
        rows = list(query_job)
        logging.info(f"Fetched {len(rows)} rows from {FULL_TABLE_ID}.")
        return rows
    except Exception as e:
        logging.error(f"Failed to fetch data from BigQuery: {e}")
        return []


def parse_embeddings(rows):
    """
    Parses each row to extract embeddings (from 'embedding_model_2') and metadata:
      - article
      - field
      - link
      - statement + evidence
      - fact_key
    Returns a list of dicts.
    """
    results = []
    for row in rows:
        article_text = row.get('article', '')
        facts_dict = row.get('facts', {})
        link_val = row.get('link', '')
        field_val = row.get('field', '')

        if not isinstance(facts_dict, dict):
            continue

        for fact_key, fact_value in facts_dict.items():
            if not isinstance(fact_value, dict):
                continue

            embedding_json = fact_value.get('embedding_model_2', None)
            if not embedding_json:
                continue

            try:
                embedding = json.loads(embedding_json)  # list of floats
            except json.JSONDecodeError:
                embedding = []

            statement = fact_value.get('Statement', '')
            evidence = fact_value.get('Evidence', '')

            if embedding:
                results.append({
                    'article': article_text.strip(),
                    'field': field_val.strip(),
                    'link': link_val.strip(),
                    'fact_key': fact_key.strip(),
                    'statement': statement.strip(),
                    'evidence': evidence.strip(),
                    'embedding': np.array(embedding, dtype=float),
                })
    logging.info(f"parse_embeddings: Found {len(results)} total items with valid embeddings.")
    return results


# -------------------------------
# Centroid Computation with Grouping
# -------------------------------
def compute_field_centroids(parsed_data, field_group_mapping):
    """
    Computes the centroid (average embedding) for each field or grouped fields.
    - field_group_mapping: dict mapping original fields to new grouped fields.

    Returns a dictionary mapping new field names to their centroid embeddings.
    """
    grouped_embeddings = {}
    for item in parsed_data:
        original_field = item['field']
        grouped_field = field_group_mapping.get(original_field,
                                                original_field)  # Default to original field if not grouped
        embedding = item['embedding']
        if grouped_field not in grouped_embeddings:
            grouped_embeddings[grouped_field] = []
        grouped_embeddings[grouped_field].append(embedding)

    field_centroids = {}
    for field, embeddings in grouped_embeddings.items():
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            field_centroids[field] = centroid
            logging.info(f"Computed centroid for field '{field}' with {len(embeddings)} embeddings.")
        else:
            logging.warning(f"No embeddings found for field '{field}'.")

    return field_centroids


# -------------------------------
# Similarity Computation
# -------------------------------
def compute_centroid_similarities(field_centroids):
    """
    Computes the cosine similarity between each pair of field centroids.
    Returns a DataFrame representing the similarity matrix.
    """
    fields = list(field_centroids.keys())
    centroids = np.vstack([field_centroids[field] for field in fields])

    similarity_matrix = cosine_similarity(centroids)
    df_similarity = pd.DataFrame(similarity_matrix, index=fields, columns=fields)

    logging.info("Computed cosine similarity matrix for field centroids.")
    return df_similarity


# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Define the fields to be grouped
    fields_to_group = [
        "Cognitive Neuroscience",
        "Neurobiology",
        "Physical Theories of Consciousness",
        "Psychedelics and Altered States of Consciousness"
    ]
    grouped_field_name = "Cognitive & Biological Sciences"

    # Create a mapping from original fields to grouped fields
    field_group_mapping = {}
    for field in fields_to_group:
        field_group_mapping[field] = grouped_field_name

    # Fetch data from BigQuery
    rows = fetch_papers_with_embeddings()
    if not rows:
        logging.error("No data retrieved from BigQuery. Exiting.")
        return

    # Parse embeddings and metadata
    parsed = parse_embeddings(rows)
    if not parsed:
        logging.error("No valid embeddings found in the data. Exiting.")
        return

    # Compute centroids for each field, with grouping
    field_centroids = compute_field_centroids(parsed, field_group_mapping)
    if not field_centroids:
        logging.error("No field centroids computed. Exiting.")
        return

    # Compute similarity matrix between field centroids
    similarity_df = compute_centroid_similarities(field_centroids)

    # Output the similarity matrix
    print("Cosine Similarity Matrix Between Field Centroids:")
    print(similarity_df)

    # Optionally, save to CSV
    output_csv = "field_centroid_similarities_grouped.csv"
    similarity_df.to_csv(output_csv)
    logging.info(f"Similarity matrix saved to {output_csv}.")


if __name__ == "__main__":
    main()
