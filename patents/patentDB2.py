import os
from google.cloud import bigquery
from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPIError

# Setup
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '...'
project_id = '...'
dataset_id = 'test1Dataset'
table_id = 'embedded_patents'
client = bigquery.Client(project=project_id)

# Initialize Vertex AI
aiplatform.init(project=project_id, location='us-east1')

def generate_embeddings(text):
    if not text:  # Check if the text is empty or None
        print("Warning: Empty input text provided, using default placeholder.")
        text = "default placeholder text"  # Use a default placeholder text if necessary
    endpoint = aiplatform.Endpoint(
        endpoint_name=""
    )
    instances = [{"inputs": text}]
    response = endpoint.predict(instances=instances)
    embeddings = response.predictions[0]
    # Ensure embeddings are a flat list of floats
    if isinstance(embeddings, list) and all(isinstance(x, list) for x in embeddings):
        embeddings = [item for sublist in embeddings for item in sublist]  # Flatten the list
    return embeddings

# Fetch data and prepare for insertion
query = """
SELECT publication_number, title, abstract, ARRAY(SELECT 'sample_term') as top_terms, url, ARRAY(SELECT 'sample_citation') as cited_by
FROM `bigquery-public-data.google_patents_research.publications`
LIMIT 10
"""
query_job = client.query(query)
results = query_job.result()

rows_to_insert = []
for row in results:
    embeddings = generate_embeddings(row.abstract)
    row_data = {
        "publication_number": row.publication_number,
        "title": row.title,
        "abstract": row.abstract,
        "top_terms": [term for term in row.top_terms],  # Convert ARRAY<STRUCT> to ARRAY<STRING>
        "url": row.url,
        "cited_by": [citation for citation in row.cited_by],  # Convert ARRAY<STRUCT> to ARRAY<STRING>
        "embeddings": embeddings  # Ensure embeddings are correctly formatted
    }
    rows_to_insert.append(row_data)

# Print a sample row to check data structure
print("Sample row to be inserted:", rows_to_insert[0])

# Insert data
full_table_id = f"{project_id}.{dataset_id}.{table_id}"
errors = client.insert_rows_json(full_table_id, rows_to_insert)
if errors:
    print("Errors occurred while inserting rows:", errors)
else:
    print("Data inserted successfully.")
