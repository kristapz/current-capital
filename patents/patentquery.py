import os
import numpy as np
from google.cloud import bigquery
from google.cloud import aiplatform
from scipy.spatial.distance import cosine
from anthropic import AnthropicVertex

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = '...'
client_bq = bigquery.Client(project=project_id)
aiplatform.init(project=project_id, location='us-east1')

# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id="")

def generate_embeddings(text):
    endpoint = aiplatform.Endpoint(
        endpoint_name=""
    )
    response = endpoint.predict(instances=[{"inputs": text}])
    embeddings = response.predictions[0]
    return np.array(embeddings[0]) if isinstance(embeddings[0], list) else np.array(embeddings)

def fetch_patent_embeddings():
    query = """
    SELECT publication_number, abstract, embeddings
    FROM `..embedded_patents`
    LIMIT 1000
    """
    query_job = client_bq.query(query)
    return [(row.publication_number, row.abstract, np.array(row.embeddings)) for row in query_job]

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def find_similar_patents(query_text):
    query_embedding = generate_embeddings(query_text)
    patents = fetch_patent_embeddings()
    similarities = [(pub_number, abstract, cosine_similarity(query_embedding, embedding)) for pub_number, abstract, embedding in patents]
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:5]

def summarize_abstracts(abstracts, query_text):
    model_name = "claude-3-5-sonnet@20240620"
    prompt_text = f"Considering the user query: '{query_text}', summarize the following abstracts: " + " ".join(abstracts)
    response = client_anthropic.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt_text}],
        model=model_name
    )
    return response.content[0].text

def main():
    query_text = input("Enter your query text here: ")
    top_patents = find_similar_patents(query_text)
    print(top_patents)
    abstracts = [abstract for _, abstract, _ in top_patents]
    summary = summarize_abstracts(abstracts, query_text)
    print("Summary of top 5 patent abstracts:", summary)

main()
