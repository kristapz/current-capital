from flask import Flask, render_template, request
import os
import numpy as np
import json
from google.cloud import bigquery
from google.cloud import aiplatform
from scipy.spatial.distance import cosine
from anthropic import AnthropicVertex
import re
import yfinance as yf
from datetime import datetime
import time
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery import ScalarQueryParameter, StructQueryParameter

app = Flask(__name__)

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
client_bq = bigquery.Client(project=project_id)
aiplatform.init(project=project_id, location='us-east1')

# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id=project_id)

def cosine_similarity(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return 1 - cosine(v1, v2)

def fetch_company_data():
    query = """
    SELECT ticker, long_business_summary, embeddings, name, sector
    FROM .stock_datasets.stocksv2
    """
    query_job = client_bq.query(query)
    return [(row.ticker, row.long_business_summary, np.array(json.loads(row.embeddings)).flatten(),
             row.name, row.sector) for row in query_job]

def fetch_most_recent_article():
    query = """
    SELECT id, date, ARRAY_TO_STRING(content, ' ') as content, embeddings.model1
    FROM `test1-427219.analysis_dataset.analysis`
    WHERE date IS NOT NULL
    ORDER BY PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) DESC
    LIMIT 1
    """
    query_job = client_bq.query(query)
    result = list(query_job.result())
    if result:
        row = result[0]
        return {
            'id': row.id,
            'date': row.date,
            'content': row.content,
            'embeddings': np.array(json.loads(row.model1)).flatten()
        }
    return None

def extract_tickers(text):
    pattern = r'\{\{TICKER \d+: ([A-Z]{1,5})\}\}'
    return re.findall(pattern, text)

def analyze_ticker(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'symbol': ticker,
        'current_price': info.get('currentPrice', 'No data available')
    }

def parse_predictions(text):
    pattern = r'TICKER: \[(\w+)\]: (\d+\.\d+), (\d+\.\d+), (\d+\.\d+), \{"(.+?)"\}'
    predictions = re.findall(pattern, text)
    print(f"Extracted predictions: {predictions}")  # Debug print
    return predictions

import json
from google.cloud import bigquery
from google.cloud.bigquery import ScalarQueryParameter

def insert_article_predictions(article_id, predictions):
    client = bigquery.Client(project=project_id)

    # Ensure article_id is an integer
    article_id = int(article_id)

    # First, fetch the existing row
    fetch_query = f"""
    SELECT *
    FROM `{project_id}.analysis_dataset.analysis`
    WHERE id = @article_id
    """
    fetch_job_config = bigquery.QueryJobConfig(query_parameters=[
        ScalarQueryParameter("article_id", "INT64", article_id)
    ])
    fetch_job = client.query(fetch_query, job_config=fetch_job_config)
    existing_row = list(fetch_job.result())[0]

    # Prepare the new stock predictions
    new_stock_predictions = [
        {
            "model": "model",
            "ticker": ticker,
            "predicted_price_1hr": float(price_1hr),
            "predicted_price_4hrs": float(price_4hrs),
            "predicted_price_24hrs": float(price_24hrs),
            "stock_price_analysis": reasoning,
            "stock_price_1hr": None,
            "stock_price_2hrs": None,
            "stock_price_3hrs": None,
            "stock_price_5hrs": None,
            "stock_price_10hrs": None,
            "stock_price_24hrs": None
        }
        for ticker, price_1hr, price_4hrs, price_24hrs, reasoning in predictions
    ]

    # Prepare the new row data
    new_row = {
        "id": existing_row.id + 1,
        "date": existing_row.date,
        "content": existing_row.content,
        "sources": existing_row.sources,
        "category": existing_row.category,
        "embeddings": existing_row.embeddings,
        "stock_prediction": new_stock_predictions
    }

    # Insert the new row
    errors = client.insert_rows_json(f"{project_id}.analysis_dataset.analysis", [new_row])
    if errors:
        print(f"Errors occurred while inserting rows: {errors}")
    else:
        print(f"New row inserted successfully with ID: {new_row['id']}")

    # Verify the insert
    verify_query = f"""
    SELECT id, date, content, sources, category, embeddings, stock_prediction
    FROM `{project_id}.analysis_dataset.analysis`
    WHERE id = @new_id
    """
    verify_job_config = bigquery.QueryJobConfig(query_parameters=[
        ScalarQueryParameter("new_id", "INT64", new_row['id'])
    ])
    try:
        verify_job = client.query(verify_query, job_config=verify_job_config)
        results = verify_job.result()
        print("Inserted new row:")
        for row in results:
            print(row)
    except Exception as e:
        print(f"Error verifying insert: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    most_recent_article = fetch_most_recent_article()
    if not most_recent_article:
        return "No articles found in the database."

    query_text = most_recent_article['content']
    query_embedding = most_recent_article['embeddings']
    article_id = most_recent_article['id']
    article_date = most_recent_article['date']

    companies = fetch_company_data()

    company_similarities = [(ticker, summary, name, sector, cosine_similarity(query_embedding, embeddings)) for ticker, summary, embeddings, name, sector in companies]
    top_companies = sorted(company_similarities, key=lambda x: x[4], reverse=True)[:7]

    # First prompt: stockprice.txt
    prompt_path_stockprice = os.path.join('testing/prompts', 'stockprice.txt')
    with open(prompt_path_stockprice, 'r') as file:
        static_prompt_stockprice = file.read()

    ticker_descriptions = [f"{{{{TICKER {i+1}: {ticker}}}}}" for i, (ticker, _, _, _, _) in enumerate(top_companies)]
    full_prompt_stockprice = f"{static_prompt_stockprice} Query: {query_text}. " + ", ".join(ticker_descriptions) + "."

    print("First Prompt: ", full_prompt_stockprice)  # Debug print

    response_stockprice = client_anthropic.messages.create(max_tokens=1024, messages=[{"role": "user", "content": full_prompt_stockprice}], model="claude-3-5-sonnet@20240620")
    response_text_stockprice = response_stockprice.content[0].text

    print("Response from First Prompt: ", response_text_stockprice)  # Debug print

    tickers = extract_tickers(response_text_stockprice)

    if not tickers:
        return "No valid tickers found in the first prompt response."

    ticker_analysis_results = []
    for ticker in tickers:
        ticker_analysis = analyze_ticker(ticker)
        ticker_analysis_results.append(ticker_analysis)

    # Prepare prices for the second prompt
    prices_info = ", ".join([f"{result['symbol']}: ${result['current_price']}" for result in ticker_analysis_results])

    # Second prompt: stock_analysis.txt
    prompt_path_stock_analysis = os.path.join('testing/prompts', 'stock_analysis.txt')
    with open(prompt_path_stock_analysis, 'r') as file:
        static_prompt_stock_analysis = file.read()

    full_prompt_stock_analysis = f"{static_prompt_stock_analysis} Query: {query_text}. Prices: {prices_info}."

    print("Second Prompt: ", full_prompt_stock_analysis)  # Debug print

    response_stock_analysis = client_anthropic.messages.create(max_tokens=1024, messages=[{"role": "user", "content": full_prompt_stock_analysis}], model="claude-3-5-sonnet@20240620")
    response_text_stock_analysis = response_stock_analysis.content[0].text

    print("Response from Second Prompt: ", response_text_stock_analysis)  # Debug print

    predictions = parse_predictions(response_text_stock_analysis)
    print(f"Parsed predictions: {predictions}")  # Debug print

    if predictions:
        insert_article_predictions(article_id, predictions)
    else:
        print("No predictions to insert.")

    analyzed_article = {'date': article_date, 'summary': query_text, 'results': ticker_analysis_results, 'final_analysis': response_text_stock_analysis}
    return render_template('analysis.html', articles=[analyzed_article])

if __name__ == '__main__':
    app.run(debug=True)
