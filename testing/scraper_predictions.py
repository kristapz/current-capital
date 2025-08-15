import os
import json
import time
import random
import logging
import anthropic
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery, aiplatform
from scipy.spatial.distance import cosine
from anthropic import AnthropicVertex
from google.api_core import retry
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import yfinance as yf
import re
import concurrent.futures
import pytz


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'scraper_articles'
table_id = 'article'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"
client_bq = bigquery.Client(project=project_id)
aiplatform.init(project=project_id, location='us-east1')

# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id=project_id)

# Endpoint for Vertex AI Model for generating embeddings
endpoint_name = ""
endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

def fetch_recent_articles(days=2):
    recent_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = f"""
    SELECT * FROM `{full_table_id}`
    WHERE DATE(date) >= '{recent_date}'
    """
    query_job = client_bq.query(query)
    articles = [row for row in query_job]
    return articles

def fetch_existing_titles():
    query = f"""
    SELECT source.title
    FROM `{project_id}.backwards_testing.calls_together`, UNNEST(sources) as source
    """
    try:
        query_job = client_bq.query(query)
        existing_titles = {row["title"] for row in query_job}
        return existing_titles
    except Exception as e:
        logging.error(f"Error fetching existing titles: {e}")
        return set()

def generate_embeddings(text):
    text = str(text).strip()
    if text == '' or text == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding.")
        return ""

    max_characters = 1450
    cleaned_text = ' '.join(text[:max_characters].split())
    instances = [{"inputs": cleaned_text}]
    try:
        response = endpoint.predict(instances=instances)
        return json.dumps(response.predictions[0])
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return ""

def cosine_similarity(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return 1 - cosine(v1, v2)

def fetch_company_data():
    query = """
    SELECT ticker, long_business_summary, embeddings, name, sector
    FROM `.stock_datasets.stocksv2`
    """
    query_job = client_bq.query(query)
    data = [(row.ticker, row.long_business_summary, np.array(json.loads(row.embeddings)).flatten(),
             row.name, row.sector) for row in query_job]
    return data

def extract_tickers(text):
    pattern = r'\{\{TICKER \d+: ([A-Z]{1,5})\}\}'
    tickers = re.findall(pattern, text)
    return tickers

def analyze_ticker(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    analysis = {
        'symbol': ticker,
        'current_price': info.get('currentPrice', 'No data available')
    }
    return analysis

def parse_predictions(text):
    pattern = r'TICKER: \[(\w+)\]: (\d+\.\d+), (\d+\.\d+), (\d+\.\d+), \{"(.+?)"\}'
    predictions = re.findall(pattern, text)
    logging.info(f"Extracted predictions: {predictions}")
    return predictions

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((google_exceptions.ServerError, google_exceptions.TooManyRequests, google_exceptions.ServiceUnavailable, requests.exceptions.RequestException))
)
def insert_article_predictions(article_id, predictions, article_data):
    client = bigquery.Client(project=project_id)
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

    # Handle missing fields by setting them to empty values
    article_data = {
        "date": article_data.get("date", ""),
        "content": article_data.get("content", ""),
        "category": article_data.get("category", ""),
        "embeddings": article_data.get("embeddings", ""),
        "link": article_data.get("link", ""),
        "publication": article_data.get("publication", ""),
        "title": article_data.get("title", "")
    }

    new_row = {
        "id": article_id,
        "date": article_data['date'],
        "content": [article_data['content']],
        "updated": ["true"],  # Set updated to "true" as a string
        "sources": [{
            "id": article_id,
            "link": article_data['link'],
            "publication": article_data['publication'],
            "title": article_data['title']
        }],
        "category": article_data['category'],
        "embeddings": {
            "model1": article_data['embeddings'],
            "model2": "",
            "model3": "",
            "model4": ""
        },
        "stock_prediction": new_stock_predictions
    }

    errors = client.insert_rows_json(f"{project_id}.backwards_testing.calls_together", [new_row])
    if errors:
        logging.error(f"Errors occurred while inserting rows: {errors}")
    else:
        logging.info(f"New row inserted successfully with ID: {new_row['id']}")

class AnthropicTimeoutError(Exception):
    pass

def anthropic_call_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise AnthropicTimeoutError(f"Anthropic API call timed out after {timeout} seconds")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((anthropic.RateLimitError, requests.exceptions.RequestException, AnthropicTimeoutError))
)
def retry_anthropic_call(func, *args, **kwargs):
    try:
        start_time = time.time()
        response = anthropic_call_with_timeout(func, timeout=60, *args, **kwargs)  # 60 seconds timeout
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds")
        logging.info(f"API Response: {response}")
        return response
    except AnthropicTimeoutError as e:
        logging.warning(f"API call timed out. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 3))  # Add a small random delay
        raise  # Re-raise the exception to be caught by the retry decorator
    except (anthropic.RateLimitError, requests.exceptions.RequestException) as e:
        logging.warning(f"API call failed. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 3))  # Add a small random delay
        raise  # Re-raise the exception to be caught by the retry decorator
    except Exception as e:
        logging.error(f"Unexpected error in API call: {e}")
        raise

def main():
    companies = fetch_company_data()

    while True:
        try:
            articles = fetch_recent_articles()
            existing_titles = fetch_existing_titles()

            for article in articles:
                if article['title'] in existing_titles:
                    logging.info(f"Article with title '{article['title']}' already exists. Skipping.")
                    continue

                article_id = random.randint(1, 10000)
                logging.info(f"Processing article ID: {article_id}")
                try:
                    article_content = article['content']
                    query_embedding = generate_embeddings(article_content)
                    query_embedding = json.dumps(np.array(json.loads(query_embedding)).tolist())

                    company_similarities = [(ticker, summary, name, sector, cosine_similarity(np.array(json.loads(query_embedding)), embeddings))
                                            for ticker, summary, embeddings, name, sector in companies]
                    top_companies = sorted(company_similarities, key=lambda x: x[4], reverse=True)[:7]

                    prompt_path_stockprice = 'prompts/stockprice.txt'
                    with open(prompt_path_stockprice, 'r') as file:
                        static_prompt_stockprice = file.read()

                    ticker_descriptions = [f"{{{{TICKER {i+1}: {ticker}}}}}" for i, (ticker, _, _, _, _) in enumerate(top_companies)]
                    full_prompt_stockprice = f"{static_prompt_stockprice} Query: {article_content}. " + ", ".join(ticker_descriptions) + "."

                    response_stockprice = retry_anthropic_call(
                        client_anthropic.messages.create,
                        max_tokens=3500,
                        messages=[{"role": "user", "content": full_prompt_stockprice}],
                        model="claude-3-5-sonnet@20240620"
                    )
                    response_text_stockprice = response_stockprice.content[0].text

                    tickers = extract_tickers(response_text_stockprice)

                    if not tickers:
                        logging.warning(f"No valid tickers found for article ID: {article_id}")
                        continue

                    ticker_analysis_results = []
                    for ticker in tickers:
                        ticker_analysis = analyze_ticker(ticker)
                        ticker_analysis_results.append(ticker_analysis)
                        time.sleep(12)  # Add a delay between requests to avoid rate limits

                    prices_info = ", ".join([f"{result['symbol']}: ${result['current_price']}" for result in ticker_analysis_results])

                    prompt_path_stock_analysis = 'prompts/stock_analysis.txt'
                    with open(prompt_path_stock_analysis, 'r') as file:
                        static_prompt_stock_analysis = file.read()

                    full_prompt_stock_analysis = f"{static_prompt_stock_analysis} Query: {article_content}. Prices: {prices_info}."

                    response_stock_analysis = retry_anthropic_call(
                        client_anthropic.messages.create,
                        max_tokens=3500,
                        messages=[{"role": "user", "content": full_prompt_stock_analysis}],
                        model="claude-3-5-sonnet@20240620"
                    )
                    response_text_stock_analysis = response_stock_analysis.content[0].text

                    predictions = parse_predictions(response_text_stock_analysis)

                    if predictions:
                        article_data = {
                            "title": article['title'],
                            "date": article['date'],
                            "author": article['author'],
                            "content": article_content,
                            "link": article['link'],
                            "publication": article['publication'],
                            "embeddings": query_embedding
                        }
                        insert_article_predictions(article_id, predictions, article_data)
                    else:
                        logging.warning(f"No predictions to insert for article ID: {article_id}")

                except Exception as e:
                    logging.error(f"Error processing article ID {article_id}: {e}")

                # Add a delay between processing articles to avoid hitting the rate limit
                time.sleep(12)

            logging.info("Sleeping for 30 minutes before next iteration...")
            time.sleep(1800)  # Sleep for 30 minutes

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")

if __name__ == "__main__":
    main()
