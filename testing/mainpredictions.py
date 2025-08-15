import os
import json
import time
import random
import logging
import anthropic
import openai
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
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
from requests.exceptions import SSLError
from urllib3.exceptions import SSLError as URLLib3SSLError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'n'
project_id = ''
dataset_id = 'scraper_articles'
table_id = 'article'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"
client_bq = bigquery.Client(project=project_id)
aiplatform.init(project=project_id, location='us-east1')

# Initialize Anthropic client
client_anthropic = AnthropicVertex(region="europe-west1", project_id=project_id)

# Endpoint for Vertex AI Model for generating embeddings
vertex_endpoint_name = ""
vertex_endpoint = aiplatform.Endpoint(endpoint_name=vertex_endpoint_name)

openai.api_key = ""  # Use a secure method to store this
vertex_large_instruct_endpoint_name = ""
vertex_large_instruct_endpoint = aiplatform.Endpoint(endpoint_name=vertex_large_instruct_endpoint_name)

def fetch_recent_articles(hours=24):
    recent_datetime = datetime.now() - timedelta(hours=hours)
    recent_date_str = recent_datetime.strftime('%m-%d-%Y %I:%M %p')
    logging.info(f"Fetching articles since: {recent_date_str}")

    query = f"""
    SELECT * FROM `{full_table_id}`
    WHERE SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) IS NOT NULL
    AND SAFE.PARSE_DATETIME('%m-%d-%Y %I:%M %p', date) >= PARSE_DATETIME('%m-%d-%Y %I:%M %p', '{recent_date_str}')
    """

    logging.info(f"Query used: {query}")

    query_job = client_bq.query(query)
    articles = [row for row in query_job]

    logging.info(f"Fetched {len(articles)} recent articles")
    return articles


def fetch_existing_titles():
    query = f"""
    SELECT DISTINCT sources.title
    FROM {project_id}.backwards_testing.main,
    UNNEST(sources) AS sources
    """
    query_job = client_bq.query(query)
    existing_titles = set(row.title for row in query_job)
    logging.info(f"Fetched {len(existing_titles)} existing titles")
    return existing_titles

def generate_embeddings(text):
    text = str(text).strip()
    if text == '' or text == 'nan':
        logging.warning("Empty or NaN text encountered, returning empty embedding.")
        return None, None, None

    max_characters = 1450
    cleaned_text = ' '.join(text[:max_characters].split())
    instances = [{"inputs": cleaned_text}]

    try:
        vertex_response = vertex_endpoint.predict(instances=instances)
        vertex_embeddings = json.dumps(vertex_response.predictions[0][0])  # Flatten the nested array
        logging.info(f"Vertex embeddings generated successfully, first 50 chars: {vertex_embeddings[:50]}")
    except Exception as e:
        logging.error(f"Error generating Vertex AI embeddings: {e}")
        vertex_embeddings = ""

    try:
        openai_response = openai.Embedding.create(
            input=cleaned_text,
            model="text-embedding-3-large"
        )
        openai_embeddings = json.dumps(openai_response['data'][0]['embedding'])
        logging.info(f"OpenAI embeddings generated successfully, first 50 chars: {openai_embeddings[:50]}")
    except Exception as e:
        logging.error(f"Error generating OpenAI embeddings: {e}")
        openai_embeddings = ""

    try:
        logging.info("Generating Vertex AI Large Instruct embeddings...")
        vertex_large_instruct_response = vertex_large_instruct_endpoint.predict(instances=instances)
        vertex_large_instruct_embeddings = json.dumps(vertex_large_instruct_response.predictions[0][0])  # Flatten the nested array
        logging.info(f"Vertex AI Large Instruct embeddings generated successfully, first 50 chars: {vertex_large_instruct_embeddings[:50]}")
    except Exception as e:
        logging.error(f"Error generating Vertex AI Large Instruct embeddings: {e}")
        vertex_large_instruct_embeddings = ""

    if not vertex_embeddings or not openai_embeddings or not vertex_large_instruct_embeddings:
        logging.warning("One or more embedding generations failed. Skipping this article.")
        return None, None, None

    return vertex_embeddings, openai_embeddings, vertex_large_instruct_embeddings

def fetch_vertex_embeddings():
    query = f"""
    SELECT ticker, embeddings
    FROM .stock_datasets.stocks
    """

    query_job = client_bq.query(query)
    results = [(row.ticker, np.array(json.loads(row.embeddings))) for row in query_job]
    logging.info(f"Fetched {len(results)} Vertex embeddings")
    return results

def fetch_additional_embeddings(tickers):
    query = f"""
    SELECT ticker, openai_embeddings, embeddings_large_instruct
    FROM .stock_datasets.stocks
    WHERE ticker IN UNNEST(@tickers)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers)
        ]
    )

    query_job = client_bq.query(query, job_config=job_config)
    results = [(row.ticker, np.array(json.loads(row.openai_embeddings)), np.array(json.loads(row.embeddings_large_instruct))) for row in query_job]
    logging.info(f"Fetched embeddings for {len(results)} tickers")
    return results

def cosine_similarity(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return 1 - cosine(v1, v2)

def extract_tickers(text):
    pattern = r'\{\{TICKER \d+: ([A-Z]{1,5})\}\}'
    tickers = re.findall(pattern, text)
    logging.info(f"Extracted tickers: {tickers}")
    return tickers

def analyze_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get current price or calculate the average of the day's high and low
        current_price = info.get('currentPrice')
        if current_price is None:
            high = info.get('dayHigh', None)
            low = info.get('dayLow', None)
            if high is not None and low is not None:
                current_price = (high + low) / 2
            else:
                current_price = 'No data available'

        logging.info(f"Analyzed ticker {ticker}: current price - {current_price}")
    except Exception as e:
        logging.error(f"Error analyzing ticker {ticker}: {e}")
        current_price = 'Error fetching data'

    analysis = {
        'symbol': ticker,
        'current_price': current_price
    }
    return analysis


def parse_predictions(text):
    pattern = r'\{\{TICKER: \[(\w+)\]\}\}: \{\{([\d\.]+)\}\}, \{\{([\d\.]+)\}\}, \{\{([\d\.]+)\}\}, \{\{"([^"]+)"\}\}, \{\{"([^"]+)"\}\}'
    predictions = re.findall(pattern, text)
    logging.info(f"Extracted predictions: {predictions}")
    return predictions

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((
        google_exceptions.ServerError,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        requests.exceptions.RequestException,
        SSLError,
        URLLib3SSLError
    ))
)
def insert_article_predictions(article_id, predictions, article_data, effect):
    client = bigquery.Client(project=project_id)

    # Get current time in PST
    pst_timezone = pytz.timezone('US/Pacific')
    current_time_pst = datetime.now(pst_timezone)
    formatted_date = current_time_pst.strftime('%m-%d-%Y %I:%M %p')

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
            "stock_price_24hrs": None,
            "trend": trend,
            "%change": ((float(price_24hrs) - float(price_1hr)) / float(price_1hr)) * 100
        }
        for ticker, price_1hr, price_4hrs, price_24hrs, reasoning, trend in predictions
    ]

    article_data = {
        "content": article_data.get("content", ""),
        "category": article_data.get("category", ""),
        "embeddings": {
            "model1": json.dumps(article_data['embeddings'].get('model1', "")),
            "model2": json.dumps(article_data['embeddings'].get('model2', "")),
            "model3": json.dumps(article_data['embeddings'].get('model3', "")),
            "model4": json.dumps(article_data['embeddings'].get('model4', ""))  # If you have a fourth model, add its embeddings here
        },
        "link": article_data.get("link", ""),
        "publication": article_data.get("publication", ""),
        "title": article_data.get("title", "")
    }

    new_row = {
        "id": article_id,
        "date": formatted_date,
        "content": [article_data['content']],
        "updated": ["true"],
        "effect": effect,
        "sources": [{
            "id": article_id,
            "link": article_data['link'],
            "publication": article_data['publication'],
            "title": article_data['title']
        }],
        "category": article_data['category'],
        "embeddings": article_data['embeddings'],
        "stock_prediction": new_stock_predictions
    }

    errors = client.insert_rows_json(f"{project_id}.backwards_testing.main", [new_row])
    if errors:
        logging.error(f"Errors occurred while inserting rows: {errors}")
    else:
        logging.info(f"New row inserted successfully with ID: {new_row['id']} at {formatted_date}")

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
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((anthropic.RateLimitError, requests.exceptions.RequestException, AnthropicTimeoutError, SSLError, URLLib3SSLError))
)
def retry_anthropic_call(func, *args, **kwargs):
    try:
        start_time = time.time()
        response = anthropic_call_with_timeout(func, timeout=60, *args, **kwargs)
        end_time = time.time()
        logging.info(f"API call completed in {end_time - start_time:.2f} seconds")
        logging.info(f"API Response: {response}")
        return response
    except (AnthropicTimeoutError, SSLError, URLLib3SSLError) as e:
        logging.warning(f"API call failed due to timeout or SSL error. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 5))  # Add a random delay before retry
        raise
    except (anthropic.RateLimitError, requests.exceptions.RequestException) as e:
        logging.warning(f"API call failed. Retrying... Error: {e}")
        time.sleep(random.uniform(1, 5))  # Add a random delay before retry
        raise
    except Exception as e:
        logging.error(f"Unexpected error in API call: {e}")
        raise

def main():
    companies = fetch_vertex_embeddings()
    backoff_time = 5  # Start with a 5-second backoff

    while True:
        try:
            # Fetch articles from the last 24 hours
            articles = fetch_recent_articles(hours=24)
            existing_titles = fetch_existing_titles()

            if not articles:
                logging.info("No recent articles found. Sleeping before next iteration...")
                time.sleep(backoff_time)
                continue

            for article in articles:
                logging.info(f"Article fetched: {article['title']} - {article['date']}")

                article_title = article['title']

                if article_title in existing_titles:
                    logging.info(f"Article with title '{article_title}' already processed. Skipping...")
                    continue

                article_id = random.randint(1, 10000)
                logging.info(f"Processing article ID: {article_id}")
                try:
                    if not article['content'] or pd.isna(article['content']):
                        logging.warning(f"Empty or NaN content for article ID {article_id}. Skipping...")
                        continue

                    article_content = article['content']
                    vertex_embeddings, openai_embeddings, vertex_large_instruct_embeddings = generate_embeddings(article_content)

                    if vertex_embeddings is None or openai_embeddings is None or vertex_large_instruct_embeddings is None:
                        logging.warning(f"Invalid embeddings for article ID {article_id}. Skipping...")
                        continue

                    try:
                        vertex_embeddings = np.array(json.loads(vertex_embeddings))
                        openai_embeddings = np.array(json.loads(openai_embeddings))
                        vertex_large_instruct_embeddings = np.array(json.loads(vertex_large_instruct_embeddings))
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON for embeddings: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error processing embeddings: {e}")
                        continue

                    logging.info(f"Generated embeddings for article ID {article_id}")
                    logging.debug(f"Vertex embeddings shape: {vertex_embeddings.shape}")
                    logging.debug(f"OpenAI embeddings shape: {openai_embeddings.shape}")
                    logging.debug(f"Vertex large instruct embeddings shape: {vertex_large_instruct_embeddings.shape}")

                    # Vertex AI Embeddings Comparison
                    company_similarities_vertex = [
                        (ticker, cosine_similarity(vertex_embeddings, embeddings))
                        for ticker, embeddings in companies
                    ]

                    # After calculating similarities for Vertex AI
                    top_100_vertex = sorted(company_similarities_vertex, key=lambda x: x[1], reverse=True)[:100]
                    top_100_tickers = [ticker for ticker, _ in top_100_vertex]

                    logging.info("Top 100 tickers used for additional embeddings:")
                    logging.info(", ".join(top_100_tickers))

                    # Fetch additional embeddings
                    additional_embeddings = fetch_additional_embeddings(top_100_tickers)

                    # Calculate similarities for OpenAI and Vertex AI Large Instruct
                    company_similarities_openai = [
                        (ticker, cosine_similarity(openai_embeddings, openai_emb))
                        for ticker, openai_emb, _ in additional_embeddings
                    ]

                    company_similarities_vertex_large_instruct = [
                        (ticker, cosine_similarity(vertex_large_instruct_embeddings, vertex_large_instruct_emb))
                        for ticker, _, vertex_large_instruct_emb in additional_embeddings
                    ]

                    # Sort and get top 3 for each model
                    top_companies_vertex = sorted(top_100_vertex, key=lambda x: x[1], reverse=True)[:3]
                    top_companies_openai = sorted(company_similarities_openai, key=lambda x: x[1], reverse=True)[:3]
                    top_companies_vertex_large_instruct = sorted(company_similarities_vertex_large_instruct, key=lambda x: x[1], reverse=True)[:3]

                    # Print out the similar stocks
                    logging.info("\nTop companies for Vertex AI Embeddings:")
                    for ticker, similarity in top_companies_vertex:
                        logging.info(f"{ticker}: {similarity}")

                    logging.info("\nTop companies for OpenAI Embeddings:")
                    for ticker, similarity in top_companies_openai:
                        logging.info(f"{ticker}: {similarity}")

                    logging.info("\nTop companies for Vertex AI Large Instruct Embeddings:")
                    for ticker, similarity in top_companies_vertex_large_instruct:
                        logging.info(f"{ticker}: {similarity}")

                    prompt_path_stockprice = 'prompts/stockprice.txt'
                    with open(prompt_path_stockprice, 'r') as file:
                        static_prompt_stockprice = file.read()

                    ticker_descriptions_vertex = [f"{{{{TICKER {i+1}: {ticker}}}}}" for i, (ticker, _) in enumerate(top_companies_vertex)]
                    ticker_descriptions_openai = [f"{{{{TICKER {i+1}: {ticker}}}}}" for i, (ticker, _) in enumerate(top_companies_openai)]
                    ticker_descriptions_vertex_large_instruct = [f"{{{{TICKER {i+1}: {ticker}}}}}" for i, (ticker, _) in enumerate(top_companies_vertex_large_instruct)]

                    full_prompt_stockprice = f"{static_prompt_stockprice} Query: {article_content}. Vertex AI: " + ", ".join(ticker_descriptions_vertex) + ". OpenAI: " + ", ".join(ticker_descriptions_openai) + ". Vertex AI Large Instruct: " + ", ".join(ticker_descriptions_vertex_large_instruct) + "."

                    logging.info(f"Constructed full prompt: {full_prompt_stockprice}")

                    response_stockprice = retry_anthropic_call(
                        client_anthropic.messages.create,
                        max_tokens=3500,
                        messages=[{"role": "user", "content": full_prompt_stockprice}],
                        model="claude-3-5-sonnet@20240620"
                    )
                    response_text_stockprice = response_stockprice.content[0].text

                    logging.info(f"API Response: {response_text_stockprice}")

                    effect_pattern = r'\{\{effect: "(\w+)"\}\}'
                    effect_match = re.search(effect_pattern, response_text_stockprice)
                    effect = effect_match.group(1) if effect_match else "none"

                    tickers = extract_tickers(response_text_stockprice)

                    if not tickers:
                        logging.warning(f"No valid tickers found for article ID: {article_id}")
                        continue

                    ticker_analysis_results = []
                    for ticker in tickers:
                        ticker_analysis = analyze_ticker(ticker)
                        ticker_analysis_results.append(ticker_analysis)
                        time.sleep(1)  # Add a delay between requests to avoid rate limits

                    prices_info = ", ".join([f"{result['symbol']}: ${result['current_price']}" for result in ticker_analysis_results])

                    prompt_path_stock_analysis = 'prompts/stock_analysis.txt'
                    with open(prompt_path_stock_analysis, 'r') as file:
                        static_prompt_stock_analysis = file.read()

                    full_prompt_stock_analysis = f"{static_prompt_stock_analysis} Query: {article_content}. Prices: {prices_info}."
                    print()

                    response_stock_analysis = retry_anthropic_call(
                        client_anthropic.messages.create,
                        max_tokens=3500,
                        messages=[{"role": "user", "content": full_prompt_stock_analysis}],
                        model="claude-3-5-sonnet@20240620"
                    )
                    response_text_stock_analysis = response_stock_analysis.content[0].text

                    logging.info(f"API Response (Stock Analysis): {response_text_stock_analysis}")

                    predictions = parse_predictions(response_text_stock_analysis)

                    if predictions:
                        article_data = {
                            "title": article['title'],
                            "date": article['date'],
                            "author": article['author'],
                            "content": article_content,
                            "link": article['link'],
                            "publication": article['publication'],
                            "embeddings": {
                                "model1": vertex_embeddings.tolist(),
                                "model2": openai_embeddings.tolist(),
                                "model3": vertex_large_instruct_embeddings.tolist(),
                                "model4": ""  # If you have a fourth model, add its embeddings here
                            }
                        }
                        insert_article_predictions(article_id, predictions, article_data, effect)
                    else:
                        logging.warning(f"No predictions to insert for article ID: {article_id}")

                except Exception as e:
                    logging.error(f"Error processing article ID {article_id}: {e}")
                    logging.debug(f"Vertex embeddings shape: {vertex_embeddings.shape}")
                    logging.debug(f"OpenAI embeddings shape: {openai_embeddings.shape}")
                    logging.debug(f"Vertex large instruct embeddings shape: {vertex_large_instruct_embeddings.shape}")

            logging.info("Sleeping for 30 minutes before next iteration...")
            time.sleep(300)  # Sleep for 5 minutes

            backoff_time = 3  # Reset backoff time on successful iteration

        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            logging.info(f"Backing off for {backoff_time} seconds before retrying...")
            time.sleep(backoff_time)
            backoff_time = min(backoff_time * 2, 300)  # Double the backoff time, max 30 minutes

if __name__ == "__main__":
    main()
