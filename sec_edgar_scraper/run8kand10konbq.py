import os
import subprocess
import sys
import logging
from google.cloud import bigquery

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def run_scrapy_spider(spider_name, ticker, max_documents=None, arg_name='tickers'):
    """
    Runs a Scrapy spider with the given name and ticker, optionally limiting to max_documents.

    Args:
        spider_name (str): The name of the Scrapy spider to run.
        ticker (str): The stock ticker symbol.
        max_documents (int, optional): Maximum number of filings to process.
        arg_name (str): The name of the argument to pass to the spider ('tickers' or 'ticker').
    """
    try:
        # Construct the Scrapy command with dynamic argument name
        cmd = [
            'scrapy',
            'crawl',
            spider_name,
            '-a',
            f'{arg_name}={ticker}'
        ]

        if max_documents is not None:
            cmd.extend(['-a', f'max_documents={max_documents}'])

        logging.info(f"Starting Scrapy spider '{spider_name}' for ticker: {ticker} with args: {cmd}")

        # Determine the project directory (assumes this script is in the project root)
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Execute the Scrapy spider
        subprocess.check_call(cmd, cwd=project_dir)

        logging.info(f"Scrapy spider '{spider_name}' completed successfully for ticker: {ticker}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Scrapy spider '{spider_name}' failed for ticker {ticker}. Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Scrapy is not installed or not found in the PATH.")
        sys.exit(1)

def run_openai_processing(script_relative_path, ticker):
    """
    Runs an OpenAI processing script with the given relative path and ticker.

    Args:
        script_relative_path (str): The relative path to the OpenAI processing script.
        ticker (str): The stock ticker symbol.
    """
    try:
        logging.info(f"Starting OpenAI processing script '{script_relative_path}' for ticker: {ticker}")

        # Determine the absolute path to the script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_relative_path)

        if not os.path.isfile(script_path):
            logging.error(f"OpenAI processing script '{script_relative_path}' not found at '{script_path}'.")
            sys.exit(1)

        # Execute the OpenAI processing script with the ticker as an argument
        subprocess.check_call(['python', script_path, '-ticker', ticker])

        logging.info(f"OpenAI processing script '{script_relative_path}' completed successfully for ticker: {ticker}")

    except subprocess.CalledProcessError as e:
        logging.error(f"OpenAI processing script '{script_relative_path}' failed for ticker {ticker}. Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Python is not installed or not found in the PATH.")
        sys.exit(1)

def query_tickers_from_bigquery():
    """
    Queries the 'stocks_filtered23' BigQuery table and retrieves the tickers.

    Returns:
        List of tickers retrieved from the table.
    """
    client = bigquery.Client()
    query = """
        SELECT ticker
        FROM `test1-427219.stock_datasets.stocks_filtered`
    """

    try:
        query_job = client.query(query)
        results = query_job.result()
        tickers = [row['ticker'] for row in results]
        logging.info(f"Retrieved {len(tickers)} tickers from BigQuery.")
        return tickers

    except Exception as e:
        logging.error(f"Error querying BigQuery: {e}")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the scraping and processing workflows for each ticker in BigQuery.
    Runs both the 8-K and additional processing scripts for each ticker.
    """
    # Fetch tickers from BigQuery
    tickers = query_tickers_from_bigquery()

    # Define spider and processing script configurations for 8-K
    spider_8k = 'edgar8k'
    processing_script_8k = os.path.join('process10k', 'process8k_openai.py')  # Ensure this path is correct
    max_documents_8k = 5  # Set the maximum number of filings to process for 8-K

    # Define spider and processing script configurations for additional processing
    spider_cisk = 'edgarcisk'
    processing_script_cisk = os.path.join('process10k', 'process10k_openai.py')  # Ensure this path is correct

    # Process each ticker
    for ticker in tickers:
        ticker = ticker.upper()  # Ensure ticker is uppercase
        logging.info(f"Processing ticker: {ticker}")

        # Run the 8-K Scrapy spider with 'tickers' argument
        run_scrapy_spider(spider_8k, ticker, max_documents=max_documents_8k, arg_name='tickers')

        # Run the corresponding OpenAI processing script for 8-K
        run_openai_processing(processing_script_8k, ticker)

        # Run the additional Scrapy spider with 'ticker' argument
        run_scrapy_spider(spider_cisk, ticker, arg_name='ticker')

        # Run the corresponding OpenAI processing script for additional data
        run_openai_processing(processing_script_cisk, ticker)

        logging.info(f"Completed processing for ticker: {ticker}")

if __name__ == "__main__":
    main()
