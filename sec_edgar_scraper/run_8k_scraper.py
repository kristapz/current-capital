import subprocess
import sys
import os
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def run_scrapy_spider(spider_name, ticker, max_documents=5):
    """
    Runs a Scrapy spider with the given name and ticker, limiting to max_documents.

    Args:
        spider_name (str): The name of the Scrapy spider to run.
        ticker (str): The stock ticker symbol.
        max_documents (int): Maximum number of filings to process.
    """
    try:
        logging.info(f"Starting Scrapy spider '{spider_name}' for ticker: {ticker} with max_documents={max_documents}")

        # Determine the project directory (assumes this script is in the project root)
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the Scrapy command with max_documents
        command = [
            'scrapy',
            'crawl',
            spider_name,
            '-a',
            f'tickers={ticker}',
            '-a',
            f'max_documents={max_documents}'
        ]

        # Execute the Scrapy spider
        subprocess.check_call(command, cwd=project_dir)

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

def main():
    """
    Main function to parse arguments and orchestrate the scraping and processing workflows.
    """
    parser = argparse.ArgumentParser(description='Run Edgar 8-K Spider and Process 8-K Filings with OpenAI.')
    parser.add_argument('ticker', type=str, help='Ticker symbol to process (e.g., AAPL)')
    args = parser.parse_args()

    ticker = args.ticker.upper()

    # Define spider and processing script configurations
    spider_name = 'edgar8k'
    processing_script = os.path.join('process10k', 'process8k_openai.py')  # Ensure this path is correct

    # Set the maximum number of filings to process
    max_documents = 5

    # Run the Scrapy spider
    run_scrapy_spider(spider_name, ticker, max_documents)

    # Run the corresponding OpenAI processing script
    run_openai_processing(processing_script, ticker)

    logging.info("All processes for 8-K filings have been executed successfully.")

if __name__ == "__main__":
    main()
