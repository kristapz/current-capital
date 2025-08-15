import subprocess
import sys
import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_scrapy_spider(ticker):
    try:
        logging.info(f"Starting Scrapy spider for ticker: {ticker}")
        # Navigate to the project directory if necessary
        project_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call(['scrapy', 'crawl', 'edgarcisk', '-a', f'ticker={ticker}'], cwd=project_dir)
        logging.info(f"Scrapy spider completed for ticker: {ticker}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Scrapy spider failed for ticker {ticker}. Error: {e}")
        sys.exit(1)

def run_process10k_openai(ticker):
    try:
        logging.info(f"Starting OpenAI processing for ticker: {ticker}")
        script_path = os.path.join('process10k', 'process10k_openai.py')
        subprocess.check_call(['python', script_path, '-ticker', ticker], cwd=os.path.dirname(os.path.abspath(__file__)))
        logging.info(f"OpenAI processing completed for ticker: {ticker}")
    except subprocess.CalledProcessError as e:
        logging.error(f"OpenAI processing failed for ticker {ticker}. Error: {e}")
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser(description='Run Edgar Spider and Process 10-K Filings with OpenAI.')
    parser.add_argument('ticker', type=str, help='Ticker symbol to process')
    args = parser.parse_args()

    ticker = args.ticker.upper()

    # Run the Scrapy spider
    run_scrapy_spider(ticker)

    # Run the OpenAI processing script
    run_process10k_openai(ticker)

if __name__ == "__main__":
    main()
