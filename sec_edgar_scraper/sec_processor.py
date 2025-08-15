import os
import sys
import logging
import subprocess
import argparse
from datetime import datetime
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from sec_cik_mapper import StockMapper

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''


class SECFilingProcessor:
    def __init__(self, max_filings_per_type=None, max_tickers=None):
        self.client = bigquery.Client()
        self.dataset_id = 'backwards_testing'
        self.combined_table_id = "combined_filings_master"
        self.max_filings_per_type = max_filings_per_type
        self.max_tickers = max_tickers

    def query_tickers_from_bigquery(self):
        """Queries tickers from filtered stocks table with optional limit"""
        limit_clause = f"LIMIT {self.max_tickers}" if self.max_tickers else ""
        query = f"""
            SELECT ticker
            FROM `.stock_datasets.filtered_stocks`
            {limit_clause}
        """
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            tickers = [row['ticker'] for row in results]
            logging.info(f"Retrieved {len(tickers)} tickers from BigQuery.")
            return tickers
        except Exception as e:
            logging.error(f"Error querying BigQuery: {e}")
            sys.exit(1)

    def run_scrapy_spider(self, spider_name, ticker, arg_name='tickers'):
        """Runs Scrapy spider for SEC filing collection"""
        try:
            cmd = ['scrapy', 'crawl', spider_name, '-a', f'{arg_name}={ticker}']
            if self.max_filings_per_type:
                cmd.extend(['-a', f'max_documents={self.max_filings_per_type}'])

            logging.debug(f"Starting spider '{spider_name}' for {ticker}")
            subprocess.check_call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            logging.debug(f"Completed spider '{spider_name}' for {ticker}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Spider failed for {ticker}: {e}")
            return False

    def run_openai_processing(self, script_path, ticker):
        """Runs OpenAI processing on collected filings"""
        try:
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_path)
            logging.debug(f"Processing {ticker} with OpenAI")
            subprocess.check_call(['python', full_path, '-ticker', ticker])
            logging.debug(f"Completed OpenAI processing for {ticker}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Processing failed for {ticker}: {e}")
            return False

    def create_combined_table(self):
        """Creates or verifies the combined filings table with metadata"""
        table_ref = self.client.dataset(self.dataset_id).table(self.combined_table_id)
        schema = [
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("ticker", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("cik", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("filing_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("processed_items", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("item_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("processed_text", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("effect", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("date", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("prediction_text", "STRING", mode="NULLABLE"),
            ]),
            bigquery.SchemaField("first_added", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("last_updated", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("update_count", "INTEGER", mode="REQUIRED"),
        ]

        try:
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            logging.info(f"Created table {self.combined_table_id}")
        except Exception as e:
            if "Already Exists" not in str(e):
                logging.error(f"Failed to create table: {e}")
                sys.exit(1)

    def check_existing_ticker(self, ticker):
        """
        Check if ticker exists and get its metadata
        Returns (exists, latest_8k_date, update_count) tuple
        """
        query = f"""
            SELECT 
                COUNT(*) as count,
                MAX(CASE WHEN source = '8k' THEN filing_date END) as latest_8k_date,
                MAX(update_count) as update_count,
                MAX(first_added) as first_added,
                MAX(last_updated) as last_updated
            FROM `{self.client.project}.{self.dataset_id}.{self.combined_table_id}`
            WHERE ticker = @ticker
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker)
            ]
        )

        try:
            results = self.client.query(query, job_config=job_config).result()
            row = next(results)
            exists = row.count > 0

            if exists:
                logging.info(f"""
                    Ticker: {ticker}
                    First added: {row.first_added}
                    Last updated: {row.last_updated}
                    Update count: {row.update_count}
                    Latest 8K: {row.latest_8k_date}
                """)
            else:
                logging.info(f"Ticker {ticker} not found in database")

            return exists, row.latest_8k_date, row.update_count if exists else 0

        except NotFound:
            logging.info(f"Table {self.combined_table_id} does not exist yet")
            return False, None, 0
        except Exception as e:
            logging.error(f"Error checking ticker existence: {e}")
            return False, None, 0

    def _insert_processed_filings(self, results, source_type, ticker, update_count):
        """Helper method to insert processed filings into combined table"""
        rows = []
        current_time = datetime.utcnow()

        for row in results:
            processed_items = row.get('processed_items', [])
            if processed_items:
                combined_row = {
                    "source": source_type,
                    "ticker": row.get('ticker'),
                    "cik": row.get('cik'),
                    "filing_date": row['filing_date'].isoformat(),
                    "processed_items": processed_items,
                    "first_added": current_time.isoformat() if update_count == 0 else None,
                    "last_updated": current_time.isoformat(),
                    "update_count": update_count + 1
                }
                rows.append(combined_row)

        if rows:
            table_ref = self.client.dataset(self.dataset_id).table(self.combined_table_id)
            errors = self.client.insert_rows_json(table_ref, rows)
            if errors:
                logging.error(f"Errors inserting {source_type} rows: {errors}")
            else:
                logging.info(f"Inserted {len(rows)} {source_type} rows for {ticker} (Update #{update_count + 1})")

    def _process_8k_filings(self, tickers, update_count):
        """Process and aggregate 8-K filings"""
        query = """
            SELECT ticker, filing_date, processed_items
            FROM `test1-427219.backwards_testing.processed_eight_k_filings`
            WHERE ticker IN UNNEST(@tickers)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", tickers)]
        )

        try:
            results = self.client.query(query, job_config=job_config).result()
            self._insert_processed_filings(results, "8k", tickers[0], update_count)
        except Exception as e:
            logging.error(f"Error processing 8-K filings: {e}")

    def _process_10k_filings(self, ciks, ticker, update_count):
        """Process and aggregate 10-K filings"""
        if not ciks:
            logging.warning("No CIKs provided for 10-K processing")
            return

        query = """
            SELECT cik, filing_date, processed_items
            FROM `test1-427219.backwards_testing.processed_ten_k_filings`
            WHERE cik IN UNNEST(@ciks)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ciks", "STRING", ciks)]
        )

        try:
            results = self.client.query(query, job_config=job_config).result()
            self._insert_processed_filings(results, "10k", ticker, update_count)
        except Exception as e:
            logging.error(f"Error processing 10-K filings: {e}")

    def aggregate_processed_data(self, tickers, update_count):
        """Aggregates processed filing data"""
        mapper = StockMapper()
        ticker = tickers[0]  # Since we're processing one at a time
        cik = mapper.ticker_to_cik.get(ticker.upper())

        if cik:
            self._process_8k_filings(tickers, update_count)
            self._process_10k_filings([cik], ticker, update_count)
        else:
            logging.warning(f"No CIK found for ticker {ticker}")
            self._process_8k_filings(tickers, update_count)

    def process_all_filings(self):
        """Main processing function"""
        # Create combined table first if it doesn't exist
        self.create_combined_table()

        # Get tickers
        tickers = self.query_tickers_from_bigquery()
        logging.info(f"Processing {len(tickers)} tickers")

        # Process each ticker
        for ticker in tickers:
            exists, latest_8k_date, update_count = self.check_existing_ticker(ticker)

            if not exists:
                logging.info(f"Processing new ticker: {ticker}")
                if self.run_scrapy_spider('edgar8k', ticker):
                    self.run_openai_processing('process10k/process8k_openai.py', ticker)

                if self.run_scrapy_spider('edgarcisk', ticker, arg_name='ticker'):
                    self.run_openai_processing('process10k/process10k_openai.py', ticker)

                self.aggregate_processed_data([ticker], update_count)
            else:
                # Check for new 8K filings since latest_8k_date
                if latest_8k_date:
                    logging.info(f"Checking for new 8K filings for {ticker} after {latest_8k_date}")
                    # Here you would implement the logic to only get new 8Ks
                    # For now, we'll just log it
                    logging.info(f"Skipping {ticker} - already processed (implement incremental 8K updates)")


def main():
    parser = argparse.ArgumentParser(description='Process SEC filings with optional limits')
    parser.add_argument('--max-filings', type=int, help='Maximum number of filings per type (8K/10K)')
    parser.add_argument('--max-tickers', type=int, help='Maximum number of tickers to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    processor = SECFilingProcessor(
        max_filings_per_type=args.max_filings,
        max_tickers=args.max_tickers
    )
    processor.process_all_filings()
    logging.info("Processing and aggregation completed successfully")


if __name__ == "__main__":
    main()