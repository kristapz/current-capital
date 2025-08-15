import scrapy
import json
import re
from datetime import datetime
import pytz
import warnings
from bs4 import BeautifulSoup
import html2text
import logging
import os
from google.cloud import bigquery
from sec_cik_mapper import StockMapper

# Suppress ScrapyDeprecationWarning
from scrapy.exceptions import ScrapyDeprecationWarning

warnings.filterwarnings("ignore", category=ScrapyDeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment and BigQuery configuration
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Update the path accordingly
project_id = ''
dataset_id = 'backwards_testing'
table_id = 'eight_k_filings'  # Renamed for clarity
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=project_id)

# Initialize Stock CIK Mapper
stock_mapper = StockMapper()

class Edgar8KSpider(scrapy.Spider):
    name = 'edgar8k'
    allowed_domains = ['sec.gov']

    custom_settings = {
        'USER_AGENT': 'Your Name yourname@example.com',  # Replace with your details
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': False,
        'LOG_LEVEL': 'INFO',
        'EXTENSIONS': {  # Disable unnecessary extensions
            'scrapy.extensions.telnet.TelnetConsole': None,
            'scrapy.extensions.logstats.LogStats': None,
        }
    }

    # Define the target 8-K items
    target_items = [
        "item1.01",
        "item1.02",
        "item2.01",
        "item2.02",
        "item2.03",
        "item2.04",
        "item5.02",
    ]

    # Mapping of 8-K items to descriptive keys for BigQuery
    known_sections = {
        "item1.01": "entry_into_a_material_definitive_agreement",
        "item1.02": "termination_of_a_material_definitive_agreement",
        "item2.01": "completion_of_acquisition_or_disposition_of_assets",
        "item2.02": "results_of_operations_and_financial_condition",
        "item2.03": "creation_of_a_direct_financial_obligation",
        "item2.04": "triggering_events_that_increase_financial_obligation",
        "item5.02": "departure_of_directors_or_officers",
    }

    def __init__(self, tickers=None, days=10000, max_documents=100, *args, **kwargs):
        """
        Initialize the spider with provided tickers.
        Usage: scrapy crawl edgar8k -a tickers=XYZ,ABC
        """
        super(Edgar8KSpider, self).__init__(*args, **kwargs)
        if not tickers:
            self.logger.error("Tickers argument is required. Usage: scrapy crawl edgar8k -a tickers=XYZ,ABC")
            raise ValueError("Tickers argument is required. Usage: scrapy crawl edgar8k -a tickers=XYZ,ABC")
        self.tickers = [ticker.strip().upper() for ticker in tickers.split(',')]
        self.days = int(days)
        self.max_documents = int(max_documents)
        self.processed_documents = {ticker: 0 for ticker in self.tickers}

    def start_requests(self):
        """
        Initiate requests for each ticker to fetch their submission data.
        """
        for ticker in self.tickers:
            specific_cik = stock_mapper.ticker_to_cik.get(ticker)
            if not specific_cik:
                self.logger.error(f"CIK not found for ticker: {ticker}")
                continue

            url = f"https://data.sec.gov/submissions/CIK{specific_cik.zfill(10)}.json"

            yield scrapy.Request(
                url=url,
                callback=self.parse_company_filings,
                meta={'cik': specific_cik, 'company': ticker}
            )

    def parse_company_filings(self, response):
        """
        Parse the submissions JSON to find relevant 8-K filings.
        """
        data = json.loads(response.text)
        cik = response.meta['cik']
        company = response.meta['company']

        recent_filings = data.get('filings', {}).get('recent', {})
        if not recent_filings:
            self.logger.warning(f"No recent filings found for CIK: {cik}")
            return

        forms = recent_filings.get('form', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        filing_dates = recent_filings.get('filingDate', [])
        primary_documents = recent_filings.get('primaryDocument', [])

        for i, form in enumerate(forms):
            if form.upper() == '8-K':
                accession_number = accession_numbers[i].replace('-', '')
                filing_date = filing_dates[i]
                primary_document = primary_documents[i]
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{primary_document}"

                # Check if the filing is within the specified days
                filing_datetime = datetime.strptime(filing_date, '%Y-%m-%d')
                current_datetime = datetime.utcnow()
                delta = current_datetime - filing_datetime
                if delta.days > self.days:
                    continue  # Skip filings older than specified days

                # Yield a request to fetch the 8-K document
                yield scrapy.Request(
                    url=doc_url,
                    callback=self.process_and_save_text,
                    meta={
                        'cik': cik,
                        'company': company,
                        'filing_date': filing_date,
                        'accession_number': accession_number,
                        'doc_url': doc_url
                    }
                )

                # Update the processed documents count
                self.processed_documents[company] += 1
                if self.processed_documents[company] >= self.max_documents:
                    break  # Move to the next ticker

    def process_and_save_text(self, response):
        """
        Process the 8-K document, extract relevant sections, and prepare data for BigQuery.
        """
        cik = response.meta['cik']
        company = response.meta['company']
        filing_date = response.meta['filing_date']
        accession_number = response.meta['accession_number']
        doc_url = response.meta['doc_url']

        # Step 1: Detect if content is HTML or XML and parse accordingly
        content_type = 'xml' if '<?xml' in response.text[:100].lower() else 'html'
        parser = 'xml' if content_type == 'xml' else 'lxml'

        cleaned_html = self.preprocess_html(response.text, parser)

        # Step 2: Convert to Markdown using html2text
        markdown_text = self.convert_html_to_markdown(cleaned_html)

        # Step 3: Extract sections and create JSON output
        sections_json = self.extract_sections(markdown_text)

        # Check if any target sections are present
        if not any(key in sections_json for key in self.target_items):
            self.logger.info(f"No target sections found in filing {accession_number}. Skipping insertion.")
            return

        # Combine metadata and sections into a single dictionary
        combined_output = {
            'cik': cik,
            'company': company,
            'filing_date': filing_date,
            'accession_number': accession_number,
            'url': doc_url,
            'sections': sections_json  # Use sections for BigQuery insertion
        }

        # Insert the data into BigQuery
        self.insert_8k_data_to_bigquery(combined_output)

    def insert_8k_data_to_bigquery(self, data):
        """
        Insert the extracted 8-K data into the specified BigQuery table.
        """
        pst_timezone = pytz.timezone('US/Pacific')
        current_time_pst = datetime.now(pst_timezone)
        formatted_date = current_time_pst.strftime('%m-%d-%Y %I:%M %p')

        row = {
            "cik": data.get('cik'),
            "company": data.get('company'),
            "filing_date": data.get('filing_date'),
            "accession_number": data.get('accession_number'),
            "url": data.get('url')
            # "full_text": data.get('full_text')  # Removed as per schema
        }

        logging.info(f"Inserting metadata for CIK: {row['cik']}, Company: {row['company']}")

        sections_mapping = self.known_sections

        for item_num, column_name in sections_mapping.items():
            if item_num in data['sections']:
                # Assign the entire 8-K content to the relevant section field
                row[column_name] = data['sections'][item_num]
                logging.info(f"Inserting {column_name}: {data['sections'][item_num][:20]}...")
            else:
                row[column_name] = None
                logging.info(f"Inserting {column_name}: None")

        try:
            errors = client_bq.insert_rows_json(full_table_id, [row])
            if errors:
                logging.error(f"Errors occurred while inserting rows: {errors}")
            else:
                logging.info(f"New row inserted successfully for CIK: {row['cik']} at {formatted_date}")
        except Exception as e:
            logging.error(f"Exception occurred while inserting into BigQuery: {e}")

    def preprocess_html(self, html_content, parser):
        """
        Clean and preprocess the HTML content using BeautifulSoup.
        """
        soup = BeautifulSoup(html_content, features=parser)

        # Remove unwanted tags
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
            tag.decompose()

        # Remove comments
        for comment in soup.findAll(text=lambda text: isinstance(text, str) and text.startswith('<!--')):
            comment.extract()

        # Remove empty tags except for structural ones
        for empty_tag in soup.find_all(lambda tag: not tag.contents and tag.name not in ['br', 'hr']):
            empty_tag.decompose()

        # Retain table and important structural tags but remove attributes
        for tag in soup.find_all(['table', 'tr', 'th', 'td', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.attrs = {}

        return str(soup)

    def convert_html_to_markdown(self, html_content):
        """
        Convert cleaned HTML content to Markdown using html2text.
        """
        h = html2text.HTML2Text()
        h.ignore_links = False  # Keep links in the text
        h.ignore_images = True  # Ignore images
        h.ignore_tables = False  # Keep tables in the Markdown output
        h.body_width = 0  # No wrapping
        h.single_line_break = True  # Ensure line breaks are not overly aggressive

        # Convert the cleaned HTML to Markdown
        markdown_text = h.handle(html_content)

        # Decapitalize all text and remove all backslashes
        markdown_text = markdown_text.lower().replace('\\', '')

        return markdown_text.strip()  # Ensure no leading/trailing whitespace

    def extract_sections(self, text):
        """
        Extract specified 8-K sections from the Markdown text using regex.
        """
        original_text = text
        text = text.lower()

        # Regex to capture 8-K item numbers (e.g., item1.01, item2.02)
        item_pattern = re.compile(r'\bitem\s+(\d+\.\d+)\b[.:]?', re.IGNORECASE)
        item_matches = list(item_pattern.finditer(text))
        total_chars = len(text)

        logging.info(f"Total 'Item #' patterns found: {len(item_matches)}")

        section_dict = {}
        item_occurrences = {}

        # Record occurrences of each item
        for match in item_matches:
            item_num = f"item{match.group(1)}"  # e.g., 'item1.01'
            start_pos = match.start()
            end_pos = match.end()

            if item_num not in item_occurrences:
                item_occurrences[item_num] = []
            item_occurrences[item_num].append({'start': start_pos, 'end': end_pos})

        logging.info(f"Recorded occurrences for {len(item_occurrences)} items.")

        # Skip the first occurrence if multiple exist (assuming it's in TOC)
        for item_num, occurrences in item_occurrences.items():
            if len(occurrences) > 1:
                # Skip the first occurrence
                item_occurrences[item_num] = occurrences[1:]
                logging.info(
                    f"Skipped first occurrence of {item_num} (assumed TOC). Remaining occurrences: {len(item_occurrences[item_num])}")
            else:
                # Only one occurrence exists; use it
                logging.info(f"Using single occurrence of {item_num}.")

        # Create a list of items sorted by their position in the text
        sorted_items = []
        for item_num, occurrences in item_occurrences.items():
            for occ in occurrences:
                if item_num in self.target_items:
                    sorted_items.append({'item_num': item_num, 'start': occ['start'], 'end': occ['end']})

        # Sort the items by their start position
        sorted_items.sort(key=lambda x: x['start'])

        logging.info("Sorted target items by their position in the text.")

        # Extract sections based on sorted target items
        for i, current_item in enumerate(sorted_items):
            section_key = current_item['item_num']
            start_pos = current_item['end']

            # Determine end position
            if i + 1 < len(sorted_items):
                end_pos = sorted_items[i + 1]['start']
            else:
                end_pos = total_chars  # Last section goes to end of text

            # Extract section content
            section_content = original_text[start_pos:end_pos].strip()
            section_content = re.sub(r'\s+', ' ', section_content)  # Replace multiple whitespace with single space

            # Map to known sections
            if section_key in self.known_sections:
                section_dict[section_key] = section_content
                logging.info(f"Extracted section '{section_key} - {self.known_sections[section_key]}'.")
            else:
                logging.warning(
                    f"Encountered unknown section '{section_key}' at position {current_item['start']}. Skipping.")

        return section_dict
