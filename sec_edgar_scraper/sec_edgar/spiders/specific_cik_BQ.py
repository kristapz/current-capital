import scrapy
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode
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

# Setup environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = 'test1-427219'
dataset_id = 'backwards_testing'
table_id = 'ten_k_filings'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# Initialize BigQuery Client
client_bq = bigquery.Client(project=project_id)

stock_mapper = StockMapper()

class EdgarSpider(scrapy.Spider):
    name = 'edgarcisk'
    allowed_domains = ['sec.gov']

    custom_settings = {
        'USER_AGENT': 'Your Name yourname@example.com',
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': False,
        'LOG_LEVEL': 'ERROR',
        'EXTENSIONS': {  # Disable unnecessary extensions
            'scrapy.extensions.telnet.TelnetConsole': None,
            'scrapy.extensions.logstats.LogStats': None,
        }
    }

    def __init__(self, ticker=None, days=10000, max_documents=1, *args, **kwargs):
        super(EdgarSpider, self).__init__(*args, **kwargs)
        if not ticker:
            self.logger.error("Ticker argument is required. Usage: scrapy crawl edgarcisk -a ticker=XYZ")
            raise ValueError("Ticker argument is required. Usage: scrapy crawl edgarcisk -a ticker=XYZ")
        self.ticker = ticker.upper()
        self.days = int(days)
        self.max_documents = int(max_documents)
        self.processed_documents = 0

    def start_requests(self):
        specific_cik = stock_mapper.ticker_to_cik.get(self.ticker)
        if not specific_cik:
            self.logger.error(f"CIK not found for ticker: {self.ticker}")
            return

        # Optionally, you can fetch the company name from the CIK submission
        url = f"https://data.sec.gov/submissions/CIK{specific_cik.zfill(10)}.json"

        yield scrapy.Request(
            url=url,
            callback=self.parse_company_filings,
            meta={'cik': specific_cik, 'company': self.ticker}  # Temporarily using ticker as company name
            # Replace 'self.ticker' with actual company name if available
        )

    def parse_feed(self, response):
        if self.processed_documents >= self.max_documents:
            return

        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in response.xpath('//atom:entry', namespaces=namespace):
            company = entry.xpath('atom:title/text()', namespaces=namespace).get().split(' - ')[0]
            cik = re.search(r'\((\d+)\)', entry.xpath('atom:title/text()', namespaces=namespace).get()).group(1)
            filing_link = entry.xpath('atom:link[@type="text/html"]/@href', namespaces=namespace).get()

            yield scrapy.Request(
                url=f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
                callback=self.parse_company_filings,
                meta={'cik': cik, 'company': company}
            )

            self.processed_documents += 1
            if self.processed_documents >= self.max_documents:
                break

    def parse_company_filings(self, response):
        data = json.loads(response.text)
        cik = response.meta['cik']
        company = response.meta['company']

        recent_filings = data.get('filings', {}).get('recent', {})
        if 'form' in recent_filings:
            for i, form in enumerate(recent_filings['form']):
                if form == '10-K':
                    accession_number = recent_filings['accessionNumber'][i].replace('-', '')
                    filing_date = recent_filings['filingDate'][i]
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{recent_filings['primaryDocument'][i]}"

                    # Request to fetch the raw text of the document
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

                    break  # Only process the most recent 10-K

    def process_and_save_text(self, response):
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

        # Combine metadata and sections into a single dictionary
        combined_output = {
            'cik': cik,
            'company': company,
            'filing_date': filing_date,
            'accession_number': accession_number,
            'url': doc_url,
            'sections': sections_json  # Use full sections for BigQuery insertion
        }

        # Insert the data into BigQuery
        self.insert_10k_data_to_bigquery(combined_output)

    def insert_10k_data_to_bigquery(self, data):
        pst_timezone = pytz.timezone('US/Pacific')
        current_time_pst = datetime.now(pst_timezone)
        formatted_date = current_time_pst.strftime('%m-%d-%Y %I:%M %p')

        row = {
            "cik": data.get('cik'),
            "company": data.get('company'),
            "filing_date": data.get('filing_date'),
            "accession_number": data.get('accession_number'),
            "url": data.get('url')
        }

        logging.info(f"Inserting metadata for CIK: {row['cik']}, Company: {row['company']}")

        sections_mapping = {
            "item1_business": 'item1',
            "item1a_risk_factors": 'item1a',
            "item1b_unresolved_staff_comments": 'item1b',
            "item1c_cybersecurity": 'item1c',
            "item2_properties": 'item2',
            "item3_legal_proceedings": 'item3',
            "item4_mine_safety_disclosures": 'item4',
            "item5_market_for_registrants_common_equity_related_stockholder_matters_and_issuer_purchases_of_equity_securities": 'item5',
            "item6_selected_financial_data": 'item6',
            "item7_management_discussion_and_analysis": 'item7',
            "item7a_quantitative_and_qualitative_disclosures_about_market_risk": 'item7a',
            "item8_financial_statements_and_supplementary_data": 'item8',
            "item9_changes_in_and_disagreements_with_accountants_on_accounting_and_financial_disclosure": 'item9',
            "item9a_controls_and_procedures": 'item9a',
            "item9b_other_information": 'item9b',
            "item9c_disclosure_regarding_foreign_jurisdictions_that_prevent_inspections": 'item9c',
            "item10_directors_executive_officers_and_corporate_governance": 'item10',
            "item11_executive_compensation": 'item11',
            "item12_security_ownership_of_certain_beneficial_owners_and_management_and_related_stockholder_matters": 'item12',
            "item13_certain_relationships_and_related_transactions_and_director_independence": 'item13',
            "item14_principal_accountant_fees_and_services": 'item14',
            "item15_exhibits_financial_statement_schedules": 'item15',
            "item16_form_10k_summary": 'item16'
        }

        for bigquery_column, section_key in sections_mapping.items():
            section_content = data['sections'].get(section_key)
            row[bigquery_column] = section_content

            if section_content:
                logging.info(f"Inserting {bigquery_column}: {section_content[:20]}...")
            else:
                logging.info(f"Inserting {bigquery_column}: None")

        errors = client_bq.insert_rows_json(full_table_id, [row])
        if errors:
            logging.error(f"Errors occurred while inserting rows: {errors}")
        else:
            logging.info(f"New row inserted successfully for CIK: {row['cik']} at {formatted_date}")

    def preprocess_html(self, html_content, parser):
        # Use BeautifulSoup with the appropriate parser
        soup = BeautifulSoup(html_content, features=parser)

        # Remove unwanted tags to clean up the HTML
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
            tag.decompose()

        # Remove comments to reduce noise
        for comment in soup.findAll(text=lambda text: isinstance(text, (str,)) and text.startswith('<!--')):
            comment.extract()

        # Additional cleanup (e.g., empty tags)
        for empty_tag in soup.find_all(lambda tag: not tag.contents and tag.name not in ['br', 'hr']):
            empty_tag.decompose()

        # Retain table tags and important structural tags but remove unnecessary attributes
        for tag in soup.find_all(['table', 'tr', 'th', 'td', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.attrs = {}  # Clear all attributes

        # Return the cleaned HTML as a string
        return str(soup)

    def convert_html_to_markdown(self, html_content):
        # Create an html2text object with options to customize the output
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
        original_text = text
        text = text.lower()

        # Revised regex without start of line anchor and more flexible spacing/punctuation
        item_pattern = re.compile(r'\bitem\s*\.?\s*(\d+[a-z]?)\b[.:]?', re.IGNORECASE)
        item_matches = list(item_pattern.finditer(text))
        total_chars = len(text)

        logging.info(f"Total 'Item #' patterns found: {len(item_matches)}")
        for match in item_matches:
            logging.debug(f"Match found: '{match.group(0)}' at position {match.start()}")

        known_sections = {
            "item1": "business",
            "item1a": "risk factors",
            "item1b": "unresolved staff comments",
            "item1c": "cybersecurity",
            "item2": "properties",
            "item3": "legal proceedings",
            "item4": "mine safety disclosures",
            "item5": "market for registrants common equity related stockholder matters and issuer purchases of equity securities",
            "item6": "selected financial data",
            "item7": "management discussion and analysis",
            "item7a": "quantitative and qualitative disclosures about market risk",
            "item8": "financial statements and supplementary data",
            "item9": "changes in and disagreements with accountants on accounting and financial disclosure",
            "item9a": "controls and procedures",
            "item9b": "other information",
            "item9c": "disclosure regarding foreign jurisdictions that prevent inspections",
            "item10": "directors executive officers and corporate governance",
            "item11": "executive compensation",
            "item12": "security ownership of certain beneficial owners and management and related stockholder matters",
            "item13": "certain relationships and related transactions and director independence",
            "item14": "principal accountant fees and services",
            "item15": "exhibits financial statement schedules",
            "item16": "form 10-k summary"
        }

        section_dict = {}
        item_occurrences = {}

        # Step 1: Record all occurrences of each item
        for match in item_matches:
            item_num = f"item{match.group(1)}"  # e.g., 'item1', 'item1a'
            start_pos = match.start()
            end_pos = match.end()

            if item_num not in item_occurrences:
                item_occurrences[item_num] = []
            item_occurrences[item_num].append({'start': start_pos, 'end': end_pos})

        logging.info(f"Recorded occurrences for {len(item_occurrences)} items.")

        # Step 2: Skip the first occurrence if multiple exist (assuming it's in TOC)
        for item_num, occurrences in item_occurrences.items():
            if len(occurrences) > 1:
                # Skip the first occurrence
                item_occurrences[item_num] = occurrences[1:]
                logging.info(
                    f"Skipped first occurrence of {item_num} (assumed TOC). Remaining occurrences: {len(item_occurrences[item_num])}")
            else:
                # Only one occurrence exists; use it
                logging.info(f"Using single occurrence of {item_num}.")

        # Step 3: Create a list of items sorted by their position in the text
        sorted_items = []
        for item_num, occurrences in item_occurrences.items():
            for occ in occurrences:
                sorted_items.append({'item_num': item_num, 'start': occ['start'], 'end': occ['end']})

        # Sort the items by their start position
        sorted_items.sort(key=lambda x: x['start'])

        logging.info("Sorted items by their position in the text.")

        # Step 4: Sequentially extract sections
        ordered_items = []
        expected_sequence = [
            "item1", "item1a", "item1b", "item1c", "item2", "item3",
            "item4", "item5", "item6", "item7", "item7a", "item8",
            "item9", "item9a", "item9b", "item9c", "item10",
            "item11", "item12", "item13", "item14", "item15", "item16"
        ]

        # Create a mapping of item to its occurrence(s)
        item_to_occurrences = {item['item_num']: [] for item in sorted_items}
        for item in sorted_items:
            item_to_occurrences[item['item_num']].append(item)

        # Iterate through the expected sequence and collect items in order
        for idx, expected_item in enumerate(expected_sequence):
            occurrences = item_to_occurrences.get(expected_item, [])
            if not occurrences:
                logging.warning(f"Missing expected section: {expected_item}.")
                continue  # Skip if the expected item is missing

            # Assume the first occurrence is the actual section after skipping TOC
            current_item = occurrences[0]

            # Determine the end position
            # Look for the next expected item in the sequence
            end_pos = total_chars  # Default to end of text
            for next_idx in range(idx + 1, len(expected_sequence)):
                next_expected_item = expected_sequence[next_idx]
                next_occurrences = item_to_occurrences.get(next_expected_item, [])
                if next_occurrences:
                    # Use the first occurrence of the next expected item
                    end_pos = next_occurrences[0]['start']
                    break

            # Extract section content
            section_content = original_text[current_item['end']:end_pos].strip()
            section_content = re.sub(r'\s+', ' ', section_content)  # Replace multiple whitespace with single space

            # Map to known sections
            if expected_item in known_sections:
                section_key = expected_item
                section_dict[section_key] = section_content
                logging.info(f"Extracted section '{section_key} - {known_sections[expected_item]}'.")
            else:
                logging.warning(
                    f"Encountered unknown section '{expected_item}' at position {current_item['start']}. Skipping.")

        return section_dict
