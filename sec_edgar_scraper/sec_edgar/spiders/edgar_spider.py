import scrapy
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode
import pytz
import warnings
from bs4 import BeautifulSoup
import html2text
from rapidfuzz import process, fuzz
import logging
import os

# Suppress ScrapyDeprecationWarning
from scrapy.exceptions import ScrapyDeprecationWarning

warnings.filterwarnings("ignore", category=ScrapyDeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EdgarSpider(scrapy.Spider):
    name = 'edgar_10q'
    allowed_domains = ['sec.gov']

    custom_settings = {
        'USER_AGENT': 'Your Name yourname@example.com',
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': False,
        'LOG_LEVEL': 'ERROR',
        'RETRY_TIMES': 3,  # Max retries for network issues
        'EXTENSIONS': {
            'scrapy.extensions.telnet.TelnetConsole': None,
            'scrapy.extensions.logstats.LogStats': None,
        }
    }

    def __init__(self, days=3, max_documents=5, *args, **kwargs):
        super(EdgarSpider, self).__init__(*args, **kwargs)
        self.days = int(days)
        self.max_documents = int(max_documents)
        self.processed_documents = 0
        self.max_retries = 3  # Custom retry for parsing failures
        # Ensure output directory exists
        self.output_directory = "10Q_files"
        os.makedirs(self.output_directory, exist_ok=True)

    def start_requests(self):
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        end_date = now.strftime('%Y%m%d')
        start_date = (now - timedelta(days=self.days)).strftime('%Y%m%d')

        url = "https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            "action": "getcurrent",
            "datea": start_date,
            "dateb": end_date,
            "owner": "include",
            "type": "10-Q",
            "count": "100",
            "output": "atom"
        }
        yield scrapy.Request(url=f"{url}?{urlencode(params)}", callback=self.parse_feed)

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
                if form == '10-Q':
                    accession_number = recent_filings['accessionNumber'][i].replace('-', '')
                    filing_date = recent_filings['filingDate'][i]
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{recent_filings['primaryDocument'][i]}"

                    # Request to fetch the raw text of the document
                    yield scrapy.Request(
                        url=doc_url,
                        callback=self.process_and_save_text,
                        meta={'cik': cik, 'company': company, 'filing_date': filing_date,
                              'accession_number': accession_number, 'doc_url': doc_url, 'retry_count': 0}
                    )

                    break  # Only process the most recent 10-Q

    def process_and_save_text(self, response):
        cik = response.meta['cik']
        company = response.meta['company']
        filing_date = response.meta['filing_date']
        accession_number = response.meta['accession_number']
        doc_url = response.meta['doc_url']
        retry_count = response.meta['retry_count']

        # Step 1: Detect if content is HTML or XML and parse accordingly
        content_type = 'xml' if '<?xml' in response.text[:100].lower() else 'html'
        parser = 'xml' if content_type == 'xml' else 'lxml'

        cleaned_html = self.preprocess_html(response.text, parser)

        # Step 2: Convert to Markdown using html2text
        markdown_text = self.convert_html_to_markdown(cleaned_html)

        # Step 3: Remove the Table of Contents from Markdown
        markdown_text = self.remove_table_of_contents(markdown_text)

        # Step 4: Extract sections and create JSON output
        sections_list = self.extract_sections_list(markdown_text)

        # Retry logic if no valid sections are found
        if not sections_list and retry_count < self.max_retries:
            logging.warning(f"No valid sections found for {company} (CIK: {cik}). Retrying... (Attempt {retry_count + 1})")
            response.meta['retry_count'] += 1
            yield scrapy.Request(response.url, callback=self.process_and_save_text, meta=response.meta, dont_filter=True)
            return

        # Create filenames for saving
        markdown_filename = os.path.join(self.output_directory, f"{cik}_{filing_date}_10Q.md")
        json_filename = os.path.join(self.output_directory, f"{cik}_{filing_date}_10Q_sections.json")

        # Save the Markdown text to a file
        with open(markdown_filename, 'w', encoding='utf-8') as file:
            file.write(markdown_text)
        logging.info(f"Saved cleaned text for {company} (CIK: {cik}) to {markdown_filename}")

        # Save sections as JSON to a file
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(sections_list, json_file, indent=4)
        logging.info(f"Saved sectioned JSON for {company} (CIK: {cik}) to {json_filename}")

        # Return the information for verification/logging
        yield {
            'cik': cik,
            'company': company,
            'filing_date': filing_date,
            'accession_number': accession_number,
            'url': doc_url,
            'saved_markdown_file': markdown_filename,
            'saved_json_file': json_filename
        }

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

        # Convert the cleaned HTML to Markdown
        markdown_text = h.handle(html_content)

        return markdown_text.strip()  # Ensure no leading/trailing whitespace

    def remove_table_of_contents(self, text):
        # Remove the Table of Contents (TOC) by identifying patterns
        lines = text.splitlines()
        filtered_lines = []
        in_toc = False

        for line in lines:
            # Detect the start of TOC by presence of vertical bars (|) directly adjacent to an item number and page number
            if re.match(r'^\s*\|\s*item\s*\d+\s*\|\s*page\s*\d+\s*\|', line, re.IGNORECASE):
                in_toc = True
                continue
            # Once TOC ends, we can continue adding lines
            if in_toc and not line.strip():
                in_toc = False
            if not in_toc:
                filtered_lines.append(line)

        # Return the cleaned text without the TOC
        return '\n'.join(filtered_lines).strip()

    def extract_sections_list(self, text):
        # Extract sections for 10-Q based on known section names
        known_sections = {
            "item 1": "financial statements",
            "item 2": "management's discussion and analysis of financial condition and results of operations",
            "item 3": "quantitative and qualitative disclosures about market risk",
            "item 4": "controls and procedures",
            "item 1": "legal proceedings",
            "item 1a": "risk factors",
            "item 2": "unregistered sales of equity securities and use of proceeds",
            "item 3": "defaults upon senior securities",
            "item 4": "mine safety disclosures",
            "item 5": "other information",
            "item 6": "exhibits"
        }

        # Extract sections and store them in a list
        sections_list = []
        found_items = set()  # Track found items

        # Find all mentions of item names in the text
        for item_num, expected_section_name in known_sections.items():
            matches = re.finditer(rf'{item_num}\.', text.lower())
            item_instances = [match for match in matches]

            # Ensure there are at least two instances for each item
            if len(item_instances) >= 2:
                for instance_index in range(len(item_instances) - 1):
                    section_text = text[item_instances[instance_index].end():item_instances[instance_index + 1].start()]
                    matched_section = process.extractOne(section_text, [expected_section_name], scorer=fuzz.token_set_ratio)
                    if matched_section and matched_section[1] >= 80:  # 80 is the threshold for a strong match
                        sections_list.append((item_num, section_text.strip()))
                        found_items.add(item_num)

        return sections_list
