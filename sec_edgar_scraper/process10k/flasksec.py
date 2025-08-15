from flask import Flask, render_template_string, request
from google.cloud import bigquery
import json
import os
import logging
import re
from sec_cik_mapper import StockMapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Initialize the StockMapper instance
stock_mapper = StockMapper()

def is_valid_json(text):
    """
    Check if a given text is valid JSON.
    Returns the parsed JSON if valid, otherwise returns the raw text.
    """
    cleaned_text = text.strip()

    # Use regex to extract JSON content within code blocks
    match = re.match(r'^```json\s*(.*?)\s*```$', cleaned_text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_text = match.group(1).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        if cleaned_text.lower() == 'null':
            return None
        elif cleaned_text == '':
            return ''
        return text

def has_meaningful_content(content):
    """
    Recursively check if the JSON content contains any meaningful data.
    Returns True if meaningful data exists, False otherwise.
    """
    if isinstance(content, dict):
        for value in content.values():
            if has_meaningful_content(value):
                return True
        return False
    elif isinstance(content, list):
        for item in content:
            if has_meaningful_content(item):
                return True
        return False
    elif isinstance(content, bool):
        return content  # True is meaningful, False is not
    elif isinstance(content, (int, float)):
        return content != 0
    elif isinstance(content, str):
        return content.strip() != '' and content.strip().lower() != 'none applicable'
    elif content is None:
        return False
    else:
        return False

def should_exclude_item(parsed_content, processed_text):
    """
    Determine whether an item should be excluded based on its content.
    """
    if parsed_content is None:
        return True
    if isinstance(parsed_content, str):
        if processed_text.lower().strip() == "null" or processed_text.strip() == "":
            return True
    if isinstance(parsed_content, (dict, list)):
        if not has_meaningful_content(parsed_content):
            return True
    return False

def format_json_content(content):
    """
    Recursively format JSON content to render it as HTML, skipping empty or None values.
    """
    if isinstance(content, dict):
        html = "<ul>"
        for key, value in sorted(content.items()):  # Sorting keys for consistency
            if value is None or value == "" or (isinstance(value, list) and not value) or (isinstance(value, dict) and not value):
                # Skip empty or None values
                continue
            html += f"<li><strong>{key}:</strong> {format_json_content(value)}</li>"
        html += "</ul>"
        return html
    elif isinstance(content, list):
        if not content:
            return ""  # Skip empty lists
        html = "<ul>"
        for item in content:
            html += f"<li>{format_json_content(item)}</li>"
        html += "</ul>"
        return html
    else:
        return f"{content}"

def format_item_content(content):
    """
    Format the content of an item.
    """
    if isinstance(content, dict):
        # Render formatted JSON
        return format_json_content(content)
    else:
        # Return plain text if the content is not JSON
        return f"<p>{content}</p>"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>10-K Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            background-color: #f7f7f7;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: #fff;
            padding: 10px;
            border-radius: 6px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin: 10px 0;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); /* Enhanced for responsiveness */
            gap: 10px;
            margin-top: 10px;
        }
        .section {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
            overflow: auto; /* Prevent overflow */
            max-height: 400px; /* Increased max height to prevent overflow */
        }
        .section h2 {
            margin: 0;
            padding: 8px;
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
            text-align: center;
        }
        .section-content {
            display: block;  /* Expanded by default */
            padding: 8px;
            overflow-y: auto; /* Add scroll if content overflows */
            max-height: 300px; /* Adjusted max height for content */
        }
        p, ul, li {
            margin: 5px 0;
            padding: 0;
            font-size: 0.9em;
        }
        ul {
            list-style-type: none; /* Removed bullets for a cleaner look */
            padding-left: 0;
        }
        li {
            margin: 5px 0;
        }
        .highlight {
            font-weight: bold;
            color: #007bff;
        }
        .toggle-btn {
            cursor: pointer;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            margin: 5px;
        }
        .search-form {
            text-align: center;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>10-K Report</h1>
        <!-- Search form for Ticker symbol -->
        <div class="search-form">
            <form method="get" action="/">
                <input type="text" name="ticker" placeholder="Enter Ticker Symbol" required>
                <button type="submit" class="toggle-btn">Search</button>
            </form>
        </div>
        {% if cik and filing_date %}
        <div class="section">
            <h2>Filing Overview</h2>
            <div class="section-content">
                <p><span class="highlight">CIK:</span> {{ cik }}</p>
                <p><span class="highlight">Filing Date:</span> {{ filing_date }}</p>
            </div>
        </div>
        {% endif %}
        <div class="grid-container">
            {% for item in processed_items %}
            <div class="section">
                <h2>{{ item['item_name'] }}</h2>
                <div class="section-content">
                    {{ item['formatted_content'] | safe }}
                    <p><span class="highlight">Processed At:</span> {{ item['processed_at'] }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

def fetch_data_from_bigquery(cik=None):
    # Setup environment with hardcoded path
    credentials_path = '/Users/kristapszilgalvis/Desktop/current-capital/sec_edgar_scraper/testkeysec/test1-key.json'
    if not os.path.exists(credentials_path):
        logging.error(f"Credentials file not found at path: {credentials_path}")
        return []

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    logging.info("Environment variable for BigQuery credentials set.")

    project_id = 'test1-427219'
    dataset_id = 'backwards_testing'
    table_id = 'processed_ten_k_filings_new'
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # Initialize BigQuery client
    try:
        client = bigquery.Client()
        logging.info("BigQuery client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize BigQuery client: {e}")
        return []

    # Query to get the most recent filing or by CIK if provided
    if cik:
        logging.info(f"Fetching data for CIK: {cik}")
        query = f"""
        SELECT * FROM `{full_table_id}`
        WHERE cik = @cik
        ORDER BY filing_date DESC
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cik", "STRING", cik)
            ]
        )
    else:
        logging.info("Fetching the most recent filing.")
        query = f"""
        SELECT * FROM `{full_table_id}`
        ORDER BY filing_date DESC
        LIMIT 1
        """
        job_config = None

    try:
        query_job = client.query(query, job_config=job_config)
        rows = list(query_job)
        logging.info(f"Number of rows fetched: {len(rows)}")
    except Exception as e:
        logging.error(f"Error querying BigQuery: {e}")
        return []

    if not rows:
        logging.info("No rows returned from BigQuery.")
        return []

    # Define the known sections in order
    known_sections = {
        "item1": "business",
        "item1a": "risk factors",
        "item1b": "unresolved staff comments",
        "item1c": "cybersecurity",
        "item2": "properties",
        "item3": "legal proceedings",
        "item4": "mine safety disclosures",
        "item5": "market for registrant's common equity, related stockholder matters and issuer purchases of equity securities",
        "item6": "selected financial data",
        "item7": "management's discussion and analysis of financial condition and results of operations",
        "item7a": "quantitative and qualitative disclosures about market risk",
        "item8": "financial statements and supplementary data",
        "item9": "changes in and disagreements with accountants on accounting and financial disclosure",
        "item9a": "controls and procedures",
        "item9b": "other information",
        "item9c": "disclosure regarding foreign jurisdictions that prevent inspections",
        "item10": "directors, executive officers and corporate governance",
        "item11": "executive compensation",
        "item12": "security ownership of certain beneficial owners and management and related stockholder matters",
        "item13": "certain relationships and related transactions, and director independence",
        "item14": "principal accountant fees and services",
        "item15": "exhibits, financial statement schedules",
        "item16": "form 10-k summary"
    }

    # Convert rows to list of dictionaries for easier processing
    data = []
    for row in rows:
        processed_items = []
        for item in row.get('processed_items', []):
            # Validate and parse JSON or return plain text
            processed_text = item.get('processed_text', '')
            parsed_content = is_valid_json(processed_text)

            # Exclude items with None, "null" string, or empty content
            if should_exclude_item(parsed_content, processed_text):
                logging.info(f"Excluding item '{item.get('item_name')}' due to empty or null content.")
                continue

            formatted_content = format_item_content(parsed_content)

            processed_items.append({
                'item_name': item.get('item_name', '').lower(),  # Lowercased for consistent sorting
                'processed_text': processed_text,
                'processed_at': item.get('processed_at', ''),
                'formatted_content': formatted_content
            })

        # Sort processed items by 'item_name' according to known_sections order
        processed_items.sort(
            key=lambda x: list(known_sections.keys()).index(x['item_name']) if x['item_name'] in known_sections else len(known_sections)
        )

        data.append({
            'cik': row.get('cik', ''),
            'filing_date': row.get('filing_date').strftime('%Y-%m-%d %H:%M:%S') if row.get('filing_date') else '',
            'processed_items': processed_items
        })

    return data

def ticker_to_cik(ticker):
    """
    Convert a ticker symbol to its corresponding CIK using the StockMapper.
    """
    if not ticker:
        logging.warning("No ticker provided for conversion.")
        return None

    ticker_upper = ticker.upper()
    cik = stock_mapper.ticker_to_cik.get(ticker_upper)
    if cik:
        logging.info(f"Found CIK {cik} for ticker {ticker_upper}.")
    else:
        logging.warning(f"No CIK found for ticker: {ticker_upper}")
    return cik

@app.route('/')
def index():
    # Get the ticker symbol from the request, if any
    ticker = request.args.get('ticker')

    if ticker:
        cik = ticker_to_cik(ticker)
        if not cik:
            return f"No CIK found for ticker: {ticker.upper()}"
    else:
        cik = None

    # Fetch data from BigQuery
    data = fetch_data_from_bigquery(cik)

    if not data:
        logging.info("No data to render.")
        return "No data available from BigQuery."

    # Render the HTML template with the fetched JSON data
    return render_template_string(
        HTML_TEMPLATE,
        cik=data[0]['cik'],
        filing_date=data[0]['filing_date'],
        processed_items=data[0]['processed_items']
    )

if __name__ == "__main__":
    app.run(debug=True)
