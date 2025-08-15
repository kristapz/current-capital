from flask import Flask, request, render_template_string, jsonify, abort
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
from sec_cik_mapper import StockMapper
import logging
import os
import json
import re
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Update the path accordingly

# BigQuery setup for 10-K and 8-K
PROJECT_ID = ''
DATASET_ID = 'backwards_testing'
PROCESSED_10K_TABLE_ID = 'processed_ten_k_filings_new'
PROCESSED_8K_TABLE_ID = 'processed_eight_k_filings_new'

FULL_PROCESSED_10K_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{PROCESSED_10K_TABLE_ID}"
FULL_PROCESSED_8K_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{PROCESSED_8K_TABLE_ID}"

client_bq = bigquery.Client(project=PROJECT_ID)

# Initialize the StockMapper instance
stock_mapper = StockMapper()

# HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Filtered Stocks Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f7f7f7;
        }
        .container {
            max-width: 1400px;
            margin: auto;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            margin-bottom: 30px;
            text-align: center;
        }
        input[type="date"], input[type="text"] {
            padding: 8px;
            margin-right: 10px;
            width: 200px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="submit"], .toggle-btn {
            padding: 8px 16px;
            cursor: pointer;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            text-decoration: none;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            background-color: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
        .message {
            text-align: center;
            color: #ff0000;
            font-weight: bold;
        }
        /* 10-K and 8-K Report Styles */
        .report-container {
            max-width: 1400px;
            margin: 20px auto;
            background-color: #fff;
            padding: 10px;
            border-radius: 6px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        }
        .report-container h1 {
            color: #333;
            text-align: center;
            margin: 10px 0;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .section {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
            overflow: auto;
            max-height: 400px;
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
            display: block;
            padding: 8px;
            overflow-y: auto;
            max-height: 300px;
        }
        p, ul, li {
            margin: 5px 0;
            padding: 0;
            font-size: 0.9em;
        }
        ul {
            list-style-type: none;
            padding-left: 0;
        }
        li {
            margin: 5px 0;
        }
        .highlight {
            font-weight: bold;
            color: #007bff;
        }
        .search-form {
            text-align: center;
            margin-bottom: 10px;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        {% if not report %}
            <h1>Filtered Stocks Dashboard</h1>
            <form method="POST">
                <label for="date">Select Date:</label>
                <input type="date" id="date" name="date" value="{{ selected_date }}">
                <label for="ticker">Ticker Symbol:</label>
                <input type="text" id="ticker" name="ticker" placeholder="e.g., AAPL" value="{{ ticker }}">
                <input type="submit" value="Filter">
            </form>

            {% if message %}
                <p class="message">{{ message }}</p>
            {% endif %}

            {% if data %}
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Addition Count</th>
                            <th>Added Price ($)</th>
                            <th>Recent Price ($)</th>
                            <th>Price Change (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in data %}
                            <tr>
                                <td><a href="{{ url_for('report', ticker=row.ticker) }}">{{ row.ticker }}</a></td>
                                <td>{{ row.addition_count }}</td>
                                <td>{{ row.added_price }}</td>
                                <td>{{ row.recent_added_price_n }}</td>
                                <td>{{ row.price_change }}</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% endif %}
        {% else %}
            <div class="report-container">
                <h1>Report for {{ ticker }}</h1>
                {% if cik and filing_date %}
                <div class="section">
                    <h2>Filing Overview</h2>
                    <div class="section-content">
                        <p><span class="highlight">CIK:</span> {{ cik }}</p>
                        <p><span class="highlight">Filing Date:</span> {{ filing_date }}</p>
                    </div>
                </div>
                {% endif %}

                {% if ten_k_items %}
                    <div class="section">
                        <h2>10-K Report</h2>
                        <div class="section-content">
                            <p><span class="highlight">Filing Date:</span> {{ filing_date }}</p>
                            <div class="grid-container">
                                {% for item in ten_k_items %}
                                    <div class="section">
                                        <h2>{{ item['item_name'] | capitalize }}</h2>
                                        <div class="section-content">
                                            {{ item['formatted_content'] | safe }}
                                            <p><span class="highlight">Processed At:</span> {{ item['processed_at'] }}</p>
                                        </div>
                                    </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                {% else %}
                    <p>No 10-K data available for this ticker.</p>
                {% endif %}

                {% if eight_k_items %}
                    <div class="section">
                        <h2>8-K Filings</h2>
                        {% for filing in eight_k_items %}
                            <div class="filing">
                                <p><strong>Filing Date:</strong> {{ filing.filing_date }}</p>
                                <p><strong>Accession Number:</strong> {{ filing.accession_number }}</p>
                                <p><strong>URL:</strong> <a href="{{ filing.url }}" target="_blank">{{ filing.url }}</a></p>
                                <p><strong>Effect:</strong> {{ filing.effect }}</p>
                                <p><strong>Prediction Text:</strong> {{ filing.stock_prediction.prediction_text }}</p>
                                <h3>Processed Items:</h3>
                                <ul>
                                    {% for item in filing.processed_items %}
                                        <li><strong>{{ item.item_name | capitalize }}:</strong> {{ item.processed_text }}</li>
                                    {% endfor %}
                                </ul>
                            </div>
                            <hr>
                        {% endfor %}
                    </div>
                {% else %}
                    <p>No 8-K filings found for this ticker.</p>
                {% endif %}

                <div style="text-align: center; margin-top: 20px;">
                    <a href="{{ url_for('index') }}" class="toggle-btn">Back to Dashboard</a>
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


# Home route with a search bar
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_date = request.form.get('date')
        ticker = request.form.get('ticker')
    else:
        selected_date = None
        ticker = None

    data = query_filtered_stocks(selected_date, ticker)

    # Default message if no data is found
    if not data:
        message = "No data found for the selected criteria."
    else:
        message = None

    return render_template_string(
        HTML_TEMPLATE,
        data=data,
        message=message,
        selected_date=selected_date or datetime.utcnow().date().isoformat(),
        ticker=ticker or "",
        report=False
    )


# Route to fetch and display both 10-K and 8-K filings
@app.route('/report/<ticker>', methods=['GET'])
def report(ticker):
    if not ticker:
        abort(400, description="Ticker symbol is required.")

    cik = ticker_to_cik(ticker)
    if not cik:
        return f"No CIK found for ticker: {ticker.upper()}"

    # Fetch 10-K data
    ten_k_data = fetch_10k_data_from_bigquery(cik)

    # Fetch 8-K data
    eight_k_data = fetch_8k_data_from_bigquery(ticker.upper())

    if not ten_k_data and not eight_k_data:
        return "No 10-K or 8-K data available for this ticker."

    # Prepare data for rendering
    return render_template_string(
        HTML_TEMPLATE,
        report=True,
        ticker=ticker.upper(),
        cik=cik,
        filing_date=ten_k_data['filing_date'] if ten_k_data else None,
        ten_k_items=ten_k_data['processed_items'] if ten_k_data else [],
        eight_k_items=eight_k_data if eight_k_data else []
    )


# Function to fetch 10-K processed filings from BigQuery
def fetch_10k_data_from_bigquery(cik):
    query = f"""
    SELECT *
    FROM `{FULL_PROCESSED_10K_TABLE_ID}`
    WHERE cik = @cik
    ORDER BY filing_date DESC
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("cik", "STRING", cik)
        ]
    )

    try:
        query_job = client_bq.query(query, job_config=job_config)
        row = next(query_job.result(), None)
        if not row:
            logger.info(f"No 10-K data found for CIK {cik}")
            return None

        processed_items = []
        for item in row.get('processed_items', []):
            processed_text = item.get('processed_text', '')
            parsed_content = is_valid_json(processed_text)

            if should_exclude_item(parsed_content, processed_text):
                logger.info(f"Excluding item '{item.get('item_name')}' due to empty or null content.")
                continue

            formatted_content = format_item_content(parsed_content)

            processed_items.append({
                'item_name': item.get('item_name', '').lower(),  # Lowercased for consistent sorting
                'processed_text': processed_text,
                'processed_at': item.get('processed_at', ''),
                'formatted_content': formatted_content
            })

        # Sort processed items by 'item_name' according to known_sections order
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

        processed_items.sort(
            key=lambda x: list(known_sections.keys()).index(x['item_name']) if x[
                                                                                   'item_name'] in known_sections else len(
                known_sections)
        )

        data = {
            'cik': cik,
            'filing_date': row.get('filing_date').strftime('%Y-%m-%d %H:%M:%S') if row.get('filing_date') else '',
            'processed_items': processed_items
        }

        logger.info(f"Fetched 10-K data for CIK {cik}")
        return data

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"BigQuery API error while fetching 10-K data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching 10-K data: {e}")
        return None


# Function to fetch 8-K processed filings from BigQuery
def fetch_8k_data_from_bigquery(ticker):
    query = f"""
    SELECT *
    FROM `{FULL_PROCESSED_8K_TABLE_ID}`
    WHERE ticker = @ticker
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker)
        ]
    )

    try:
        query_job = client_bq.query(query, job_config=job_config)
        rows = [dict(row) for row in query_job]

        if not rows:
            logger.info(f"No 8-K data found for ticker {ticker}")
            return None

        # Assuming multiple 8-K filings, process each
        eight_k_reports = []
        for row in rows:
            processed_items = []
            for item in row.get('processed_items', []):
                processed_text = item.get('processed_text', '')
                parsed_content = is_valid_json(processed_text)

                if should_exclude_item(parsed_content, processed_text):
                    logger.info(f"Excluding 8-K item '{item.get('item_name')}' due to empty or null content.")
                    continue

                formatted_content = format_item_content(parsed_content)

                processed_items.append({
                    'item_name': item.get('item_name', '').lower(),
                    'processed_text': processed_text,
                    'processed_at': item.get('processed_at', ''),
                    'formatted_content': formatted_content
                })

            # Sort processed items
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

            processed_items.sort(
                key=lambda x: list(known_sections.keys()).index(x['item_name']) if x[
                                                                                       'item_name'] in known_sections else len(
                    known_sections)
            )

            eight_k_reports.append({
                'filing_date': row.get('filing_date').strftime('%Y-%m-%d %H:%M:%S') if row.get('filing_date') else '',
                'accession_number': row.get('accession_number', ''),
                'url': row.get('url', ''),
                'effect': row.get('effect', ''),
                'stock_prediction': row.get('stock_prediction', {}),
                'processed_items': processed_items
            })

        logger.info(f"Fetched {len(eight_k_reports)} 8-K filings for ticker {ticker}")
        return eight_k_reports

    except google_exceptions.GoogleAPIError as e:
        logger.error(f"BigQuery API error while fetching 8-K data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching 8-K data: {e}")
        return None


# Function to fetch filtered stocks based on date and/or ticker
def query_filtered_stocks(selected_date=None, ticker=None):
    """
    Queries the filtered_stocks table based on the selected date and/or ticker.

    Args:
        selected_date (str): Date in 'YYYY-MM-DD' format. Defaults to today if None.
        ticker (str): Ticker symbol to filter. If None, no ticker filter is applied.

    Returns:
        list: List of dictionaries containing the query results.
    """
    if not selected_date:
        selected_date = datetime.utcnow().date().isoformat()

    query = f"""
    SELECT 
        ticker,
        stock_pick.addition_count,
        stock_pick.added_price,
        stock_pick.recent_added_price_n
    FROM `test1-427219.stock_datasets.filtered_stocks`
    WHERE DATE(stock_pick.recent_added_date_n) = @selected_date
    """

    params = [
        bigquery.ScalarQueryParameter("selected_date", "DATE", selected_date)
    ]

    if ticker:
        query += " AND LOWER(ticker) = LOWER(@ticker)"
        params.append(bigquery.ScalarQueryParameter("ticker", "STRING", ticker))

    job_config = bigquery.QueryJobConfig(
        query_parameters=params
    )

    try:
        query_job = client_bq.query(query, job_config=job_config)
        results = query_job.result()
        rows = []
        for row in results:
            price_change = None
            if row.recent_added_price_n and row.added_price:
                try:
                    price_change = ((row.recent_added_price_n - row.added_price) / row.added_price) * 100
                except ZeroDivisionError:
                    price_change = None

            rows.append({
                "ticker": row.ticker,
                "addition_count": row.addition_count,
                "added_price": row.added_price,
                "recent_added_price_n": row.recent_added_price_n,
                "price_change": f"{price_change:.2f}%" if price_change is not None else "N/A"
            })
        logger.info(f"Query successful. Retrieved {len(rows)} rows.")
        return rows
    except google_exceptions.NotFound:
        logger.error(f"Table `filtered_stocks` not found.")
        return []
    except Exception as e:
        logger.error(f"Error querying BigQuery: {e}")
        return []


# Utility functions
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
            if value is None or value == "" or (isinstance(value, list) and not value) or (
                    isinstance(value, dict) and not value):
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


# Function to convert ticker symbol to CIK
def ticker_to_cik(ticker):
    """
    Convert a ticker symbol to its corresponding CIK using the StockMapper.
    """
    if not ticker:
        logger.warning("No ticker provided for conversion.")
        return None

    ticker_upper = ticker.upper()
    cik = stock_mapper.ticker_to_cik.get(ticker_upper)
    if cik:
        logger.info(f"Found CIK {cik} for ticker {ticker_upper}.")
    else:
        logger.warning(f"No CIK found for ticker: {ticker_upper}")
    return cik


# Run the Flask app
if __name__ == "__main__":
    # Optionally, perform upsert operations here
    # filter_and_upsert_data()

    # To avoid port conflicts, specify a different port if needed
    # Example: port=5001
    app.run(debug=True, host='0.0.0.0', port=5009)
