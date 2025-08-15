
from flask import Flask, request, render_template_string, jsonify
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
from sec_cik_mapper import StockMapper
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# BigQuery setup
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Update the path
project_id = ''
dataset_id = 'backwards_testing'
processed_table_id = 'processed_eight_k_filings'
full_processed_table_id = f"{project_id}.{dataset_id}.{processed_table_id}"

client_bq = bigquery.Client(project=project_id)

# HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>8-K Processed Filings Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .container { max-width: 600px; margin: 0 auto; }
        .search-bar { display: flex; margin-bottom: 1em; }
        .search-bar input[type="text"] { flex: 1; padding: 0.5em; }
        .search-bar button { padding: 0.5em; }
        .filings { margin-top: 1em; }
        .filing { margin-bottom: 1em; padding: 1em; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Search for Processed 8-K Filings</h1>
        <form class="search-bar" method="POST" action="/search">
            <input type="text" name="ticker" placeholder="Enter Ticker Symbol" required>
            <button type="submit">Search</button>
        </form>
        {% if ticker %}
            <h2>Results for {{ ticker }}</h2>
            <div class="filings">
                {% if filings %}
                    {% for filing in filings %}
                        <div class="filing">
                            <p><strong>Filing Date:</strong> {{ filing.filing_date }}</p>
                            <p><strong>Effect:</strong> {{ filing.effect }}</p>
                            <p><strong>Prediction Text:</strong> {{ filing.stock_prediction.prediction_text }}</p>
                            <h3>Processed Items:</h3>
                            <ul>
                                {% for item in filing.processed_items %}
                                    <li><strong>{{ item.item_name }}:</strong> {{ item.processed_text }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No processed filings found for this ticker.</p>
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Home route with a search bar
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


# Route to fetch processed 8-K filings based on ticker symbol
@app.route('/search', methods=['POST'])
def search_filings():
    ticker = request.form.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        filings = fetch_processed_items_from_bigquery(ticker)
        return render_template_string(HTML_TEMPLATE, ticker=ticker, filings=filings)

    except Exception as e:
        logging.error(f"Error retrieving filings for ticker {ticker}: {e}")
        return jsonify({"error": "An error occurred while fetching filings"}), 500


# Fetch processed items from the BigQuery processed table for a specific ticker
def fetch_processed_items_from_bigquery(ticker):
    query = f"""
    SELECT *
    FROM `{full_processed_table_id}`
    WHERE ticker = @ticker
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper()),
        ]
    )

    try:
        query_job = client_bq.query(query, job_config=job_config)
        rows = [dict(row) for row in query_job]  # Convert rows to dictionaries
        logging.info(f"Fetched {len(rows)} rows for ticker {ticker}")
        return rows
    except google_exceptions.GoogleAPIError as e:
        logging.error(f"BigQuery API error: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error querying BigQuery: {e}")
        return []


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
