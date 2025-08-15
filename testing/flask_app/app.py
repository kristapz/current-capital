import os
from flask import Flask, render_template, request
from google.cloud import bigquery
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import logging
from datetime import datetime
import numpy as np  # Import numpy for numerical operations

# Initialize Flask app
app = Flask(__name__)

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey3/test1-key.json'
project_id = 'test1-427219'
dataset_id = 'backwards_testing'
table_id = 'main'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# Initialize BigQuery client
client = bigquery.Client()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def query_database():
    query = f"""
        SELECT *
        FROM `{full_table_id}`
        LIMIT 10
    """
    query_job = client.query(query)
    results = query_job.result()
    return results

def format_datetime(dt_str):
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d')
        except ValueError:
            return dt_str
    return dt.strftime('%m-%d-%Y %I:%M %p')

@app.route('/')
def index():
    sort_by = request.args.get('sort_by', 'date')  # Default sorting by date

    logging.info("Querying the database...")
    results = query_database()
    data = []
    logging.info("Processing results...")
    for row in results:
        formatted_date = format_datetime(row['date'])
        for prediction in row['stock_prediction']:
            data.append({
                'id': row['id'],
                'date': formatted_date,
                'sources': row['sources'],
                'content': row['content'][0][:200],  # First 200 characters of the article
                'predictions': prediction,
                'effect': row['effect'],
                'trend': prediction['trend'],
                'percentage_change': prediction.get('%change', 0)
            })
    logging.info(f"Data processed: {len(data)} entries.")

    # Sorting data based on user's choice
    if sort_by == 'percentage_change':
        data.sort(key=lambda x: x['percentage_change'], reverse=True)
    elif sort_by == 'likelihood':
        trend_likelihood = {'High': 3, 'Medium': 2, 'Low': 1}
        data.sort(key=lambda x: trend_likelihood.get(x['trend'], 0), reverse=True)
    else:
        data.sort(key=lambda x: datetime.strptime(x['date'], '%m-%d-%Y %I:%M %p') if x['date'] else datetime.min, reverse=True)

    images = {}
    count = 0
    logging.info("Generating plots...")
    for entry in data:
        content = entry['content']
        if content not in images:
            images[content] = {
                'plots_and_analyses': [],
                'date': entry['date'],
                'sources': entry['sources'],
                'id': entry['id']
            }

        ticker = entry['predictions']['ticker']
        predicted_prices = [
            entry['predictions'].get('predicted_price_1hr'),
            entry['predictions'].get('predicted_price_4hrs'),
            entry['predictions'].get('predicted_price_24hrs')
        ]
        predicted_hours = ['1hr', '4hrs', '24hrs']

        actual_prices = [
            entry['predictions'].get('stock_price_1hr'),
            entry['predictions'].get('stock_price_3hrs'),
            entry['predictions'].get('stock_price_4hrs'),
            entry['predictions'].get('stock_price_5hrs'),
            entry['predictions'].get('stock_price_10hrs'),
            entry['predictions'].get('stock_price_24hrs')
        ]
        actual_hours = ['1hr', '3hrs', '4hrs', '5hrs', '10hrs', '24hrs']

        fig, axs = plt.subplots(1, 2, figsize=(8, 2))
        fig.tight_layout(pad=3.0)

        # Plot predicted prices
        axs[0].plot(predicted_hours, predicted_prices, 'r-', label='Predicted')
        axs[0].set_title(f'Predicted Prices for {ticker}')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Price')
        axs[0].legend()

        # Filter out null values from actual prices and their corresponding times
        filtered_actual_hours = []
        filtered_actual_prices = []

        for hour, price in zip(actual_hours, actual_prices):
            if price is not None:  # Check if the price is not null
                filtered_actual_hours.append(hour)
                filtered_actual_prices.append(price)

        # Convert filtered_actual_hours to numeric values for plotting
        time_mapping = {'1hr': 1, '3hrs': 3, '4hrs': 4, '5hrs': 5, '10hrs': 10, '24hrs': 24}
        numeric_actual_hours = [time_mapping[hour] for hour in filtered_actual_hours]

        # Plot the actual prices as points
        axs[1].plot(numeric_actual_hours, filtered_actual_prices, 'go')  # Plot without labels or best fit line

        axs[1].set_title(f'Actual Prices for {ticker}')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Price')

        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = 'data:image/png;base64,' + string.decode('utf-8')
        plt.close(fig)

        images[content]['plots_and_analyses'].append({
            'uri': uri,
            'ticker': ticker,
            'analysis': entry['predictions']['stock_price_analysis'],
            'percentage_change': entry['percentage_change'],
            'effect': entry['effect'],
            'trend': entry['trend']
        })
        count += 1

    logging.info(f"Generated {count} plots.")
    return render_template('index.html', images=images, sort_by=sort_by)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
