import os
import time
import schedule
from twilio.rest import Client
from google.cloud import bigquery
import logging
from datetime import datetime, timedelta
import pytz

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'backwards_testing'
table_id = 'main'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# Initialize BigQuery client
client = bigquery.Client()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Twilio configuration
account_sid = ''
auth_token = ''
twilio_phone_number = '+'
recipient_phone_numbers = ['+', '+']  # Add additional numbers here
twilio_client = Client(account_sid, auth_token)

# Set to track recommended stocks
recommended_stocks = set()

def send_sms(body, recipient_numbers):
    for number in recipient_numbers:
        try:
            message = twilio_client.messages.create(
                body=body,
                from_=twilio_phone_number,
                to=number
            )
            logging.info(f"SMS sent to {number}: {message.sid}")
        except Exception as e:
            logging.error(f"Failed to send SMS to {number}: {e}")

def query_database():
    # Get current time in PST
    pst = pytz.timezone('US/Pacific')
    current_time_pst = datetime.now(pst)
    time_threshold_pst = current_time_pst - timedelta(hours=2)
    time_threshold_str = time_threshold_pst.strftime('%m-%d-%Y %I:%M %p')

    logging.info(f"Querying for articles since: {time_threshold_str} PST")

    query = f"""
        SELECT *
        FROM `{full_table_id}`
        WHERE DATETIME(PARSE_DATETIME('%m-%d-%Y %I:%M %p', date)) >= PARSE_DATETIME('%m-%d-%Y %I:%M %p', '{time_threshold_str}')
        AND EXISTS (
            SELECT 1
            FROM UNNEST(stock_prediction) AS prediction
            WHERE LOWER(prediction.trend) LIKE '%high likelihood%'
        )
    """
    logging.info(f"Query: {query}")

    query_job = client.query(query)
    results = list(query_job.result())

    logging.info(f"Number of articles found: {len(results)}")

    return results

def check_predictions():
    logging.info("Checking predictions...")
    results = query_database()
    for row in results:
        for prediction in row['stock_prediction']:
            if 'high likelihood' in prediction['trend'].lower():
                stock_id = f"{row['id']}_{prediction['ticker']}"
                if stock_id not in recommended_stocks:
                    stock_info = f"""
                    Stock: {prediction['ticker']}
                    Date: {row['date']}
                    Source: {row['sources']}
                    Analysis: {prediction['stock_price_analysis']}
                    Predicted Price 1hr: {prediction['predicted_price_1hr']}
                    Predicted Price 4hrs: {prediction['predicted_price_4hrs']}
                    Predicted Price 24hrs: {prediction['predicted_price_24hrs']}
                    """
                    logging.info(f"New stock recommendation: {stock_info}")
                    print(stock_info)
                    send_sms(f"Stock Alert: {prediction['ticker']} has a high likelihood trend.\n{stock_info}", recipient_phone_numbers)
                    recommended_stocks.add(stock_id)

# Run the initial check before scheduling
check_predictions()

# Schedule the task to run every 5 minutes
schedule.every(5).minutes.do(check_predictions)

logging.info("Starting the scheduler...")
while True:
    schedule.run_pending()
    time.sleep(1)
