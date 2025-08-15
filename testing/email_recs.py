import os
import time
import schedule
from google.cloud import bigquery
import logging
from datetime import datetime, timedelta
import pytz
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Content

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

# SendGrid API key
sendgrid_api_key = ''

# Email configuration
sender_email = ''  # Replace with your sender email
recipient_emails = ['', '']  # Add recipient emails here

# Set to track recommended stocks
recommended_stocks = set()

def format_sources_html(sources):
    """Formats the sources into a readable HTML string."""
    formatted_sources = []
    for source in sources:
        link = source.get('link', '')
        title = source.get('title', 'No title')
        publication = source.get('publication', 'No publication')
        formatted_sources.append(f"""
        <p><strong>Title:</strong> {title}<br>
        <strong>Publication:</strong> {publication}<br>
        <strong>Link:</strong> <a href="{link}">Read More</a></p>
        """)
    return "".join(formatted_sources)

def send_email(subject, html_body, recipient_emails):
    for recipient_email in recipient_emails:
        try:
            message = Mail(
                from_email=sender_email,
                to_emails=recipient_email,
                subject=subject,
                html_content=html_body
            )
            sg = SendGridAPIClient(sendgrid_api_key)
            response = sg.send(message)
            logging.info(f"Email sent to {recipient_email}: Status code {response.status_code}")
        except Exception as e:
            logging.error(f"Failed to send email to {recipient_email}: {e}")

def query_database():
    # Get current time in PST
    pst = pytz.timezone('US/Pacific')
    current_time_pst = datetime.now(pst)
    time_threshold_pst = current_time_pst - timedelta(hours=7)
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
                    source_info = format_sources_html(row['sources'])
                    stock_info_html = f"""
                    <html>
                    <body>
                        <h2>Stock Update: {prediction['ticker']} - Hurricane Impact Analysis and Price Predictions</h2>
                        <p><strong>Stock:</strong> {prediction['ticker']}</p>
                        <p><strong>Date:</strong> {row['date']}</p>
                        <h3>Source:</h3>
                        {source_info}
                        <h3>Analysis:</h3>
                        <p>{prediction['stock_price_analysis']}</p>
                        <h3>Predicted Stock Prices</h3>
                        <table border="1" cellpadding="8" cellspacing="0">
                            <tr>
                                <th>Time Frame</th>
                                <th>Predicted Price</th>
                            </tr>
                            <tr>
                                <td>In 1 hour</td>
                                <td>{prediction['predicted_price_1hr']}</td>
                            </tr>
                            <tr>
                                <td>In 4 hours</td>
                                <td>{prediction['predicted_price_4hrs']}</td>
                            </tr>
                            <tr>
                                <td>In 24 hours</td>
                                <td>{prediction['predicted_price_24hrs']}</td>
                            </tr>
                        </table>
                        <p>Best regards,<br>[Your Name]</p>
                    </body>
                    </html>
                    """
                    logging.info(f"New stock recommendation: {prediction['ticker']}")
                    send_email(
                        subject=f"Stock Alert: {prediction['ticker']} has a high likelihood trend.",
                        html_body=stock_info_html,
                        recipient_emails=recipient_emails
                    )
                    recommended_stocks.add(stock_id)

# Run the initial check before scheduling
check_predictions()

# Schedule the task to run every 5 minutes
schedule.every(5).minutes.do(check_predictions)

logging.info("Starting the scheduler...")
while True:
    schedule.run_pending()
    time.sleep(1)
