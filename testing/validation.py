import os
import time
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


def query_articles_with_high_effect():
    # Get current time in PST
    pst = pytz.timezone('US/Pacific')
    current_time_pst = datetime.now(pst)
    time_threshold_pst = current_time_pst - timedelta(weeks=15)
    time_threshold_str = time_threshold_pst.strftime('%m-%d-%Y %I:%M %p')

    logging.info(f"Querying for articles with high effect since: {time_threshold_str} PST")

    # Query for articles with high or very high effect
    query = f"""
        SELECT 
            id,
            date,
            sources,
            effect
        FROM `{full_table_id}`
        WHERE DATETIME(PARSE_DATETIME('%m-%d-%Y %I:%M %p', date)) >= PARSE_DATETIME('%m-%d-%Y %I:%M %p', '{time_threshold_str}')
        AND LOWER(effect) IN ('high', 'very high')
    """
    logging.info(f"Query: {query}")

    query_job = client.query(query)
    results = list(query_job.result())

    logging.info(f"Number of articles found with high effect: {len(results)}")

    return results


def query_all_stock_predictions(article_ids):
    if not article_ids:
        logging.info("No articles found with high effect to query further.")
        return []

    # Join article_ids for SQL IN clause
    article_ids_str = ', '.join([str(id) for id in article_ids])

    # Query for all stock predictions within articles with high effect
    query = f"""
        SELECT 
            id,
            date,
            sources,
            model,
            ticker,
            trend,
            stock_price_analysis,
            predicted_price_1hr,
            predicted_price_4hrs,
            predicted_price_24hrs,
            stock_price_1hr,
            stock_price_2hrs,
            stock_price_3hrs,
            stock_price_5hrs,
            stock_price_10hrs,
            stock_price_24hrs
        FROM `{full_table_id}`,
        UNNEST(stock_prediction) AS stock_prediction
        WHERE id IN ({article_ids_str})
    """
    logging.info(f"Query: {query}")

    query_job = client.query(query)
    results = list(query_job.result())

    logging.info(f"Number of stock predictions found: {len(results)}")

    return results


def filter_predictions_with_high_trend(predictions):
    # Filter the predictions to only include those with a high likelihood in the trend description
    filtered_predictions = [
        prediction for prediction in predictions
        if 'high likelihood' in prediction['trend'].lower() and prediction['stock_price_1hr'] is not None
    ]
    logging.info(f"Number of stock predictions with high trend: {len(filtered_predictions)}")
    return filtered_predictions


def calculate_percentage_change(old_price, new_price):
    if old_price is None or new_price is None:
        return None
    try:
        return ((new_price - old_price) / old_price) * 100
    except ZeroDivisionError:
        return None


def display_results():
    # Step 1: Query articles with high effect
    high_effect_articles = query_articles_with_high_effect()

    # Extract article IDs for the next query
    article_ids = [article['id'] for article in high_effect_articles]

    # Step 2: Query all stock predictions within these articles
    all_predictions = query_all_stock_predictions(article_ids)

    # Step 3: Filter stock predictions with high trend
    high_trend_predictions = filter_predictions_with_high_trend(all_predictions)

    # Prepare the results with calculated percentage changes
    ranked_results = []
    for row in high_trend_predictions:
        # Calculate percentage changes
        actual_change = calculate_percentage_change(row['stock_price_1hr'], row['stock_price_24hrs'])
        predicted_change = calculate_percentage_change(row['predicted_price_1hr'], row['predicted_price_24hrs'])

        # Disqualify predictions with 0% actual price change
        if actual_change is None or actual_change == 0:
            continue

        # Check if the predicted and actual changes have the same direction (both positive or both negative)
        correct_direction = (actual_change > 0 and predicted_change > 0) or (actual_change < 0 and predicted_change < 0)

        # Calculate proximity (absolute difference) between predicted and actual changes
        proximity = abs(actual_change - predicted_change) if predicted_change is not None else None

        # If the direction is correct, calculate weighted proximity, otherwise set a high value for incorrect direction
        if correct_direction and proximity is not None:
            # Weighted proximity gives more weight to larger price changes
            weighted_proximity = proximity / abs(actual_change)
        else:
            # Penalize incorrect direction predictions
            weighted_proximity = float('inf')

        # Extract the link from the sources field
        link = row['sources'][0]['link'] if row['sources'] else 'No link available'

        # Append to results list
        ranked_results.append({
            'stock': row['ticker'],
            'link': link,
            'actual_change': actual_change,
            'predicted_change': predicted_change,
            'proximity': proximity,
            'weighted_proximity': weighted_proximity,
            'correct_direction': correct_direction
        })

    # Sort results by weighted proximity (ascending order)
    ranked_results.sort(key=lambda x: (x['weighted_proximity'], not x['correct_direction']))

    # Display the ranked results
    for result in ranked_results:
        direction_status = "Correct Direction" if result['correct_direction'] else "Wrong Direction"
        print(f"Stock: {result['stock']}, Link: {result['link']}")
        print(
            f"Actual Price Change (1hr to 24hrs): {result['actual_change']:.2f}%, Predicted Price Change (1hr to 24hrs): {result['predicted_change']:.2f}%")
        print(f"Weighted Proximity to Actual: {result['weighted_proximity']:.2f} ({direction_status})")
        print("-" * 80)


# Run the check to display predictions
display_results()
