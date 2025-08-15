import os
from google.cloud import bigquery
from google.api_core.exceptions import Conflict, NotFound

# Set your Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
project_id = ''
dataset_id = 'backwards_testing'
table_id = 'calls_together'
full_table_id = f"{project_id}.{dataset_id}.{table_id}"


# Initialize BigQuery Client
client = bigquery.Client(project=project_id)

def create_or_replace_table():
    try:
        # Define the table schema
        table_schema = [
            bigquery.SchemaField("id", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("date", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("content", "STRING", mode="REPEATED"),
            bigquery.SchemaField("updated", "STRING", mode="REPEATED"),
            bigquery.SchemaField("sources", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("id", "INTEGER"),
                bigquery.SchemaField("link", "STRING"),
                bigquery.SchemaField("publication", "STRING"),
                bigquery.SchemaField("title", "STRING")
            ]),
            bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embeddings", "RECORD", mode="NULLABLE", fields=[
                bigquery.SchemaField("model1", "STRING"),
                bigquery.SchemaField("model2", "STRING"),
                bigquery.SchemaField("model3", "STRING"),
                bigquery.SchemaField("model4", "STRING")
            ]),
            bigquery.SchemaField("stock_prediction", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("model", "STRING"),
                bigquery.SchemaField("ticker", "STRING"),
                bigquery.SchemaField("predicted_price_1hr", "FLOAT"),
                bigquery.SchemaField("predicted_price_4hrs", "FLOAT"),
                bigquery.SchemaField("predicted_price_24hrs", "FLOAT"),
                bigquery.SchemaField("stock_price_analysis", "STRING"),
                bigquery.SchemaField("stock_price_1hr", "FLOAT"),
                bigquery.SchemaField("stock_price_2hrs", "FLOAT"),
                bigquery.SchemaField("stock_price_3hrs", "FLOAT"),
                bigquery.SchemaField("stock_price_5hrs", "FLOAT"),
                bigquery.SchemaField("stock_price_10hrs", "FLOAT"),
                bigquery.SchemaField("stock_price_24hrs", "FLOAT")
            ])
        ]

        # Create or replace the table
        table = bigquery.Table(full_table_id, schema=table_schema)
        client.delete_table(full_table_id, not_found_ok=True)  # Delete the table if it exists
        client.create_table(table)  # Create a new table
        print(f"Table {full_table_id} created or replaced successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_or_replace_table()
