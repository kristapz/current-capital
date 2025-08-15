# create_bigquery_table.py

import os
import logging
from google.cloud import bigquery
from google.api_core.exceptions import Conflict, NotFound

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_bigquery_table():
    # Setup environment
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Update the path if necessary
    project_id = 'test1-427219'
    dataset_id = 'backwards_testing'
    table_id = 'eight_k_filings'  # Consider renaming to 'eight_k_filings' for clarity
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # Initialize BigQuery Client
    client = bigquery.Client(project=project_id)

    # Define the schema for the table
    schema = [
        bigquery.SchemaField("cik", "STRING", mode="REQUIRED", description="Central Index Key of the company"),
        bigquery.SchemaField("company", "STRING", mode="REQUIRED", description="Stock ticker of the company"),
        bigquery.SchemaField("filing_date", "DATE", mode="REQUIRED", description="Date of the 8-K filing"),
        bigquery.SchemaField("accession_number", "STRING", mode="REQUIRED", description="Accession number of the filing"),
        bigquery.SchemaField("url", "STRING", mode="REQUIRED", description="URL to the 8-K filing on SEC EDGAR"),
        bigquery.SchemaField("entry_into_a_material_definitive_agreement", "STRING", mode="NULLABLE", description="Content of Item 1.01"),
        bigquery.SchemaField("termination_of_a_material_definitive_agreement", "STRING", mode="NULLABLE", description="Content of Item 1.02"),
        bigquery.SchemaField("completion_of_acquisition_or_disposition_of_assets", "STRING", mode="NULLABLE", description="Content of Item 2.01"),
        bigquery.SchemaField("results_of_operations_and_financial_condition", "STRING", mode="NULLABLE", description="Content of Item 2.02"),
        bigquery.SchemaField("creation_of_a_direct_financial_obligation", "STRING", mode="NULLABLE", description="Content of Item 2.03"),
        bigquery.SchemaField("triggering_events_that_increase_financial_obligation", "STRING", mode="NULLABLE", description="Content of Item 2.04"),
        bigquery.SchemaField("departure_of_directors_or_officers", "STRING", mode="NULLABLE", description="Content of Item 5.02"),
    ]

    # Reference to the dataset
    dataset_ref = client.dataset(dataset_id)

    try:
        dataset = client.get_dataset(dataset_ref)
        logging.info(f"Dataset '{dataset_id}' found in project '{project_id}'.")
    except NotFound:
        logging.error(f"Dataset '{dataset_id}' not found in project '{project_id}'. Please create the dataset first.")
        return

    # Reference to the table
    table_ref = dataset_ref.table(table_id)

    try:
        table = client.get_table(table_ref)  # Make an API request.
        logging.info(f"Table '{table_id}' already exists in dataset '{dataset_id}'.")
    except NotFound:
        # Define table
        table = bigquery.Table(table_ref, schema=schema)
        table.description = "Table to store specific 8-K filings for stock trading insights."

        try:
            table = client.create_table(table)  # Make an API request.
            logging.info(f"Table '{table_id}' created successfully in dataset '{dataset_id}'.")
        except Conflict:
            logging.error(f"Table '{table_id}' already exists in dataset '{dataset_id}'.")
        except Exception as e:
            logging.error(f"Failed to create table '{table_id}': {e}")
            return

    # Optional: Verify the table schema
    try:
        table = client.get_table(table_ref)
        logging.info(f"Verified schema for table '{table_id}':")
        for field in table.schema:
            logging.info(f" - {field.name} ({field.field_type}) - {field.description}")
    except Exception as e:
        logging.error(f"Failed to retrieve table schema for verification: {e}")

if __name__ == "__main__":
    create_bigquery_table()
