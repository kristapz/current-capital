import os
import logging
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions
import argparse

# ================================
# Configuration and Setup
# ================================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to your service account key file
# Update this path to point to your actual key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'  # Update as needed

# Define your project ID, dataset ID, and table ID
project_id = 'test1-427219'      # Replace with your GCP project ID
dataset_id = 'consciousness'     # Replace with your BigQuery dataset ID
table_id = 'papers'              # Replace with your BigQuery table ID

# Full table identifier in the format `project.dataset.table`
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# Initialize BigQuery client
client_bq = bigquery.Client(project=project_id)

# ================================
# Define Delete Function
# ================================

def delete_paper(row_number):
    """
    Deletes the paper with the specified row_number from the BigQuery table.

    Args:
        row_number (int): The unique row_number of the paper to delete.
    """
    try:
        # Step 1: Check if the row exists
        query_check = f"""
            SELECT COUNT(*) as count
            FROM `{full_table_id}`
            WHERE row_number = @row_number
        """
        job_config_check = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("row_number", "INT64", row_number)
            ]
        )
        logging.info(f"Checking existence of row_number = {row_number}...")
        query_job_check = client_bq.query(query_check, job_config=job_config_check)
        result_check = query_job_check.result()
        count = next(result_check).count

        if count == 0:
            logging.warning(f"No paper found with row_number = {row_number}.")
            return

        # Step 2: Prompt for confirmation
        confirm = input(f"Are you sure you want to delete the paper with row_number = {row_number}? (y/n): ")
        if confirm.lower() != 'y':
            logging.info("Deletion canceled by the user.")
            return

        # Step 3: Perform the deletion
        query_delete = f"""
            DELETE FROM `{full_table_id}`
            WHERE row_number = @row_number
        """
        job_config_delete = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("row_number", "INT64", row_number)
            ]
        )
        logging.info(f"Deleting row_number = {row_number}...")
        query_job_delete = client_bq.query(query_delete, job_config=job_config_delete)
        query_job_delete.result()  # Wait for the deletion to complete

        logging.info(f"Successfully deleted paper with row_number = {row_number}.")

    except google_exceptions.GoogleCloudError as gce:
        logging.error(f"Google Cloud Error: {gce}")
    except Exception as e:
        logging.error(f"Error deleting paper with row_number {row_number}: {e}")

# ================================
# Main Entry Point
# ================================

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Delete a paper from BigQuery table by row_number.")
    parser.add_argument('row_number', type=int, help='The row_number of the paper to delete.')

    # Parse the arguments
    args = parser.parse_args()
    row_number = args.row_number

    # Call the delete function
    delete_paper(row_number)

if __name__ == "__main__":
    main()
