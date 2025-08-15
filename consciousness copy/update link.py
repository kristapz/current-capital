import os
import logging
from google.cloud import bigquery
from google.api_core import exceptions as google_exceptions

# ================================
# Configuration and Setup
# ================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testkey10k/test1-key.json'  # Update the path accordingly

# Define your project ID, dataset ID, and table ID
PROJECT_ID = 'test1-427219'
DATASET_ID = 'consciousness'
TABLE_ID = 'papers'

# Full table identifier
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize BigQuery client
client_bq = bigquery.Client(project=PROJECT_ID)

# Define allowed fields
ALLOWED_FIELDS = [
    "Philosophy of Mind",
    "Cognitive Neuroscience",
    "Psychology and Experimental Research",
    "Artificial Intelligence and Computational Models",
    "Psychedelics and Altered States of Consciousness",
    "Quantum Theories of Consciousness",
    "Cultural, Spiritual, and Anthropological Perspectives"
]

# ================================
# Function Definitions
# ================================

def list_rows():
    """
    Retrieves and displays rows with row_number, processed_at, and a snippet of the article.
    """
    query = f"""
        SELECT row_number, processed_at, SUBSTR(article, 1, 50) AS article_snippet
        FROM `{FULL_TABLE_ID}`
        ORDER BY row_number ASC
    """
    try:
        query_job = client_bq.query(query)
        results = query_job.result()
        rows = list(results)

        if not rows:
            logging.info("No rows found in the table.")
            return []

        print("\nAvailable Rows (row_number, processed_at, article snippet):")
        print("===========================================================")
        for row in rows:
            snippet = row.article_snippet + '...' if len(row.article_snippet) == 50 else row.article_snippet
            print(f"row_number={row.row_number}, processed_at={row.processed_at}, article=\"{snippet}\"")
        print("===========================================================\n")

        return [row.row_number for row in rows]

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Failed to retrieve rows: {e}")
        return []

def update_field(row_number, new_field):
    """
    Updates the 'field' column for the specified row_number.
    """
    query = f"""
        UPDATE `{FULL_TABLE_ID}`
        SET field = @new_field
        WHERE row_number = @row_number
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("new_field", "STRING", new_field),
            bigquery.ScalarQueryParameter("row_number", "INT64", row_number),
        ]
    )

    try:
        query_job = client_bq.query(query, job_config=job_config)
        query_job.result()  # Wait for the job to complete
        logging.info(f"Successfully updated 'field' for row_number {row_number} to '{new_field}'.")
    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Failed to update 'field' for row_number {row_number}: {e}")

def get_valid_row_number(existing_row_numbers):
    """
    Prompts the user to enter a valid row_number.
    """
    while True:
        try:
            user_input = input("Enter the row_number you wish to update (or 'cancel' to return to menu): ").strip()
            if user_input.lower() == 'cancel':
                return None
            row_number = int(user_input)
            if row_number in existing_row_numbers:
                return row_number
            else:
                print(f"row_number {row_number} does not exist. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a numeric row_number or 'cancel'.")

def get_valid_field():
    """
    Prompts the user to enter a valid field from the allowed categories.
    """
    print("\nAllowed field categories:")
    for idx, field in enumerate(ALLOWED_FIELDS, start=1):
        print(f"{idx}. {field}")
    print()

    while True:
        user_input = input("Enter the number corresponding to the new field (or 'cancel' to return to menu): ").strip()
        if user_input.lower() == 'cancel':
            return None
        if user_input.isdigit():
            selection = int(user_input)
            if 1 <= selection <= len(ALLOWED_FIELDS):
                return ALLOWED_FIELDS[selection - 1]
        print("Invalid selection. Please enter a valid number or 'cancel'.")

def manual_update():
    """
    Handles the manual update process.
    """
    existing_row_numbers = list_rows()
    if not existing_row_numbers:
        return

    row_number = get_valid_row_number(existing_row_numbers)
    if row_number is None:
        print("Update canceled. Returning to main menu.\n")
        return

    new_field = get_valid_field()
    if new_field is None:
        print("Update canceled. Returning to main menu.\n")
        return

    confirm = input(f"Are you sure you want to update row_number {row_number} to '{new_field}'? (yes/no): ").strip().lower()
    if confirm == 'yes':
        update_field(row_number, new_field)
    else:
        print("Update canceled. Returning to main menu.\n")

def display_menu():
    """
    Displays the interactive menu.
    """
    print("========== Manual Field Updater ==========")
    print("1. List available rows")
    print("2. Update the 'field' column for a row")
    print("3. Exit")
    print("==========================================")

def main():
    """
    Main function to run the interactive CLI.
    """
    print("Welcome to the Manual Field Updater for BigQuery.\n")
    while True:
        display_menu()
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            list_rows()
        elif choice == '2':
            manual_update()
        elif choice == '3':
            print("Exiting the script. Goodbye!")
            break
        else:
            print("Invalid choice. Please select an option from 1 to 3.\n")

if __name__ == "__main__":
    main()
