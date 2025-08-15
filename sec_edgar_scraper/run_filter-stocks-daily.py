import subprocess
import time
from datetime import datetime

def run_stock_processor(source_table_id: str):
    """Runs the stock_processor.py script with the given source_table_id."""
    try:
        print(f"Starting processing for {source_table_id} at {datetime.now()}")
        subprocess.run(["python", "filter-stocks-daily.py", source_table_id], check=True)
        print(f"Completed processing for {source_table_id} at {datetime.now()}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing {source_table_id}: {e}")

def main():
    tables_to_process = ['stocksbio', 'stocksai']
    for idx, table in enumerate(tables_to_process):
        run_stock_processor(table)
        if idx < len(tables_to_process) - 1:
            print(f"Waiting for 3 minutes before processing the next table...")
            time.sleep(180)  # 3 minutes in seconds

    print("All tables have been processed successfully.")

if __name__ == "__main__":
    main()
