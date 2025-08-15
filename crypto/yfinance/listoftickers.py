import requests
import pandas as pd
import time
import os
import re
import sys
import csv

from tqdm import tqdm

def get_api_key():
    """
    Retrieves the CoinMarketCap API key from environment variables.
    If not found, prompts the user to input it.

    Returns:
        str: The API key.
    """
    api_key = os.getenv('CMC_API_KEY')
    if not api_key:
        api_key = input("Enter your CoinMarketCap API Key: ").strip()
    return api_key

def is_valid_symbol(symbol):
    """
    Validates if the cryptocurrency symbol is purely alphanumeric.

    Parameters:
        symbol (str): The cryptocurrency symbol to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    return symbol.isalnum()

def safe_get_first(lst):
    """
    Safely retrieves the first element from a list.

    Parameters:
        lst (list or None): The list to retrieve the first element from.

    Returns:
        str or None: The first element if available, else None.
    """
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    return None

def fetch_latest_listings(api_key, start=1, limit=5000, convert='USD'):
    """
    Fetches the latest cryptocurrency listings from CoinMarketCap.

    Parameters:
        api_key (str): Your CoinMarketCap API key.
        start (int): The starting rank.
        limit (int): Number of cryptocurrencies to fetch (max 5000).
        convert (str): The fiat currency to convert prices into.

    Returns:
        pd.DataFrame: DataFrame containing latest listings.
    """
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }
    parameters = {
        'start': start,
        'limit': limit,
        'convert': convert
    }

    try:
        print("Fetching latest cryptocurrency listings...")
        response = requests.get(url, headers=headers, params=parameters)
        response.raise_for_status()
        data = response.json()

        if data['status']['error_code'] != 0:
            raise Exception(f"Error fetching latest listings: {data['status']['error_message']}")

        df_listings = pd.json_normalize(data['data'])
        print(f"Retrieved {len(df_listings)} listings.")
        return df_listings
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching listings: {http_err}")
    except Exception as e:
        print(f"An error occurred while fetching listings: {e}")
    return pd.DataFrame()

def fetch_crypto_info(api_key, symbol):
    """
    Fetches static information for a single cryptocurrency symbol.

    Parameters:
        api_key (str): Your CoinMarketCap API key.
        symbol (str): The cryptocurrency symbol.

    Returns:
        dict or None: Dictionary containing static info of the cryptocurrency or None if failed.
    """
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/info'
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }
    parameters = {
        'symbol': symbol
    }

    try:
        response = requests.get(url, headers=headers, params=parameters)
        response.raise_for_status()
        data = response.json()

        if data['status']['error_code'] != 0:
            print(f"Error fetching info for symbol {symbol}: {data['status']['error_message']}")
            return None

        info = data['data'].get(symbol)
        if not info:
            print(f"No data returned for symbol {symbol}.")
            return None

        record = {
            'Symbol': symbol,
            'Name': info.get('name'),
            'Description': info.get('description'),
            'Category': info.get('category'),
            'Slug': info.get('slug'),
            'Logo': info.get('logo'),
            'Website': safe_get_first(info.get('urls', {}).get('website')),
            'Twitter': safe_get_first(info.get('urls', {}).get('twitter')),
            'Reddit': safe_get_first(info.get('urls', {}).get('reddit')),
            'Message Board': safe_get_first(info.get('urls', {}).get('message_board')),
            'Announcement': safe_get_first(info.get('urls', {}).get('announcement')),
            'Chat URL': safe_get_first(info.get('urls', {}).get('chat_url')),
            'Explorer': safe_get_first(info.get('urls', {}).get('explorer')),
            'Source Code': safe_get_first(info.get('urls', {}).get('source_code')),
            'Twitter Handle': info.get('twitter_handle'),
            'CoinMarketCap URL': f"https://coinmarketcap.com/currencies/{info.get('slug')}/"
        }
        return record
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:
            print("Rate limit exceeded. Please wait for a minute before retrying.")
            sys.exit(1)
        else:
            print(f"HTTP error occurred while fetching info for symbol {symbol}: {http_err}")
    except Exception as e:
        print(f"An error occurred while fetching info for symbol {symbol}: {e}")
    return None

def save_header_to_csv(filename, headers):
    """
    Saves the header to the CSV file.

    Parameters:
        filename (str): The CSV file name.
        headers (list): List of header names.
    """
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
        print(f"CSV header saved to {filename}.")
    except Exception as e:
        print(f"An error occurred while saving header to CSV: {e}")
        sys.exit(1)

def append_row_to_csv(filename, row, headers):
    """
    Appends a single row to the CSV file.

    Parameters:
        filename (str): The CSV file name.
        row (dict): Dictionary containing row data.
        headers (list): List of header names.
    """
    try:
        with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writerow(row)
    except Exception as e:
        print(f"An error occurred while writing to CSV for symbol {row.get('Symbol')}: {e}")
        sys.exit(1)

def merge_listing_info(df_listings, record_info):
    """
    Merges listing data with static info for a single symbol.

    Parameters:
        df_listings (pd.DataFrame): DataFrame containing latest listings.
        record_info (dict): Dictionary containing static info.

    Returns:
        dict: Merged row data.
    """
    symbol = record_info.get('Symbol')
    listing = df_listings[df_listings['symbol'] == symbol]

    if listing.empty:
        print(f"No listing data found for symbol: {symbol}")
        return None

    listing = listing.iloc[0]

    row = {
        'Symbol': symbol,
        'Name_listing': listing.get('name'),
        'Slug_listing': listing.get('slug'),
        'Num_market_pairs': listing.get('num_market_pairs'),
        'Date_added_listing': listing.get('date_added'),
        'Tags_listing': ','.join(listing.get('tags', [])) if isinstance(listing.get('tags'), list) else listing.get('tags'),
        'Max_supply': listing.get('max_supply'),
        'Circulating_supply': listing.get('circulating_supply'),
        'Total_supply': listing.get('total_supply'),
        'Date_added_info': record_info.get('Slug'),  # This might not be the actual date
        'Description': record_info.get('Description'),
        'Category': record_info.get('Category'),
        'Logo': record_info.get('Logo'),
        'Website': record_info.get('Website'),
        'Twitter': record_info.get('Twitter'),
        'Reddit': record_info.get('Reddit'),
        'Message Board': record_info.get('Message Board'),
        'Announcement': record_info.get('Announcement'),
        'Chat URL': record_info.get('Chat URL'),
        'Explorer': record_info.get('Explorer'),
        'Source Code': record_info.get('Source Code'),
        'Twitter Handle': record_info.get('Twitter Handle'),
        'CoinMarketCap URL': record_info.get('CoinMarketCap URL'),
        'Price_listing': listing.get('quote.USD.price'),
        'Volume_24h_listing': listing.get('quote.USD.volume_24h'),
        'Market_cap_listing': listing.get('quote.USD.market_cap'),
        'Percent_change_1h': listing.get('quote.USD.percent_change_1h'),
        'Percent_change_24h': listing.get('quote.USD.percent_change_24h'),
        'Percent_change_7d': listing.get('quote.USD.percent_change_7d')
    }

    # Replace NaN with None
    for key in row:
        if pd.isna(row[key]):
            row[key] = None

    return row

def main():
    # Step 1: Get API Key
    api_key = get_api_key()
    if not api_key:
        print("API Key is required to proceed.")
        sys.exit(1)

    # Step 2: Fetch Latest Listings
    df_listings = fetch_latest_listings(api_key, start=1, limit=5000, convert='USD')
    if df_listings.empty:
        print("No listings retrieved. Exiting.")
        sys.exit(1)

    # Step 3: Extract Symbols and Validate
    symbols = df_listings['symbol'].dropna().unique().tolist()
    print(f"Total unique symbols fetched: {len(symbols)}")

    # Step 4: Initialize CSV
    output_filename = 'all_crypto_data.csv'
    headers = [
        'Symbol', 'Name_listing', 'Slug_listing', 'Num_market_pairs',
        'Date_added_listing', 'Tags_listing', 'Max_supply', 'Circulating_supply',
        'Total_supply', 'Date_added_info', 'Description', 'Category', 'Logo',
        'Website', 'Twitter', 'Reddit', 'Message Board', 'Announcement',
        'Chat URL', 'Explorer', 'Source Code', 'Twitter Handle',
        'CoinMarketCap URL', 'Price_listing', 'Volume_24h_listing',
        'Market_cap_listing', 'Percent_change_1h', 'Percent_change_24h',
        'Percent_change_7d'
    ]

    save_header_to_csv(output_filename, headers)

    # Step 5: Process Each Symbol Individually
    print("Processing each symbol individually...")
    for idx, symbol in enumerate(tqdm(symbols, desc="Processing Symbols", unit="symbol")):
        if not is_valid_symbol(symbol):
            print(f"Skipping invalid symbol: {symbol}")
            continue

        # Fetch static info
        record_info = fetch_crypto_info(api_key, symbol)
        if not record_info:
            print(f"Skipping symbol due to missing info: {symbol}")
            continue

        # Merge listing and info data
        row = merge_listing_info(df_listings, record_info)
        if not row:
            print(f"Skipping symbol due to missing listing data: {symbol}")
            continue

        # Print the row
        print(row)

        # Append to CSV
        append_row_to_csv(output_filename, row, headers)

        # Respect rate limits
        time.sleep(6)  # Adjust based on your rate limit. For Free Tier (10 requests/minute), sleep ~6 seconds.

    print("Data fetching and saving completed successfully.")

if __name__ == "__main__":
    main()
