import yfinance as yf
import sys


def get_stock_price(ticker_symbol):
    """
    Fetches and prints the current price of the given ticker symbol.

    Parameters:
    ticker_symbol (str): The ticker symbol to fetch the price for.

    Returns:
    float: The current price of the ticker, or None if not found.
    """
    try:
        # Create a Ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Fetch the latest market data
        data = ticker.history(period='1d')

        if data.empty:
            print(f"No data found for ticker symbol '{ticker_symbol}'. Please verify the symbol.")
            return None

        # Extract the closing price
        current_price = data['Close'].iloc[-1]
        print(f"The current price of {ticker_symbol} is: ${current_price:.2f}")
        return current_price

    except Exception as e:
        print(f"An error occurred while fetching data for '{ticker_symbol}': {e}")
        return None


if __name__ == "__main__":
    # Default ticker symbol
    default_ticker = "BINK"  # Replace with "kaai" if that's the correct symbol

    # Allow user to input a ticker symbol
    if len(sys.argv) > 1:
        ticker_input = sys.argv[1].upper()
    else:
        ticker_input = default_ticker

    get_stock_price(ticker_input)
