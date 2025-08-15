import yfinance as yf


def get_stock_price(ticker):
    # Download data for the specified ticker
    stock = yf.Ticker(ticker)
    # Fetch the most recent closing price
    stock_info = stock.history(period="1d")

    # Check if stock_info has data to avoid errors
    if not stock_info.empty:
        latest_price = stock_info['Close'].iloc[-1]
        return latest_price
    else:
        return "No data available for this ticker."


# Example usage
ticker = "WEYS"  # Replace with the ticker symbol you're interested in
price = get_stock_price(ticker)
print(f"The latest closing price for {ticker} is: ${price:.2f}")
