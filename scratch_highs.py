import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

SP100_SYMBOLS = ["AAPL", "MSFT", "AMZN"]
lookback_days = 90

end_date = datetime.today()
start_date = end_date - timedelta(days=lookback_days + 252 + 50)

data = yf.download(SP100_SYMBOLS, start=start_date, end=end_date, progress=False)
high_df = data["High"].ffill()

print(f"Total rows fetched: {len(high_df)}")

rolling_high = high_df.rolling(window=252).max()
print(f"Non-NaN rolling_high rows: {rolling_high.notna().sum().iloc[0]}")

new_highs = (high_df >= rolling_high).sum(axis=1)

print("Recent new highs:")
print(new_highs.iloc[-lookback_days:].value_counts())
