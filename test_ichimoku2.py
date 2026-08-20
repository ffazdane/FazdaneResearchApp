import yfinance as yf
import pandas as pd
import numpy as np

def test():
    full_data = yf.download(['AMD', 'AAPL'], period='1y')
    ichimoku_results = {}
    is_multi = isinstance(full_data.columns, pd.MultiIndex)

    for ticker in ['AMD', 'AAPL']:
        try:
            if is_multi:
                df = full_data.xs(ticker, level=1, axis=1)
            else:
                df = full_data  # Only one ticker, columns are Open, High, Low, Close
                
            # Ichimoku Cloud Check
            try:
                high_9 = df['High'].rolling(window=9).max()
                low_9 = df['Low'].rolling(window=9).min()
                tenkan_sen = (high_9 + low_9) / 2

                high_26 = df['High'].rolling(window=26).max()
                low_26 = df['Low'].rolling(window=26).min()
                kijun_sen = (high_26 + low_26) / 2

                senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

                high_52 = df['High'].rolling(window=52).max()
                low_52 = df['Low'].rolling(window=52).min()
                senkou_span_b = ((high_52 + low_52) / 2).shift(26)

                valid_close = df['Close'].dropna()
                if len(valid_close) > 0:
                    current_price = valid_close.iloc[-1]
                    curr_span_a = senkou_span_a.iloc[-1]
                    curr_span_b = senkou_span_b.iloc[-1]

                    # Check if price is below ANY of the cloud (meaning below the highest part of the cloud)
                    if not (pd.isna(curr_span_a) or pd.isna(curr_span_b)):
                        max_cloud = max(curr_span_a, curr_span_b)
                        if current_price <= max_cloud:
                            ichimoku_results[ticker] = 'below'
            except Exception as e_ich:
                print(f"Could not calc ichimoku for {ticker}: {e_ich}")
                
        except Exception as e:
            print(f"Error calculating FDTS/Ichimoku for {ticker}: {e}")

    print("Ichimoku results:", ichimoku_results)

test()
