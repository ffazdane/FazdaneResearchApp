import yfinance as yf
import pandas as pd

df = yf.download('AMD', period='1y')
if isinstance(df.columns, pd.MultiIndex):
    df = df.xs('AMD', level=1, axis=1)

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
current_price = valid_close.iloc[-1]
curr_span_a = senkou_span_a.iloc[-1]
curr_span_b = senkou_span_b.iloc[-1]
print(f"Price: {current_price}, A: {curr_span_a}, B: {curr_span_b}")
if not (pd.isna(curr_span_a) or pd.isna(curr_span_b)):
    max_cloud = max(curr_span_a, curr_span_b)
    print(f"Max cloud: {max_cloud}")
    if current_price <= max_cloud:
        print("Below or inside")
    else:
        print("Above")
