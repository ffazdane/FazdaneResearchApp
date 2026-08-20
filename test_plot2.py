import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

full_data = yf.download(['AMD', 'AAPL', 'NVDA'], period='1y')
is_multi = isinstance(full_data.columns, pd.MultiIndex)
ichimoku_results = {}
for ticker in ['AMD', 'AAPL', 'NVDA']:
    if is_multi:
        df = full_data.xs(ticker, level=1, axis=1)
    else:
        df = full_data
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
    if not (pd.isna(curr_span_a) or pd.isna(curr_span_b)):
        max_cloud = max(curr_span_a, curr_span_b)
        if current_price <= max_cloud:
            ichimoku_results[ticker] = 'below'

df_plot = pd.DataFrame(np.random.randn(5, 3), columns=['AMD ☎️', 'AAPL', 'NVDA'])
fig, ax = plt.subplots()
sns.heatmap(df_plot, ax=ax)

for tick_label in ax.get_xticklabels():
    label_text = tick_label.get_text()
    raw_ticker = label_text.replace(" ☎️", "").strip()
    
    has_earnings = '☎' in label_text
    is_below_cloud = ichimoku_results.get(raw_ticker) == 'below'

    if has_earnings:
        tick_label.set_color("red")
    elif is_below_cloud:
        tick_label.set_color("orange")

fig.savefig("test_plot2.png")
print("ichimoku_results:", ichimoku_results)
