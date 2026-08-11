import sqlite3
try:
    conn = sqlite3.connect('data/master_ticker_analysis.sqlite')
    conn.execute('ALTER TABLE master_analysis ADD COLUMN next_earnings_date TEXT;')
    conn.commit()
    print('Success')
except Exception as e:
    print(e)
