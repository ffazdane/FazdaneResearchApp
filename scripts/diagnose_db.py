"""
Quick diagnostic: check raw_analysis_json status in trade_recommendation DB.
Run with: .\.venv\Scripts\python.exe scripts\diagnose_db.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.trade_recommendation.database import get_connection, deserialize_analysis_data
import json

conn = get_connection()
cursor = conn.cursor()

print("\n=== Latest snapshot per ticker ===")
cursor.execute("""
    SELECT ticker, snapshot_datetime,
           CASE WHEN raw_analysis_json IS NULL THEN 'NULL'
                WHEN length(raw_analysis_json) = 0 THEN 'EMPTY'
                ELSE 'HAS_DATA (' || length(raw_analysis_json) || ' bytes)'
           END as json_status,
           trade_decision, trade_score
    FROM ticker_signal_snapshot
    WHERE snapshot_id IN (
        SELECT MAX(snapshot_id) FROM ticker_signal_snapshot GROUP BY ticker
    )
    ORDER BY snapshot_datetime DESC
    LIMIT 20
""")
rows = cursor.fetchall()
for row in rows:
    print(f"  {row[0]:8s} | {row[1]} | {row[2]:50s} | {row[3]} | {row[4]}")

print("\n=== Deserialization test (first 3 with data) ===")
cursor.execute("""
    SELECT ticker, raw_analysis_json
    FROM ticker_signal_snapshot
    WHERE raw_analysis_json IS NOT NULL AND length(raw_analysis_json) > 10
    ORDER BY snapshot_datetime DESC
    LIMIT 3
""")
for row in cursor.fetchall():
    ticker = row[0]
    json_str = row[1]
    try:
        data = deserialize_analysis_data(json_str)
        keys = list(data.keys())[:8]
        print(f"  ✅ {ticker}: deserialized OK. Keys: {keys}")
    except Exception as e:
        print(f"  ❌ {ticker}: DESERIALIZATION FAILED: {e}")
        # Show first 200 chars of JSON
        print(f"      JSON preview: {json_str[:200]}")

conn.close()
print("\nDone.")
