import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest

conn = sqlite3.connect("data/transactions.db")

query = """
SELECT user_id, SUM(amount) as total, COUNT(*) as freq
FROM transactions
GROUP BY user_id
"""

df = pd.read_sql(query, conn)

# Anomaly detection
model = IsolationForest(contamination=0.05)
df["anomaly"] = model.fit_predict(df[["total", "freq"]])

# Flag fraud
df["fraud"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

print(df.head())
