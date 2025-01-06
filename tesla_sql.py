import sqlite3
import pandas as pd

conn = sqlite3.connect('tesla_data.db')
cursor = conn.cursor()

file_path = "tesla.15.24.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])

df.to_sql('Stock', conn, if_exists='replace', index=False)

query = """
SELECT Date, Close
FROM Tesla
ORDER BY Close DESC
LIMIT 1;
"""
result = cursor.execute(query).fetchall()
print("Day with Highest Closing Price:", result)

# ----------------------------- Queries -----------------------------

query = """
SELECT AVG(Volume) AS Avg_Daily_Volume
FROM Tesla;
"""
result = cursor.execute(query).fetchall()
print("Average Daily Volume:", result[0][0])

query = """
SELECT Date, Open, Close, ((Close - Open) / Open) * 100 AS Price_Change_Percent
FROM Tesla
WHERE ((Close - Open) / Open) * 100 > 5;
"""
result = cursor.execute(query).fetchall()
print("Days with >5% Price Increase:", result)

query = """
SELECT Date, Volume
FROM Tesla
ORDER BY Volume DESC
LIMIT 1;
"""
result = cursor.execute(query).fetchall()
print("Busiest Trading Day:", result)

query = """
CREATE TABLE IF NOT EXISTS Insights AS
SELECT Date, Close, Open, High, Low, Volume,
       ((Close - Open) / Open) * 100 AS Price_Change_Percent
FROM Tesla
WHERE ((Close - Open) / Open) * 100 > 5;
"""
cursor.execute(query)
conn.commit()
print("Insights table created.")


# ----------------------------- Format Results -----------------------------
insights_df = pd.read_sql_query("SELECT * FROM Insights;", conn)
print(insights_df.head())

insights_df.to_csv("tesla_insights.csv", index=False)