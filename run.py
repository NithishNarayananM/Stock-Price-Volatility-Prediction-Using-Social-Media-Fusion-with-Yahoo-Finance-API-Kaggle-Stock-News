# run.py — robust download + flattening + merge
import yfinance as yf
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# CONFIG
stock_symbol = "TSLA"
start_date = "2022-01-01"
end_date = "2023-01-01"
news_csv_path = "data/stock_news.csv"   # update if your path differs

# ---------------------------
# 1) Download stock data (explicit auto_adjust to avoid future warning)
data = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False)
# If yfinance returned multi-level columns (happens when multiple tickers were requested), flatten them:
if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
    # flatten tuples like ('TSLA','Close') -> 'TSLA_Close' (or just 'Close' if that's clearer)
    data.columns = ["_".join([str(x) for x in col if x is not None and str(x).strip() != ""]).strip() for col in data.columns.values]

# Ensure Date is a normal column
if 'Date' not in data.columns:
    data = data.reset_index()   # will create a Date column if index was a DatetimeIndex

# if date column exists but with a different name and dtype, try to detect & rename it to 'Date'
if 'Date' not in data.columns:
    # try to find a datetime-like column
    for c in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[c]):
            data = data.rename(columns={c: 'Date'})
            break

# As a last resort, if index is datetime, reset it
if 'Date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
    data = data.reset_index().rename(columns={data.columns[0]: 'Date'})

# Normalize Date column to date (no time) for merging
data['Date'] = pd.to_datetime(data['Date']).dt.date

# 2) Ensure there's a Close column (detect common close-like column names)
close_candidates = [c for c in data.columns if 'close' in str(c).lower()]
if not close_candidates:
    raise RuntimeError(f"Couldn't find any 'Close' column in stock data. Columns: {data.columns.tolist()}")
# pick the first match and copy to standardized 'Close'
data['Close'] = data[close_candidates[0]]

# Compute returns & volatility flag
data['Return'] = data['Close'].pct_change()
data['Volatility'] = np.where(data['Return'].abs() > 0.02, 1, 0)

# ---------------------------
# 3) Load & sanitize news CSV
news_df = pd.read_csv(news_csv_path)

# normalize column names to simple lowercase keys
news_df.columns = [str(c).strip().lower() for c in news_df.columns]

# find 'date' column
if 'date' not in news_df.columns:
    possible_date_cols = [c for c in news_df.columns if 'date' in c]
    if possible_date_cols:
        news_df = news_df.rename(columns={possible_date_cols[0]: 'date'})
    else:
        raise RuntimeError(f"No date column found in news CSV. Columns: {news_df.columns.tolist()}")

# find 'headline' column (or fallbacks like 'title' or 'text')
if 'headline' not in news_df.columns:
    for alt in ['title', 'news', 'text']:
        if alt in news_df.columns:
            news_df = news_df.rename(columns={alt: 'headline'})
            break
if 'headline' not in news_df.columns:
    raise RuntimeError(f"No headline/title/text column found in news CSV. Columns: {news_df.columns.tolist()}")

# Normalize date column to date (no time)
news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce').dt.date
# drop rows where date could not be parsed
news_df = news_df.dropna(subset=['date'])

# ---------------------------
# 4) Clean headlines & sentiment
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'http\S+', '', t)          # remove URLs
    t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t)  # keep alpha-num
    t = re.sub(r'\s+', ' ', t).strip()
    return t

news_df['clean_headline'] = news_df['headline'].apply(clean_text)
news_df['sentiment'] = news_df['clean_headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Aggregate daily sentiment (mean)
daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()

# ---------------------------
# 5) Merge (left: stock data, right: daily_sentiment)
print("DEBUG: stock data columns:", data.columns.tolist())
print("DEBUG: daily_sentiment columns:", daily_sentiment.columns.tolist())
print(f"DEBUG: stock rows = {len(data)}, sentiment days = {len(daily_sentiment)}")

merged = pd.merge(data, daily_sentiment, left_on='Date', right_on='date', how='inner')
print(f"DEBUG: merged shape = {merged.shape}")

if merged.empty:
    print("WARNING: merged DataFrame is empty — possible reasons:")
    print("- Date formats don't match (stock Date vs news date).")
    print("- No overlapping dates between stock data and news data.")
    # Show a few sample dates from each to help debug
    print("Sample stock dates:", sorted(list(set(data['Date'])))[0:5])
    print("Sample news dates:", sorted(list(set(daily_sentiment['date'])))[0:5])
    raise SystemExit("Merge produced empty dataset. Fix date alignment or check input CSV.")

# keep only needed columns and drop NA rows
merged = merged[['Date', 'Close', 'Return', 'sentiment', 'Volatility']].dropna()

# ---------------------------
# 6) Train/test model (same as before)
X = merged[['Return', 'sentiment']]
y = merged['Volatility']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# 7) Plots (EDA)
plt.figure(figsize=(8,6))
sns.scatterplot(data=merged, x='sentiment', y='Return', hue='Volatility', palette="coolwarm")
plt.title("Scatterplot: Sentiment vs Stock Returns")
plt.show()

# ---------------------------
# 8) More Visualizations

# 1. Time series of Close price with sentiment overlay
plt.figure(figsize=(12,6))
plt.plot(merged['Date'], merged['Close'], label='Stock Close Price', color='blue')
plt.scatter(merged['Date'], merged['sentiment']*merged['Close'].max()/2 + merged['Close'].mean(), 
            c=merged['sentiment'], cmap='coolwarm', alpha=0.6, label='Sentiment (scaled)')
plt.title(f"{stock_symbol} Stock Price with Sentiment Overlay")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# 2. Boxplot of returns grouped by volatility
plt.figure(figsize=(8,6))
sns.boxplot(data=merged, x='Volatility', y='Return', palette='Set2')
plt.title("Boxplot of Returns by Volatility Class")
plt.xlabel("Volatility (0 = Low, 1 = High)")
plt.ylabel("Daily Return")
plt.show()

# 3. KDE (density) plot of sentiment for low vs high volatility days
plt.figure(figsize=(8,6))
sns.kdeplot(data=merged[merged['Volatility']==0], x='sentiment', label='Low Volatility', fill=True)
sns.kdeplot(data=merged[merged['Volatility']==1], x='sentiment', label='High Volatility', fill=True)
plt.title("Sentiment Distribution by Volatility")
plt.legend()
plt.show()

# 4. Pairplot to visualize relationships
sns.pairplot(merged[['Return','sentiment','Volatility']], hue='Volatility', palette='husl')
plt.suptitle("Pairplot of Features vs Volatility", y=1.02)
plt.show()

# 5. Rolling average of sentiment vs returns
merged_sorted = merged.sort_values('Date')
plt.figure(figsize=(12,6))
plt.plot(merged_sorted['Date'], merged_sorted['Return'].rolling(7).mean(), label='7-Day Avg Return', color='green')
plt.plot(merged_sorted['Date'], merged_sorted['sentiment'].rolling(7).mean(), label='7-Day Avg Sentiment', color='red')
plt.title("Rolling Averages: Returns vs Sentiment")
plt.legend()
plt.show()

# 6. Volatility counts barplot
plt.figure(figsize=(6,5))
sns.countplot(x='Volatility', data=merged, palette='pastel')
plt.title("Class Balance: Low vs High Volatility Days")
plt.show()


plt.figure(figsize=(8,6))
sns.histplot(merged['sentiment'], bins=20, kde=True)
plt.title("Histogram of News Sentiment")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(merged['Return'].dropna(), bins=20, kde=True)
plt.title("Histogram of Daily Stock Returns")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(merged[['Return','sentiment','Volatility']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
