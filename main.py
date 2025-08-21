import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Load environment variables
load_dotenv()

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pattern Detection -----
def detect_patterns(df):
    patterns = []
    for i in range(1, len(df)):
        o, h, l, c = df.iloc[i][['Open', 'High', 'Low', 'Close']]
        prev_o, prev_c = df.iloc[i-1][['Open', 'Close']]
        body = abs(c - o)

        if c > o and prev_c < prev_o and c > prev_o and o < prev_c:
            pattern = "Bullish Engulfing"
        elif o > c and prev_c > prev_o and c < prev_o and o > prev_c:
            pattern = "Bearish Engulfing"
        elif c > o and (o - l) >= 2 * body and (h - c) <= body:
            pattern = "Hammer"
        elif o > c and (h - o) >= 2 * body and (c - l) <= body:
            pattern = "Shooting Star"
        elif body <= 0.005 * c:
            pattern = "Doji"
        else:
            pattern = None
        patterns.append(pattern)

    patterns.insert(0, None)
    df['Pattern'] = patterns
    return df

# ----- Technical Prediction -----
def predict_next_day(df):
    temp_df = df.copy()
    temp_df['SMA5'] = temp_df['Close'].rolling(5).mean()
    temp_df['SMA20'] = temp_df['Close'].rolling(20).mean()
    temp_df['Return'] = temp_df['Close'].pct_change()
    temp_df['Volatility'] = temp_df['Return'].rolling(5).std()
    temp_df.dropna(inplace=True)

    X = temp_df[['Return', 'Volatility']].copy()
    X['SMA_ratio'] = temp_df['SMA5'] / temp_df['SMA20']
    X = X[['Return', 'Volatility', 'SMA_ratio']]
    y = (temp_df['Close'].shift(-1) > temp_df['Close']).astype(int).dropna()
    X = X.iloc[:len(y)]

    if len(X) < 10:
        return {"prediction": "UP", "confidence": 50.0}

    model = LogisticRegression()
    model.fit(X, y)
    last_features = X.iloc[[-1]]
    pred = model.predict(last_features)[0]
    prob = model.predict_proba(last_features)[0][pred]

    return {"prediction": "UP" if pred == 1 else "DOWN", "confidence": round(float(prob) * 100, 2)}

# ----- Sentiment Analysis -----
def get_stock_sentiment(symbol):
    try:
        sentiments = []
        news = []

        # Yahoo Finance news
        stock = yf.Ticker(symbol)
        if hasattr(stock, 'news') and stock.news:
            news = stock.news
        # Fallback to NewsAPI
        if not news:
            API_KEY = os.getenv("NEWSAPI_KEY")
            if API_KEY:
                url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
                r = requests.get(url)
                data = r.json()
                news = [{"title": a["title"], "link": a["url"]} for a in data.get("articles", [])]

        news = news[:10]
        for article in news:
            title = article.get('title', 'No title')
            score = analyzer.polarity_scores(title)
            sentiments.append({
                "title": title,
                "url": article.get('link', '#'),
                "score": score['compound']
            })

        if not sentiments:
            return {"average_score": 0, "sentiment": "Neutral", "news": []}

        avg_score = np.mean([s['score'] for s in sentiments])
        sentiment_label = "Bullish" if avg_score > 0.01 else "Bearish" if avg_score < 0.01 else "Neutral"

        return {
            "average_score": avg_score,
            "sentiment": sentiment_label,
            "news": sentiments
        }

    except Exception as e:
        return {"average_score": 0, "sentiment": "Neutral", "news": []}

# ----- Joji Indicator -----
def calculate_joji_indicator(df):
    sma5 = df['Close'].rolling(5).mean()
    sma20 = df['Close'].rolling(20).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        joji = ((sma5 - sma20) / sma20) * 100

    latest_joji = joji.iloc[-1] if not joji.empty else None
    if pd.isna(latest_joji):
        return None

    if latest_joji > 1:
        signal = "Bullish Momentum"
    elif latest_joji < -1:
        signal = "Bearish Momentum"
    else:
        signal = "Neutral"

    return {"value": round(float(latest_joji), 2), "signal": signal}

# ----- Main Stock Endpoint -----
@app.get("/stock/{symbol}")
def get_stock_data(symbol: str, timeframe: str = Query("2m")):
    tf_map = {"2m": "2mo", "4m": "4mo", "6m": "6mo", "1y": "1y"}
    period = tf_map.get(timeframe, "2mo")

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval="1d")
        df.reset_index(inplace=True)

        if df.empty:
            return {"error": "No data for symbol"}

        df = detect_patterns(df)
        next_day_pred = predict_next_day(df)
        sentiment = get_stock_sentiment(symbol)
        joji_indicator = calculate_joji_indicator(df)

        highs = df['High'].tail(30).values
        lows = df['Low'].tail(30).values
        support = np.min(lows) if len(lows) else None
        resistance = np.max(highs) if len(highs) else None

        ohlc = df[['Date', 'Open', 'High', 'Low', 'Close', 'Pattern', 'Volume']].copy()
        ohlc['Date'] = ohlc['Date'].astype(str)

        raw_patterns = ohlc[ohlc['Pattern'].notnull()].tail(50)
        detected_patterns = []
        for _, row in raw_patterns.iterrows():
            detected_patterns.append({
                "Date": str(row["Date"]),
                "Pattern": row["Pattern"],
                "Price": float(row["High"])
            })

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlc": ohlc.to_dict(orient="records"),
            "detected_patterns": detected_patterns,
            "next_day_prediction": next_day_pred,
            "sentiment": sentiment,
            "support": round(float(support), 2) if support else None,
            "resistance": round(float(resistance), 2) if resistance else None,
            "joji_indicator": joji_indicator
        }

    except Exception as e:
        return {"error": str(e)}

# ----- Patterns Endpoint -----
@app.get("/patterns/{symbol}")
def get_patterns(
    symbol: str,
    pattern: str = Query(..., description="Pattern name to filter, e.g. 'Bullish Engulfing'"),
    timeframe: str = Query("2m", description="Timeframe: '2m', '4m', '6m', '1y'")
):
    tf_map = {"2m": "2mo", "4m": "4mo", "6m": "6mo", "1y": "1y"}
    period = tf_map.get(timeframe, "2mo")

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval="1d")
        df.reset_index(inplace=True)

        if df.empty:
            return {"error": "No data for symbol"}

        df = detect_patterns(df)
        filtered = df[df['Pattern'] == pattern]

        patterns_list = []
        for _, row in filtered.iterrows():
            patterns_list.append({
                "Date": str(row["Date"]),
                "Pattern": row["Pattern"],
                "Price": float(row["High"])
            })

        return {
            "symbol": symbol,
            "pattern": pattern,
            "timeframe": timeframe,
            "patterns": patterns_list
        }
    except Exception as e:
        return {"error": str(e)}
