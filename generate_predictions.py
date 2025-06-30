import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- Configuration ---
# List of stocks you want to track
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
# Name of the output file
OUTPUT_FILE = 'predictions.json'

# --- API Setup ---
# Load API keys from environment variables for security.
# This is handled by the GitHub Actions workflow.
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY') # Optional, but recommended
except KeyError:
    print("FATAL: API keys (GEMINI_API_KEY) not found in environment variables.")
    exit(1)

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- Core Functions ---

def get_stock_data_and_news(symbol):
    """Fetches historical stock data and recent news."""
    # Fetch historical price data for the last 3 months
    stock_data = yf.download(symbol, period="3mo", auto_adjust=True)
    if stock_data.empty:
        raise ValueError(f"No historical data found for {symbol}")

    news_headlines = "No recent news found."
    if NEWS_API_KEY:
        try:
            # Fetch the top 10 most recent articles in English for the symbol
            news_url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            response = requests.get(news_url)
            response.raise_for_status() # Raise an exception for bad status codes (like 4xx or 5xx)
            articles = response.json().get('articles', [])
            news_headlines = "\n".join([f"- {a['title']}" for a in articles])
        except requests.RequestException as e:
            # Handle cases where the News API fails
            news_headlines = f"Could not fetch news: {e}"
            
    return stock_data, news_headlines

def get_ai_analysis(symbol, historical_data, news_headlines):
    """Generates a structured qualitative analysis from the Gemini API."""
    # Add a simple moving average to the data to give the AI a technical signal
    historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
    prompt_data = historical_data.tail(30).to_string() # Use last 30 days of data for the prompt

    prompt = f"""
    Analyze the financial data for **{symbol}**.
    Based on the historical price data (including the 20-day Simple Moving Average) and recent news, provide a short-term forecast for the next trading day.
    Your response must be a single, clean JSON object with these exact keys: "sentiment", "reasoning", "predicted_low", "predicted_high".

    - "sentiment": String. Must be "Bullish", "Bearish", or "Neutral".
    - "reasoning": String. A brief, data-driven explanation for your sentiment in 2-3 sentences.
    - "predicted_low": Number. Your estimated lowest price for the next trading day.
    - "predicted_high": Number. Your estimated highest price for the next trading day.

    Do not include any text, markdown formatting like ```json, or explanations outside of the JSON object itself.

    **Historical Data (Price and 20-Day SMA):**
    {prompt_data}

    **Recent News Headlines:**
    {news_headlines}
    """

    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    response.raise_for_status() # This will raise an error if the API call fails
    
    # Extract the text content and clean it up
    text_content = response.json()['candidates'][0]['content']['parts'][0]['text']
    clean_text = text_content.strip().replace('```json', '').replace('```', '')
    
    # Parse the cleaned text into a Python dictionary
    return json.loads(clean_text)

# --- Main Execution Logic ---

def main():
    """Main function to generate and save all predictions."""
    all_predictions = {
        'last_updated': datetime.utcnow().isoformat() + 'Z', # Use UTC for a standard timestamp
        'predictions': []
    }

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            stock_data, news = get_stock_data_and_news(symbol)
            analysis = get_ai_analysis(symbol, stock_data, news)

            # Combine all data into one record for the JSON output
            prediction_record = {
                'symbol': symbol,
                # FIX IS HERE: Convert the pandas/numpy number to a standard Python float before rounding
                'current_price': round(float(stock_data['Close'].iloc[-1]), 2),
                'sentiment': analysis.get('sentiment'),
                'reasoning': analysis.get('reasoning'),
                'predicted_range': [analysis.get('predicted_low'), analysis.get('predicted_high')]
            }
            all_predictions['predictions'].append(prediction_record)
            print(f"Successfully processed {symbol}.")
        
        except Exception as e:
            # If any step fails for a symbol, record the error and continue
            print(f"ERROR processing {symbol}: {e}")
            error_record = {'symbol': symbol, 'error': str(e)}
            all_predictions['predictions'].append(error_record)

    # Write the final dictionary to the JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_predictions, f, indent=4) # Use indent=4 for nice formatting
        
    print(f"\nSuccessfully generated predictions and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
