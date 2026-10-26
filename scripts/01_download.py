from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

tickers = pd.read_csv(ROOT / "tickers.csv")["ticker"].tolist()
print(f"Downloading {len(tickers)} stocks...")

prices = yf.download(tickers, start="2015-01-01", end="2025-12-31",
                     auto_adjust=True, threads=False)["Close"]

missing_share = prices.isna().mean()
keep = missing_share[missing_share <= 0.01].index
dropped = sorted(set(tickers) - set(keep))
prices = prices[keep].ffill().dropna()

print(f"Kept {prices.shape[1]} stocks over {prices.shape[0]} days")
print("Dropped:", dropped)

prices.to_csv(DATA / "prices.csv")
print("Saved to", DATA / "prices.csv")