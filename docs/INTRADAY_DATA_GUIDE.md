# How to Get Intraday Data (The Fuel for Sharpe 3.0)

To achieve Sharpe 3.0+, our "Institutional Engine" needs **1-minute bar data**.
Daily data is too slow for modern statistical arbitrage.

## 1. Free / Testing Data (Start Here)
We can get limited intraday data for free using `yfinance`.
*   **Limit:** Last 7 days (1-min bars) or 2 years (1-hour bars).
*   **Use Case:** Testing code, verifying logic, short-term paper trading.

**Action:** Run the script I created:
```powershell
python scripts/download_intraday.py
```

## 2. Professional Data (Required for Real Trading)
For a serious backtest (e.g., 2020-2024 on 1-min bars), you need a paid provider.

### Option A: Alpaca (Best for Individuals)
*   **Cost:** Free (IEX data) or $99/mo (SIP data - all exchanges).
*   **History:** ~2016 to present.
*   **API:** Very easy Python SDK.
*   **Why:** It's the standard for retail quants.

### Option B: Databento (Best Quality)
*   **Cost:** Pay-as-you-go (approx $20-50 for a full dataset).
*   **Quality:** Institutional grade (nanosecond precision).
*   **Why:** If you want to simulate "perfect" execution.

### Option C: Binance (Crypto Only)
*   **Cost:** Free.
*   **History:** Full history available.
*   **Why:** Essential for the "Funding Arb" component.

## 3. How to Integrate
Once you have the data (e.g., from Alpaca), save it as Parquet files in `data/intraday/1m/`.
The `InstitutionalEngine` is already designed to load high-frequency data if available.

**Recommendation:**
1.  Run `scripts/download_intraday.py` now to test the pipeline.
2.  If results look promising on the 7-day sample, sign up for Alpaca (Free Tier) to get more history.
