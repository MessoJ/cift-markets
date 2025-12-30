# World-Class Quant Research: The Path to Sharpe 3.0+

## 1. The "Secret Sauce" of Top Firms
Retail traders use **Correlation** and **Z-Scores**.
Top Firms (Citadel, Two Sigma, DE Shaw) use **Copulas**, **Machine Learning**, and **Hierarchical Risk Parity**.

### A. Copula-Based Pairs Trading (Non-Linear Dependence)
*   **The Flaw of Correlation:** Correlation assumes a linear relationship. It fails during market crashes when "everything goes to 1".
*   **The Solution (Copulas):** Copulas model the *joint distribution* of assets. They tell us: "Given Stock A dropped 5%, what is the *conditional probability* that Stock B drops 5%?"
*   **The Edge:** We trade based on **Mispricing Index (MI)**, not Z-Score. MI is the cumulative probability of the current spread. It captures tail dependencies that Z-scores miss.

### B. ML-Enhanced Residuals (Predicting Reversion)
*   **The Flaw of Stat Arb:** We assume mean reversion happens *eventually*. We don't know *when*.
*   **The Solution (XGBoost/LightGBM):** Train a model on the *residuals* (spread).
    *   **Features:** Spread Momentum, Volatility of Spread, Volume Imbalance, Distance from Moving Average.
    *   **Target:** Will the spread revert to mean within N days?
*   **The Edge:** We filter out "falling knives". If the ML model says "Momentum is too strong, spread will widen", we **don't trade**, saving us from massive drawdowns.

### C. Hierarchical Risk Parity (HRP)
*   **The Flaw of Markowitz (Mean-Variance):** It's unstable. It inverts a covariance matrix, which amplifies noise.
*   **The Solution (HRP):** Uses graph theory (clustering) to group similar assets and allocate capital hierarchically.
*   **The Edge:** It's robust to market noise and doesn't require expected return estimates (which are usually wrong).

---

## 2. Crypto Funding Arbitrage Optimization
*   **Basis Trading:** Long Spot + Short Perp.
*   **Optimization:**
    *   **Cross-Exchange Arb:** Long Spot on Binance, Short Perp on Bybit (if Bybit funding > Binance).
    *   **Execution:** Use TWAP/VWAP to enter large positions without slippage.
    *   **Leverage:** 2x-3x is safe for this strategy because it's delta-neutral. This turns a 15% APY into 30-45%.

---

## 3. Intraday Alpha (The Missing Link)
*   **Microstructure Features:**
    *   **Order Flow Imbalance (OFI):** (Bid Vol - Ask Vol) at best limits.
    *   **Trade Flow Imbalance (TFI):** Aggressor volume (market buys vs market sells).
*   **Data Sources:**
    *   **Databento:** High-quality tick data (pay-as-you-go).
    *   **Alpaca:** Free minute bars (good for starting).
    *   **Binance Public Data:** Free tick data for crypto.

---

## 4. Actionable Implementation Plan
We are upgrading our engine from "Genius" to **"Institutional"**.

### New Components:
1.  **`cift/ml/advanced_quant.py`**:
    *   `CopulaPairs`: For non-linear signal generation.
    *   `MLSignalFilter`: XGBoost model to veto bad trades.
    *   `HRPPortfolio`: For robust capital allocation.

2.  **`cift/ml/institutional_production.py`**:
    *   Integrates Copulas + ML + HRP.
    *   Runs on 1-hour or 1-minute data (simulated for now, ready for real feed).

### The Goal
*   **Sharpe 1.37** (Current) -> **Sharpe 2.0+** (With Copulas/ML) -> **Sharpe 3.0+** (With Intraday Data).

This is the bleeding edge. No sugarcoating. This is how the big boys play.
