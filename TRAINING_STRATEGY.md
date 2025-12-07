# CIFT Markets - ML Training & Deployment Strategy

## 1. Overview
This document outlines the end-to-end strategy for training, validating, and deploying the advanced machine learning models (Hawkes, Transformer, GNN, HMM, XGBoost) integrated into the CIFT platform.

## 2. Where to Train? (Infrastructure)

### Option A: Local Training (Development)
- **Hardware**: NVIDIA GPU (RTX 3080/4090 recommended) with >12GB VRAM.
- **Environment**: Docker container with GPU support (`nvidia-docker`).
- **Pros**: Free, immediate feedback.
- **Cons**: Slower for large datasets (TB scale).

### Option B: Cloud Training (Production)
- **Provider**: AWS (EC2 p3.2xlarge) or Lambda Labs (A100).
- **Setup**:
    1.  Spin up GPU instance.
    2.  Pull `cift-markets` repo.
    3.  Run `docker-compose -f docker-compose.train.yml up`.
- **Pros**: Massive scale, faster convergence.
- **Cons**: Cost (~$3-10/hour).

**Recommendation**: Start with **Local Training** on a subset of data to debug the pipeline, then move to **Cloud** for the full historical backtest.

---

## 3. With What? (Data Pipeline)

The models require high-frequency Order Book (L2/L3) and Trade data.

### Data Sources
1.  **QuestDB**: Stores raw tick data (Trades, Quotes).
2.  **ClickHouse**: Stores aggregated features (OFI, Volume Profiles) for fast retrieval.

### Feature Engineering Pipeline
Before training, raw data must be processed into tensors:
1.  **Raw Ingestion**: Ticks $\rightarrow$ OHLCV + Order Flow Imbalance (OFI).
2.  **Normalization**: Z-score normalization (rolling window).
3.  **Tensor Construction**:
    -   **Hawkes**: Sequence of timestamps $(t_1, t_2, ...)$.
    -   **Transformer**: $(Batch, SeqLen, Features)$.
    -   **GNN**: Adjacency Matrix $A$ + Node Features $X$.

---

## 4. How to Train? (The Pipeline)

We have created a master training script: `scripts/train_models.py`.

### Step-by-Step Process
1.  **Data Loading**:
    ```bash
    python scripts/train_models.py --mode data_prep --start 2023-01-01 --end 2023-12-31
    ```
    *Fetches data from QuestDB, computes OFI, saves to `.parquet` files.*

2.  **Model Training**:
    ```bash
    python scripts/train_models.py --mode train --model all --epochs 50
    ```
    *Sequentially trains:*
    -   **HMM**: Learns market regimes (Volatile, Trending, Mean-Reverting).
    -   **Hawkes**: Fits power-law kernels to order arrival times.
    -   **GNN**: Learns asset correlations (Lead-Lag).
    -   **Transformer**: Trains on residuals from Hawkes/Linear models.
    -   **XGBoost**: Trains the meta-learner on outputs of above models.

3.  **Evaluation**:
    ```bash
    python scripts/train_models.py --mode eval --split test
    ```
    *Generates `metrics.json` with Sharpe, Accuracy, and Calibration plots.*

---

## 5. How to Use? (Inference & Deployment)

Once trained, models are saved to `models/checkpoints/`.

### Real-Time Inference Flow
1.  **Market Data Stream**: WebSocket receives a new tick.
2.  **Feature Update**: `FeatureStore` updates rolling windows in memory (Redis/Dragonfly).
3.  **Model Prediction**:
    -   **Hawkes**: Updates intensity $\lambda(t)$.
    -   **Transformer**: Predicts next 1s price movement.
    -   **Ensemble**: Aggregates predictions weighted by regime (HMM).
4.  **Signal Generation**: If `Signal > Threshold`, generate Buy/Sell order.

### Integration
The `cift.inference.service` loads these models on startup:
```python
# cift/inference/service.py
class InferenceEngine:
    def load_models(self):
        self.hawkes = HawkesOrderFlowModel.load("models/checkpoints/hawkes_v1.pt")
        self.transformer = OrderFlowTransformer.load("models/checkpoints/transformer_v1.pt")
        # ...
```

---

## 6. Action Items
1.  Run `python scripts/seed_market_data.py` to generate synthetic training data (if no real data).
2.  Run `python scripts/train_models.py` to produce initial model artifacts.
3.  Start the platform: `docker-compose up -d`.
