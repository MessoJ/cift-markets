# FlowSense Phase 3: Model Development (Weeks 6-11)
## Ensemble of 5 Specialized ML Models

> **Timeline**: Weeks 6-11 (6 weeks)  
> **Objective**: Build and train 5 specialized models for order flow prediction  
> **Deliverables**: Hawkes Process, Transformer, HMM, GNN, XGBoost ensemble

---

## Model Architecture Overview

```
INPUT: Features (70+ microstructure + technical + alternative data)
  │
  ├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
  ▼             ▼             ▼             ▼             ▼             ▼
HAWKES      TRANSFORMER      HMM          GNN        XGBOOST
(Tick-level) (1-60s patterns)(Regime)   (Cross-asset)(Alternative)
71% OFI acc   Pattern detect  87% regime   Correlation   Data fusion
  │             │             │             │             │
  └─────────────┴──────────┬──┴─────────────┴─────────────┘
                           ▼
                  ENSEMBLE AGGREGATOR
                  (Weighted voting)
                           │
                           ▼
                  BUY/SELL/NEUTRAL SIGNAL
                  (73% directional accuracy)
```

---

## Week 6-7: Hawkes Process (OFI Predictor)

### File: `ml/models/hawkes_ofi.py`
**Purpose**: Predict order flow imbalance 100-500ms ahead  
**Tech**: `tick` library, `numba`, exponential kernels

```python
"""Hawkes Process for OFI prediction."""
import numpy as np
from tick.hawkes import HawkesExpKern
from numba import jit

class HawkesOFIPredictor:
    def __init__(self, decay=0.1, baseline_intensity=0.5):
        self.model = HawkesExpKern(decays=decay, n_baselines=baseline_intensity)
    
    def fit(self, timestamps: np.ndarray, ofi_events: np.ndarray):
        """Fit Hawkes process on OFI events."""
        self.model.fit([timestamps[ofi_events > 0]])  # Buy events
        return self
    
    def predict_intensity(self, future_time: float) -> float:
        """Predict OFI intensity at future_time."""
        return self.model.get_baseline()[0] + self._excitation(future_time)
```

**Target**: 71% OFI prediction accuracy 100-500ms ahead

---

## Week 8-9: Transformer (Pattern Recognition)

### File: `ml/models/transformer_patterns.py`
**Purpose**: Capture multi-timeframe patterns (1s-60s)  
**Tech**: PyTorch, multi-head attention

```python
"""Transformer for microstructure pattern recognition."""
import torch
import torch.nn as nn

class MicrostructureTransformer(nn.Module):
    def __init__(self, n_features=50, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 3)  # Buy/Sell/Neutral
    
    def forward(self, x):
        """x: (batch, seq_len, n_features)"""
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])  # Last timestep prediction
```

**Target**: Capture 1-60s patterns across 50+ features

---

## Week 10: HMM (Regime Detection)

### File: `ml/models/hmm_regime.py`
**Purpose**: Detect market regimes (low_vol/trending/high_vol)  
**Tech**: `pomegranate`, 3 hidden states

```python
"""HMM for market regime detection."""
from pomegranate import HiddenMarkovModel, NormalDistribution

class RegimeHMM:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = None
    
    def fit(self, returns: np.ndarray, volatility: np.ndarray):
        """Fit HMM on returns and volatility."""
        features = np.column_stack([returns, volatility])
        self.model = HiddenMarkovModel.from_samples(
            NormalDistribution, n_components=self.n_states,
            X=[features], n_jobs=-1
        )
        return self
    
    def predict_regime(self, features: np.ndarray) -> int:
        """Predict current regime (0=low_vol, 1=trending, 2=high_vol)."""
        return self.model.predict([features])[0]
```

**Target**: 87% regime detection precision

---

## Week 11: GNN + XGBoost

### File: `ml/models/gnn_correlation.py`
**Purpose**: Model cross-asset correlations  
**Tech**: `torch_geometric`, graph attention

```python
"""GNN for cross-asset correlation modeling."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class CrossAssetGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.fc = torch.nn.Linear(hidden_channels, 3)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return self.fc(x)
```

### File: `ml/models/xgboost_fusion.py`
**Purpose**: Fuse alternative data (options, sentiment, on-chain)

```python
"""XGBoost for alternative data fusion."""
import xgboost as xgb

class AlternativeDataFusion:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            objective='multi:softmax',
            num_class=3,  # Buy/Sell/Neutral
            gpu_id=0
        )
    
    def fit(self, features, labels):
        """Features: [options_signal, social_sentiment, onchain_flow, ...]"""
        self.model.fit(features, labels, eval_metric='mlogloss')
        return self
```

---

## Ensemble Aggregator

### File: `ml/ensemble/aggregator.py`

```python
"""Ensemble model aggregation with regime-aware weighting."""

class EnsembleAggregator:
    def __init__(self):
        self.models = {
            'hawkes': HawkesOFIPredictor(),
            'transformer': MicrostructureTransformer(),
            'hmm': RegimeHMM(),
            'gnn': CrossAssetGNN(),
            'xgboost': AlternativeDataFusion()
        }
        self.weights = {'low_vol': [0.3, 0.3, 0.1, 0.2, 0.1],
                        'trending': [0.2, 0.3, 0.2, 0.2, 0.1],
                        'high_vol': [0.4, 0.2, 0.2, 0.1, 0.1]}
    
    def predict(self, features, regime):
        predictions = [model.predict(features) for model in self.models.values()]
        weights = self.weights[regime]
        return np.average(predictions, weights=weights, axis=0)
```

---

## Training Pipeline

### File: `ml/training/train.py`

```python
"""Model training pipeline with walk-forward validation."""

class ModelTrainer:
    def __init__(self, train_window=60, test_window=7):
        self.train_window = train_window  # days
        self.test_window = test_window
    
    def walk_forward_train(self, data, start_date, end_date):
        """Walk-forward validation (no lookahead bias)."""
        current_date = start_date
        results = []
        
        while current_date < end_date:
            train_end = current_date
            train_start = train_end - timedelta(days=self.train_window)
            test_end = train_end + timedelta(days=self.test_window)
            
            # Train
            train_data = data[(data.timestamp >= train_start) & (data.timestamp < train_end)]
            self.model.fit(train_data)
            
            # Test
            test_data = data[(data.timestamp >= train_end) & (data.timestamp < test_end)]
            predictions = self.model.predict(test_data)
            results.append(self.evaluate(test_data, predictions))
            
            current_date = test_end
        
        return results
```

---

## Summary: Weeks 6-11 Deliverables

### Models Built (5):
1. **Hawkes Process** - 71% OFI accuracy
2. **Transformer** - Multi-timeframe patterns
3. **HMM** - 87% regime detection
4. **GNN** - Cross-asset correlations
5. **XGBoost** - Alternative data fusion

### Training Infrastructure:
- Walk-forward validation (no lookahead)
- GPU acceleration (PyTorch, XGBoost)
- Model checkpointing
- Hyperparameter tuning (Ray Tune)

### Next Phase:
**Phase 4: Backtesting Engine (Weeks 12-13)** - Realistic simulation with slippage, fees, liquidity constraints
