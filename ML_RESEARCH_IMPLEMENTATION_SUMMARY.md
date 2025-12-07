# CIFT Markets - ML Research Implementation Summary

## Overview

All 5 ML models have been upgraded with **research-validated SOTA techniques** based on deep analysis of academic papers. Every enhancement has been verified against original arXiv sources.

---

## 1. Hawkes Process (`cift/ml/hawkes.py`)

### Enhancement: Power-Law Kernel Approximation

**Research Source:** 
- Hardiman et al. (2013) "Critical reflexivity in financial markets" [arXiv:1302.1405](https://arxiv.org/abs/1302.1405)
- "Forecasting High Frequency Order Flow Imbalance" [arXiv:2408.03594](https://arxiv.org/abs/2408.03594)

**Key Finding:** 
Market microstructure exhibits power-law decay with exponent ~**-1.15** (not exponential). Standard exponential kernels miss long-range dependencies.

**Implementation:**
```python
class PowerLawApproximationKernel(nn.Module):
    """
    Approximates power-law kernel using sum of exponentials:
    K(t) ≈ Σᵢ αᵢ · βᵢ · exp(-βᵢt)
    
    Uses geometric β spacing (100, 10, 1, 0.1, 0.01 Hz) to capture
    decay from 10ms to 100s timescales.
    """
```

**Parameters:**
- 5-component mixture with geometric β spacing
- Decay rates: 100, 10, 1, 0.1, 0.01 Hz
- Covers 4 orders of magnitude of temporal scales

---

## 2. Transformer (`cift/ml/transformer.py`)

### Enhancement A: Rotary Position Embeddings (RoPE)

**Research Source:** 
- Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding" [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Key Finding:**
RoPE encodes relative positions through rotation matrices, improving generalization and extrapolation to longer sequences.

**Implementation:**
```python
class RotaryPositionalEmbedding(nn.Module):
    """
    Applies rotation to query/key vectors:
    RoPE(x, m) = R(θ_m) · x
    where R is rotation matrix with angle θ_m = m · θ_base
    """
```

### Enhancement B: Gated Residual Networks (GRN)

**Research Source:**
- Lim et al. (2019) "Temporal Fusion Transformers" [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)

**Key Finding:**
GLU (Gated Linear Units) gating allows adaptive feature suppression, critical for handling heterogeneous financial data.

**Implementation:**
```python
class GatedResidualNetwork(nn.Module):
    """
    GRN with GLU gating:
    1. Linear projection to 2*hidden
    2. Split into (a, b)
    3. Gate: a * sigmoid(b)
    4. Project back + residual
    """
```

### Enhancement C: Variable Selection Network

**Research Source:**
- Lim et al. (2019) [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)

**Key Finding:**
Soft feature selection via learned attention weights improves interpretability and performance.

---

## 3. Hidden Markov Model (`cift/ml/hmm.py`)

### Enhancement: Input-Output HMM (IO-HMM)

**Research Source:**
- Nystrup et al. (2017) "Dynamic Asset Allocation with Hidden Markov Models"

**Key Finding:**
Transition probabilities should depend on macro features (VIX, volume) rather than being static. This allows faster reaction to regime changes.

**Implementation:**
```python
class MarketRegimeHMM(nn.Module):
    """
    IO-HMM: Transition matrix is feature-conditioned:
    P(z_t = j | z_{t-1} = i, x_t) = softmax(W_i @ x_t + b_i)
    
    Uses separate network per source state with persistence bias.
    """
```

**Parameters:**
- `transition_feature_dim=4`: Features conditioning transitions (VIX, Volume, etc.)
- Persistence bias via diagonal-dominant initialization
- Backward-compatible with standard HMM when `use_io_hmm=False`

---

## 4. Graph Neural Network (`cift/ml/gnn.py`)

### Enhancement: Dynamic Graph Learning

**Research Source:**
- Wu et al. (2019) "Graph WaveNet for Deep Spatial-Temporal Graph Modeling" [arXiv:1906.00121](https://arxiv.org/abs/1906.00121)

**Key Finding:**
Financial correlations are non-stationary. Learn adaptive adjacency matrix from node embeddings instead of using static correlations.

**Implementation:**
```python
class DynamicGraphLearning(nn.Module):
    """
    Learns adaptive adjacency:
    A_adaptive = softmax(ReLU(E1 @ E2.T))
    
    Combines with static graph:
    A_final = α*A_static + (1-α)*A_learned
    """
```

**Parameters:**
- `embed_dim=16`: Node embedding dimension
- `static_graph_weight=0.3`: Weight for correlation-based graph
- `top_k=10`: Sparsification (keep top-10 edges per node)
- Optional feature-based edge learning

---

## 5. XGBoost Fusion (`cift/ml/xgboost_fusion.py`)

### Enhancement A: Monotonic Constraints

**Research Source:**
- Economic domain knowledge + XGBoost documentation

**Key Finding:**
Enforce sensible feature-target relationships (e.g., higher sentiment → higher return probability) to improve generalization and interpretability.

**Implementation:**
```python
MONOTONIC_CONSTRAINTS = {
    "put_call_ratio": -1,      # Higher = bearish
    "news_sentiment": 1,       # Positive = bullish
    "vix_level": -1,           # High VIX = bearish
    "order_flow_imbalance": 1, # Buy imbalance = bullish
    # ... 27 features total
}
```

### Enhancement B: Purged K-Fold Cross-Validation

**Research Source:**
- de Prado (2018) "Advances in Financial Machine Learning"

**Key Finding:**
Standard K-Fold causes temporal leakage because training samples near test boundaries contain future information.

**Implementation:**
```python
def _get_purged_cv_splits(self, n_samples, n_splits=5):
    """
    Purges samples within `purge_gap` periods of fold boundaries.
    Prevents look-ahead bias in time series data.
    """
```

**Parameters:**
- `purge_gap=10`: Number of periods purged between train/val
- Configurable via constructor

---

## Summary Table

| Model | Enhancement | Paper | Key Insight |
|-------|-------------|-------|-------------|
| Hawkes | Power-Law Kernel | arXiv:1302.1405 | ~-1.15 decay exponent |
| Transformer | RoPE | arXiv:2104.09864 | Relative position encoding |
| Transformer | GRN + VSN | arXiv:1912.09363 | GLU gating + feature selection |
| HMM | IO-HMM | Nystrup 2017 | Feature-conditioned transitions |
| GNN | Dynamic Graph | arXiv:1906.00121 | Learned adjacency matrix |
| XGBoost | Monotonic | Domain knowledge | Economic consistency |
| XGBoost | Purged CV | de Prado 2018 | No temporal leakage |

---

## Usage Examples

### Power-Law Hawkes
```python
from cift.ml.hawkes import HawkesOrderFlowModel

model = HawkesOrderFlowModel(
    input_dim=8,
    hidden_dim=64,
    kernel_type="power_law",  # Enable research-validated kernel
)
```

### IO-HMM
```python
from cift.ml.hmm import MarketRegimeHMM

model = MarketRegimeHMM(
    num_states=5,
    use_io_hmm=True,
    transition_feature_dim=4,  # VIX, Volume, Spread, Return
)

# Forward pass with transition features
transition_features = torch.tensor([vix, volume, spread, ret]).unsqueeze(0)
log_lik, alpha = model.forward_algorithm(observations, transition_features=transition_features)
```

### Dynamic GNN
```python
from cift.ml.gnn import CrossAssetGNN

model = CrossAssetGNN(
    node_features=8,
    use_dynamic_graph=True,
    num_nodes=50,
    static_graph_weight=0.3,  # 70% learned, 30% static
)

# Graph learner available for direct use
adj_learned = model.graph_learner(static_adj, node_features)
```

### XGBoost with Purged CV
```python
from cift.ml.xgboost_fusion import XGBoostFusion

model = XGBoostFusion(
    use_monotonic=True,
    use_purged_cv=True,
    purge_gap=10,
)

# Train with purged cross-validation
cv_scores = model.train_with_purged_cv(X, y_direction, y_magnitude, n_splits=5)
```

---

## Validation Checksum

All implementations verified against original papers:
- ✅ arXiv:1302.1405 - Power-law exponent validated
- ✅ arXiv:2104.09864 - RoPE formula verified
- ✅ arXiv:1912.09363 - GLU gating structure confirmed
- ✅ arXiv:1906.00121 - Graph WaveNet adjacency learning confirmed
- ✅ arXiv:2408.03594 - Sum of exponentials for Hawkes validated

---

*Generated: Research-validated implementation complete*
