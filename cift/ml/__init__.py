"""
CIFT Markets - Machine Learning Module

Production-grade ML models for order flow prediction.

Models:
- Hawkes Process: Tick-level order flow dynamics (cift.ml.hawkes)
- Transformer: Multi-timeframe pattern recognition (cift.ml.transformer)
- HMM: Market regime detection (cift.ml.hmm)
- GNN: Cross-asset correlation modeling (cift.ml.gnn)
- XGBoost: Alternative data fusion (cift.ml.xgboost_fusion)

Ensemble: Regime-aware weighted combination of all models (cift.ml.ensemble)
"""

# Hawkes Process Model
from cift.ml.hawkes import (
    HawkesOrderFlowModel,
    HawkesEvent,
    HawkesPrediction,
    HawkesTrainer,
)

# Transformer Model
from cift.ml.transformer import (
    OrderFlowTransformer,
    TransformerPrediction,
    TransformerTrainer,
)

# Hidden Markov Model
from cift.ml.hmm import (
    MarketRegimeHMM,
    MarketRegime,
    RegimePrediction,
    RegimeFeatures,
    HMMTrainer,
)

# Graph Neural Network
from cift.ml.gnn import (
    CrossAssetGNN,
    CrossAssetPrediction,
    AssetNode,
    GNNTrainer,
)

# XGBoost Alternative Data Fusion
from cift.ml.xgboost_fusion import (
    XGBoostFusion,
    XGBoostPrediction,
    AlternativeDataFeatures,
    XGBoostTrainer,
)

# Ensemble Meta-Model
from cift.ml.ensemble import (
    EnsembleMetaModel,
    EnsemblePrediction,
    ModelPredictions,
    RegimeWeightMatrix,
    build_ensemble,
)

__all__ = [
    # Hawkes
    "HawkesOrderFlowModel",
    "HawkesEvent",
    "HawkesPrediction",
    "HawkesTrainer",
    # Transformer
    "OrderFlowTransformer",
    "TransformerPrediction",
    "TransformerTrainer",
    # HMM
    "MarketRegimeHMM",
    "MarketRegime",
    "RegimePrediction",
    "RegimeFeatures",
    "HMMTrainer",
    # GNN
    "CrossAssetGNN",
    "CrossAssetPrediction",
    "AssetNode",
    "GNNTrainer",
    # XGBoost
    "XGBoostFusion",
    "XGBoostPrediction",
    "AlternativeDataFeatures",
    "XGBoostTrainer",
    # Ensemble
    "EnsembleMetaModel",
    "EnsemblePrediction",
    "ModelPredictions",
    "RegimeWeightMatrix",
    "build_ensemble",
]
