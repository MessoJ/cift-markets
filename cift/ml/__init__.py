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

from __future__ import annotations

__all__: list[str] = []


# Optional imports: keep `import cift.ml` working even when heavy deps aren't installed.
try:
    from cift.ml.hawkes import HawkesEvent, HawkesOrderFlowModel, HawkesPrediction, HawkesTrainer

    __all__ += [
        "HawkesOrderFlowModel",
        "HawkesEvent",
        "HawkesPrediction",
        "HawkesTrainer",
    ]
except ImportError:
    pass

try:
    from cift.ml.hmm import (
        HMMTrainer,
        MarketRegime,
        MarketRegimeHMM,
        RegimeFeatures,
        RegimePrediction,
    )

    __all__ += [
        "MarketRegimeHMM",
        "MarketRegime",
        "RegimePrediction",
        "RegimeFeatures",
        "HMMTrainer",
    ]
except ImportError:
    pass

try:
    from cift.ml.xgboost_fusion import (
        AlternativeDataFeatures,
        XGBoostFusion,
        XGBoostPrediction,
        XGBoostTrainer,
    )

    __all__ += [
        "XGBoostFusion",
        "XGBoostPrediction",
        "AlternativeDataFeatures",
        "XGBoostTrainer",
    ]
except ImportError:
    pass

try:
    from cift.ml.ensemble import (
        EnsembleMetaModel,
        EnsemblePrediction,
        ModelPredictions,
        RegimeWeightMatrix,
        build_ensemble,
    )

    __all__ += [
        "EnsembleMetaModel",
        "EnsemblePrediction",
        "ModelPredictions",
        "RegimeWeightMatrix",
        "build_ensemble",
    ]
except ImportError:
    pass

try:
    from cift.ml.gnn import AssetNode, CrossAssetGNN, CrossAssetPrediction, GNNTrainer

    __all__ += [
        "CrossAssetGNN",
        "CrossAssetPrediction",
        "AssetNode",
        "GNNTrainer",
    ]
except ImportError:
    pass

try:
    from cift.ml.transformer import OrderFlowTransformer, TransformerPrediction, TransformerTrainer

    __all__ += [
        "OrderFlowTransformer",
        "TransformerPrediction",
        "TransformerTrainer",
    ]
except ImportError:
    pass
