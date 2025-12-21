"""
CIFT Markets - Ensemble Meta-Model

Combines predictions from all 5 specialized models:
1. Hawkes Process: Tick-level order flow dynamics
2. Transformer: Multi-timeframe pattern recognition
3. HMM: Market regime detection
4. GNN: Cross-asset correlation
5. XGBoost: Alternative data fusion

Ensemble Strategy:
- Regime-aware dynamic weighting
- Confidence-based model selection
- Agreement thresholds for trade signals
- Continuous weight adaptation

Key Features:
- Minimum agreement requirement (3/5 models)
- Regime-specific model weights
- Calibrated ensemble probabilities
- Explainable decisions with model contributions

References:
- Caruana et al. (2004): "Ensemble Selection"
- Dietterich (2000): "Ensemble Methods in Machine Learning"
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger

from cift.ml.gnn import CrossAssetGNN, CrossAssetPrediction

# Import model types for type hints
from cift.ml.hawkes import HawkesEvent, HawkesOrderFlowModel, HawkesPrediction
from cift.ml.hmm import MarketRegime, MarketRegimeHMM, RegimePrediction
from cift.ml.transformer import OrderFlowTransformer, TransformerPrediction
from cift.ml.xgboost_fusion import XGBoostFusion, XGBoostPrediction

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EnsemblePrediction:
    """Final ensemble prediction output."""
    timestamp: float

    # Primary signal
    direction: str                   # "long", "short", "neutral"
    direction_probability: float     # 0-1 probability of direction
    magnitude: float                 # Expected return in bps

    # Confidence metrics
    confidence: float                # Overall confidence 0-1
    model_agreement: int             # Number of models agreeing
    min_agreement: int               # Required for trade

    # Regime context
    current_regime: MarketRegime
    regime_probability: float

    # Individual model predictions
    hawkes_contribution: float
    transformer_contribution: float
    hmm_contribution: float
    gnn_contribution: float
    xgboost_contribution: float

    # Model weights used
    model_weights: dict[str, float]

    # Trade recommendation
    should_trade: bool
    position_size: float             # 0-1 fraction of max

    # Risk metrics
    stop_loss_bps: float
    take_profit_bps: float
    risk_reward_ratio: float

    # Latency
    inference_latency_ms: float


@dataclass
class ModelPredictions:
    """Container for all individual model predictions."""
    hawkes: HawkesPrediction | None = None
    transformer: TransformerPrediction | None = None
    hmm: RegimePrediction | None = None
    gnn: CrossAssetPrediction | None = None
    xgboost: XGBoostPrediction | None = None


# ============================================================================
# REGIME-AWARE WEIGHTING
# ============================================================================

class RegimeWeightMatrix:
    """
    Dynamic model weights based on market regime.

    Different models excel in different market conditions:
    - Hawkes: Best in high-activity trending markets
    - Transformer: Best with clear patterns
    - HMM: Always useful for regime context
    - GNN: Best when correlations are stable
    - XGBoost: Best with strong alt-data signals
    """

    def __init__(self):
        # Base weights per regime [hawkes, transformer, hmm, gnn, xgboost]
        # Weights sum to 1.0 per regime
        self.regime_weights = {
            MarketRegime.LOW_VOLATILITY: np.array([0.15, 0.25, 0.10, 0.25, 0.25]),
            MarketRegime.TRENDING_UP: np.array([0.30, 0.25, 0.10, 0.15, 0.20]),
            MarketRegime.TRENDING_DOWN: np.array([0.30, 0.25, 0.10, 0.15, 0.20]),
            MarketRegime.HIGH_VOLATILITY: np.array([0.35, 0.20, 0.15, 0.10, 0.20]),
            MarketRegime.CRISIS: np.array([0.40, 0.15, 0.20, 0.05, 0.20]),
        }

        # Recent model performance tracking
        self.model_performance = np.ones(5) * 0.5  # Running accuracy
        self.performance_decay = 0.99

    def get_weights(
        self,
        regime: MarketRegime,
        model_confidences: np.ndarray,
    ) -> np.ndarray:
        """
        Get model weights adjusted for regime and confidence.

        Args:
            regime: Current market regime
            model_confidences: Confidence from each model [5]

        Returns:
            Adjusted weights [5]
        """
        # Start with regime-based weights
        base_weights = self.regime_weights[regime].copy()

        # Adjust by model confidence
        confidence_factor = np.clip(model_confidences, 0.1, 1.0)
        adjusted = base_weights * confidence_factor

        # Adjust by recent performance
        performance_factor = np.clip(self.model_performance, 0.3, 1.0)
        adjusted = adjusted * performance_factor

        # Normalize
        adjusted = adjusted / adjusted.sum()

        return adjusted

    def update_performance(self, model_idx: int, correct: bool):
        """Update model performance tracking."""
        target = 1.0 if correct else 0.0
        self.model_performance[model_idx] = (
            self.performance_decay * self.model_performance[model_idx] +
            (1 - self.performance_decay) * target
        )


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleMetaModel:
    """
    Meta-model combining all specialized models.

    Features:
    - Regime-aware dynamic weighting
    - Confidence-based model selection
    - Agreement thresholds
    - Calibrated ensemble output
    - Performance-adaptive weights
    """

    def __init__(
        self,
        hawkes_model: HawkesOrderFlowModel,
        transformer_model: OrderFlowTransformer,
        hmm_model: MarketRegimeHMM,
        gnn_model: CrossAssetGNN,
        xgboost_model: XGBoostFusion,
        min_agreement: int = 3,
        confidence_threshold: float = 0.65,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.hawkes = hawkes_model
        self.transformer = transformer_model
        self.hmm = hmm_model
        self.gnn = gnn_model
        self.xgboost = xgboost_model

        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Move PyTorch models to device
        self.hawkes.to(device)
        self.transformer.to(device)
        self.hmm.to(device)
        self.gnn.to(device)

        # Regime-aware weighting
        self.weight_matrix = RegimeWeightMatrix()

        # Model names for logging
        self.model_names = ["hawkes", "transformer", "hmm", "gnn", "xgboost"]

        # Ensemble calibration
        self._calibration_a = 1.0
        self._calibration_b = 0.0

        logger.info(f"EnsembleMetaModel initialized (min_agreement={min_agreement})")

    def predict(
        self,
        # Data for different models
        hawkes_events: np.ndarray | None = None,
        transformer_features: Any | None = None,
        hmm_features: np.ndarray | None = None,
        gnn_node_features: np.ndarray | None = None,
        gnn_edge_index: np.ndarray | None = None,
        gnn_symbol_map: dict[int, str] | None = None,
        xgboost_features: np.ndarray | None = None,
        target_symbol: str | None = None,
        timestamp: float = 0.0,
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction from all models.

        Each model receives its specific input format.
        """
        start_time = time.time()

        predictions = ModelPredictions()
        model_directions = []      # Direction signals (-1, 0, 1)
        model_confidences = []     # Confidence values
        model_available = []       # Which models have valid predictions

        # ============ HMM (first for regime context) ============
        current_regime = MarketRegime.LOW_VOLATILITY
        regime_probability = 0.5

        if hmm_features is not None:
            try:
                predictions.hmm = self.hmm.predict(hmm_features, timestamp)
                current_regime = predictions.hmm.current_regime
                regime_probability = predictions.hmm.regime_probability
            except Exception as e:
                logger.warning(f"HMM prediction failed: {e}")

        # ============ Hawkes ============
        if hawkes_events is not None:
            try:
                # Treat `hawkes_events` as the full recent window. Rebuild history each call
                # to avoid double-counting if upstream re-sends the same window.
                self.hawkes.clear_history()
                he = np.asarray(hawkes_events)
                if he.ndim == 2 and he.shape[1] >= 3:
                    for row in he[-(self.hawkes.max_history or len(he)) :]:
                        self.hawkes.add_event(
                            HawkesEvent(
                                timestamp=float(row[0]),
                                event_type=int(row[1]),
                                size=float(row[2]),
                            )
                        )

                hawkes_features = self._derive_hawkes_features(transformer_features)
                predictions.hawkes = self.hawkes.predict(hawkes_features, float(timestamp))

                # Convert to direction
                direction_signal = predictions.hawkes.buy_intensity - predictions.hawkes.sell_intensity
                direction = 1 if direction_signal > 0 else -1 if direction_signal < 0 else 0

                model_directions.append(direction * abs(direction_signal))
                model_confidences.append(predictions.hawkes.confidence)
                model_available.append("hawkes")
            except Exception as e:
                logger.warning(f"Hawkes prediction failed: {e}")
                model_directions.append(0)
                model_confidences.append(0)
        else:
            model_directions.append(0)
            model_confidences.append(0)

        # ============ Transformer ============
        if transformer_features is not None:
            try:
                tick_f, sec_f, min_f = self._coerce_transformer_inputs(transformer_features)
                predictions.transformer = self.transformer.predict(
                    tick_f,
                    sec_f,
                    min_f,
                    timestamp,
                )

                # Transformer outputs P(up). Convert to signed directional signal.
                up_prob = float(predictions.transformer.direction_prob)
                direction = 1 if up_prob > 0.5 else -1 if up_prob < 0.5 else 0
                model_directions.append(direction * abs(up_prob - 0.5) * 2)
                model_confidences.append(predictions.transformer.confidence)
                model_available.append("transformer")
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
                model_directions.append(0)
                model_confidences.append(0)
        else:
            model_directions.append(0)
            model_confidences.append(0)

        # ============ HMM direction contribution ============
        # HMM provides regime context, derive directional signal from regime
        if predictions.hmm is not None:
            if predictions.hmm.trend_regime == "up":
                hmm_direction = 0.5
            elif predictions.hmm.trend_regime == "down":
                hmm_direction = -0.5
            else:
                hmm_direction = 0

            model_directions.append(hmm_direction)
            model_confidences.append(regime_probability)
            model_available.append("hmm")
        else:
            model_directions.append(0)
            model_confidences.append(0)

        # ============ GNN ============
        if gnn_node_features is not None and gnn_edge_index is not None:
            try:
                predictions.gnn = self.gnn.predict(
                    gnn_node_features, gnn_edge_index,
                    gnn_symbol_map or {}, None, timestamp
                )

                # Get direction for target symbol
                if target_symbol and target_symbol in predictions.gnn.asset_predictions:
                    pred = predictions.gnn.asset_predictions[target_symbol]
                    direction = (pred["direction"] - 0.5) * 2  # Convert 0-1 to -1 to 1
                    model_directions.append(direction)
                    model_confidences.append(pred["confidence"])
                    model_available.append("gnn")
                else:
                    model_directions.append(0)
                    model_confidences.append(0)
            except Exception as e:
                logger.warning(f"GNN prediction failed: {e}")
                model_directions.append(0)
                model_confidences.append(0)
        else:
            model_directions.append(0)
            model_confidences.append(0)

        # ============ XGBoost ============
        if xgboost_features is not None:
            try:
                predictions.xgboost = self.xgboost.predict(xgboost_features, timestamp)

                # Use 500ms prediction as primary
                direction = (predictions.xgboost.direction_500ms - 0.5) * 2
                model_directions.append(direction)
                model_confidences.append(predictions.xgboost.confidence_500ms)
                model_available.append("xgboost")
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
                model_directions.append(0)
                model_confidences.append(0)
        else:
            model_directions.append(0)
            model_confidences.append(0)

        # Convert to arrays
        directions = np.array(model_directions)
        confidences = np.array(model_confidences)

        # ============ Get regime-aware weights ============
        weights = self.weight_matrix.get_weights(current_regime, confidences)

        # ============ Compute weighted ensemble ============
        # Weighted average direction
        weighted_direction = np.sum(directions * weights)

        # Compute agreement
        direction_signs = np.sign(directions)
        dominant_sign = np.sign(weighted_direction) if abs(weighted_direction) > 0.01 else 0
        agreement_count = np.sum(
            (direction_signs == dominant_sign) & (confidences > 0.3)
        )

        # ============ Determine final direction ============
        if abs(weighted_direction) < 0.1:
            final_direction = "neutral"
            direction_probability = 0.5
        elif weighted_direction > 0:
            final_direction = "long"
            direction_probability = 0.5 + min(0.5, weighted_direction / 2)
        else:
            final_direction = "short"
            direction_probability = 0.5 - min(0.5, abs(weighted_direction) / 2)

        # Calibrate probability
        direction_probability = self._calibrate_probability(direction_probability)

        # ============ Compute confidence ============
        weighted_confidence = np.sum(confidences * weights)

        # Adjust confidence by agreement
        agreement_factor = agreement_count / 5
        final_confidence = weighted_confidence * (0.5 + 0.5 * agreement_factor)

        # ============ Should trade decision ============
        should_trade = (
            agreement_count >= self.min_agreement and
            final_confidence >= self.confidence_threshold and
            final_direction != "neutral"
        )

        # ============ Position sizing ============
        if should_trade:
            # Scale by confidence and regime
            regime_scale = predictions.hmm.suggested_position_scale if predictions.hmm else 1.0
            position_size = min(1.0, final_confidence * regime_scale)
        else:
            position_size = 0.0

        # ============ Risk parameters ============
        # Base on volatility regime
        vol_multiplier = {
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.CRISIS: 3.0,
        }.get(current_regime, 1.5)

        base_stop = 10  # 10 bps
        stop_loss_bps = base_stop * vol_multiplier
        take_profit_bps = stop_loss_bps * 1.5  # 1.5:1 risk/reward

        # Compute magnitude
        magnitude = 0.0
        if predictions.transformer:
            magnitude += predictions.transformer.magnitude * weights[1]
        if predictions.xgboost:
            magnitude += predictions.xgboost.magnitude_500ms * weights[4] * 100  # to bps

        # ============ Build result ============
        inference_time = (time.time() - start_time) * 1000

        return EnsemblePrediction(
            timestamp=timestamp,
            direction=final_direction,
            direction_probability=direction_probability,
            magnitude=magnitude,
            confidence=final_confidence,
            model_agreement=int(agreement_count),
            min_agreement=self.min_agreement,
            current_regime=current_regime,
            regime_probability=regime_probability,
            hawkes_contribution=float(directions[0] * weights[0]),
            transformer_contribution=float(directions[1] * weights[1]),
            hmm_contribution=float(directions[2] * weights[2]),
            gnn_contribution=float(directions[3] * weights[3]),
            xgboost_contribution=float(directions[4] * weights[4]),
            model_weights={
                "hawkes": float(weights[0]),
                "transformer": float(weights[1]),
                "hmm": float(weights[2]),
                "gnn": float(weights[3]),
                "xgboost": float(weights[4]),
            },
            should_trade=should_trade,
            position_size=position_size,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            risk_reward_ratio=take_profit_bps / stop_loss_bps if stop_loss_bps > 0 else 0,
            inference_latency_ms=inference_time,
        )

    def _calibrate_probability(self, prob: float) -> float:
        """Apply calibration to probability."""
        # Platt scaling: P_calibrated = 1 / (1 + exp(a * prob + b))
        # But we use it to adjust confidence, not invert
        logit = np.log(prob / (1 - prob + 1e-10) + 1e-10)
        calibrated_logit = self._calibration_a * logit + self._calibration_b
        return 1 / (1 + np.exp(-calibrated_logit))

    def _coerce_transformer_inputs(
        self,
        transformer_features: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coerce various upstream formats into (tick, second, minute) arrays.

        Upstream currently passes a dict like:
          {"tick": [seq, d], "second": [seq, d], "minute": [seq, d]}

        The transformer expects the *same* feature_dim across timeframes.
        We pad/truncate to `self.transformer.feature_dim`.
        """
        if isinstance(transformer_features, dict):
            tick_raw = transformer_features.get("tick")
            sec_raw = transformer_features.get("second")
            minute_raw = transformer_features.get("minute")

            tick = (
                np.asarray(tick_raw, dtype=np.float32)
                if tick_raw is not None
                else np.zeros((50, 1), dtype=np.float32)
            )
            sec = (
                np.asarray(sec_raw, dtype=np.float32)
                if sec_raw is not None
                else np.zeros((60, 1), dtype=np.float32)
            )
            minute = (
                np.asarray(minute_raw, dtype=np.float32)
                if minute_raw is not None
                else np.zeros((30, 1), dtype=np.float32)
            )
        else:
            tick = np.asarray(transformer_features, dtype=np.float32)
            sec = np.zeros((60, 1), dtype=np.float32)
            minute = np.zeros((30, 1), dtype=np.float32)

        tick = self._pad_or_truncate_2d(tick, self.transformer.feature_dim)
        sec = self._pad_or_truncate_2d(sec, self.transformer.feature_dim)
        minute = self._pad_or_truncate_2d(minute, self.transformer.feature_dim)
        return tick, sec, minute

    def _derive_hawkes_features(self, transformer_features: Any) -> np.ndarray:
        """Build a feature vector for the Hawkes model.

        Hawkes expects a 1D vector of length `self.hawkes.feature_dim`.
        We attempt to take the latest tick feature row; otherwise zeros.
        """
        feature_dim = int(getattr(self.hawkes, "feature_dim", 20))
        tick = None
        if isinstance(transformer_features, dict) and transformer_features.get("tick") is not None:
            tick = np.asarray(transformer_features.get("tick"), dtype=np.float32)
        elif transformer_features is not None and not isinstance(transformer_features, dict):
            tick = np.asarray(transformer_features, dtype=np.float32)

        if tick is None or tick.size == 0:
            return np.zeros(feature_dim, dtype=np.float32)

        if tick.ndim == 1:
            row = tick
        else:
            row = tick[-1]

        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.shape[0] >= feature_dim:
            return row[:feature_dim]
        out = np.zeros(feature_dim, dtype=np.float32)
        out[: row.shape[0]] = row
        return out

    @staticmethod
    def _pad_or_truncate_2d(arr: np.ndarray, feature_dim: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={arr.shape}")
        if arr.shape[1] == feature_dim:
            return arr
        if arr.shape[1] > feature_dim:
            return arr[:, :feature_dim]
        pad = np.zeros((arr.shape[0], feature_dim - arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=1)

    def calibrate(self, probs: np.ndarray, labels: np.ndarray):
        """
        Calibrate ensemble probabilities using Platt scaling.

        Args:
            probs: Ensemble probability outputs [n_samples]
            labels: True binary labels [n_samples]
        """
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            a, b = params
            logits = np.log(probs / (1 - probs + 1e-10) + 1e-10)
            calibrated = 1 / (1 + np.exp(-(a * logits + b)))

            # Binary cross-entropy
            eps = 1e-10
            loss = -np.mean(
                labels * np.log(calibrated + eps) +
                (1 - labels) * np.log(1 - calibrated + eps)
            )
            return loss

        result = minimize(neg_log_likelihood, [1.0, 0.0], method='Nelder-Mead')
        self._calibration_a, self._calibration_b = result.x

        logger.info(f"Ensemble calibrated: a={self._calibration_a:.4f}, b={self._calibration_b:.4f}")

    def update_performance(self, prediction: EnsemblePrediction, actual_direction: int):
        """
        Update model performance tracking after observing actual outcome.

        Args:
            prediction: The prediction that was made
            actual_direction: Actual direction (1 for up, -1 for down)
        """
        # Determine which models were correct

        # Update individual model performance based on their contributions
        contributions = [
            prediction.hawkes_contribution,
            prediction.transformer_contribution,
            prediction.hmm_contribution,
            prediction.gnn_contribution,
            prediction.xgboost_contribution,
        ]

        for i, contrib in enumerate(contributions):
            if abs(contrib) > 0.01:  # Model contributed
                model_correct = (np.sign(contrib) == actual_direction)
                self.weight_matrix.update_performance(i, model_correct)


# ============================================================================
# ENSEMBLE BUILDER
# ============================================================================

def build_ensemble(
    config: dict[str, Any] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> EnsembleMetaModel:
    """
    Factory function to build complete ensemble from configuration.

    Args:
        config: Configuration dictionary
        device: PyTorch device

    Returns:
        Configured EnsembleMetaModel
    """
    config = config or {}

    # Build individual models with defaults
    hawkes = HawkesOrderFlowModel(
        num_event_types=config.get("hawkes_event_types", 3),
        hidden_dim=config.get("hawkes_hidden_dim", 64),
    )

    transformer = OrderFlowTransformer(
        tick_features=config.get("transformer_tick_features", 32),
        second_features=config.get("transformer_second_features", 16),
        minute_features=config.get("transformer_minute_features", 8),
        hidden_dim=config.get("transformer_hidden_dim", 128),
    )

    hmm = MarketRegimeHMM(
        num_states=config.get("hmm_num_states", 5),
        observation_dim=config.get("hmm_observation_dim", 16),
    )

    gnn = CrossAssetGNN(
        node_features=config.get("gnn_node_features", 8),
        hidden_dim=config.get("gnn_hidden_dim", 64),
    )

    xgboost = XGBoostFusion(
        n_features=config.get("xgboost_n_features", 27),
    )

    ensemble = EnsembleMetaModel(
        hawkes_model=hawkes,
        transformer_model=transformer,
        hmm_model=hmm,
        gnn_model=gnn,
        xgboost_model=xgboost,
        min_agreement=config.get("min_agreement", 3),
        confidence_threshold=config.get("confidence_threshold", 0.65),
        device=device,
    )

    logger.info("Ensemble built successfully")

    return ensemble


__all__ = [
    "EnsembleMetaModel",
    "EnsemblePrediction",
    "ModelPredictions",
    "RegimeWeightMatrix",
    "build_ensemble",
]
