"""
CIFT Markets - Order Flow Transformer Inference Service

Loads the trained TPU model and provides real-time predictions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
import os

@dataclass  
class OrderFlowPrediction:
    """Prediction from the Order Flow Transformer."""
    timestamp: float
    direction: str  # "up", "down", "neutral"
    direction_probs: Dict[str, float]  # {"down": 0.3, "neutral": 0.4, "up": 0.3}
    confidence: float  # max probability
    signal_strength: float  # edge over neutral
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "direction": self.direction,
            "direction_probs": self.direction_probs,
            "confidence": self.confidence,
            "signal_strength": self.signal_strength,
        }


class OrderFlowTransformer(nn.Module):
    """
    Order Flow Transformer Model.
    
    Trained on BTCUSDT data to predict short-term price direction.
    Architecture matches the TPU-trained v8 model.
    """
    def __init__(
        self,
        n_features: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last token
        return self.fc_out(x)


class OrderFlowPredictor:
    """
    Inference service for the Order Flow Transformer.
    
    Loads the trained model and provides real-time predictions
    based on recent market data.
    """
    
    DIRECTION_MAP = {0: "down", 1: "neutral", 2: "up"}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model: Optional[OrderFlowTransformer] = None
        self.feature_buffer: list = []  # Rolling buffer of features
        
        # Feature normalization stats (approximate from training)
        self.feature_means = np.array([0.0, 0.0, 0.0, 0.0])
        self.feature_stds = np.array([1.0, 1.0, 1.0, 1.0])
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model from checkpoint."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Create model
            self.model = OrderFlowTransformer(
                n_features=4,
                d_model=64,
                n_heads=4,
                n_layers=2,
                seq_len=64,
                num_classes=3,
                dropout=0.1,
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Loaded model with {num_params:,} parameters from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def compute_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute input features from raw market data.
        
        Args:
            prices: Array of close prices (at least 64 + 20 for rolling features)
            volumes: Array of volumes
            
        Returns:
            Feature array of shape (seq_len, 4)
        """
        if len(prices) < 84:  # 64 seq + 20 for rolling window
            raise ValueError(f"Need at least 84 data points, got {len(prices)}")
        
        # Compute returns
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        
        # Compute log volume diff
        log_vol = np.diff(np.log1p(volumes))
        
        # Rolling features
        n = len(returns)
        volatility = np.zeros(n)
        momentum = np.zeros(n)
        
        for i in range(20, n):
            volatility[i] = np.std(returns[i-20:i])
        for i in range(10, n):
            momentum[i] = np.mean(returns[i-10:i])
        
        # Align all arrays to the same length (from index 20 onwards)
        start_idx = 20
        aligned_returns = returns[start_idx:]
        aligned_log_vol = log_vol[start_idx:]
        aligned_volatility = volatility[start_idx:]
        aligned_momentum = momentum[start_idx:]
        
        # Stack features
        features = np.stack([
            aligned_returns,
            aligned_log_vol,
            aligned_volatility,
            aligned_momentum,
        ], axis=1).astype(np.float32)
        
        # Clip extreme values
        features = np.clip(features, -10, 10)
        
        # Normalize
        features = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Return last seq_len features
        return features[-64:]
    
    def predict(
        self,
        features: np.ndarray,
        timestamp: float = 0.0,
    ) -> OrderFlowPrediction:
        """
        Make a prediction from feature array.
        
        Args:
            features: Array of shape (64, 4) with normalized features
            timestamp: Current timestamp
            
        Returns:
            OrderFlowPrediction with direction and confidence
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if features.shape != (64, 4):
            raise ValueError(f"Expected features shape (64, 4), got {features.shape}")
        
        # Convert to tensor
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Parse prediction
        direction_idx = int(np.argmax(probs))
        direction = self.DIRECTION_MAP[direction_idx]
        confidence = float(probs[direction_idx])
        
        # Signal strength = how much better than neutral
        signal_strength = float(probs[direction_idx] - probs[1]) if direction_idx != 1 else 0.0
        
        return OrderFlowPrediction(
            timestamp=timestamp,
            direction=direction,
            direction_probs={
                "down": float(probs[0]),
                "neutral": float(probs[1]),
                "up": float(probs[2]),
            },
            confidence=confidence,
            signal_strength=signal_strength,
        )
    
    def predict_from_ohlcv(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        timestamp: float = 0.0,
    ) -> OrderFlowPrediction:
        """
        Make prediction from raw OHLCV data.
        
        Args:
            prices: Array of at least 84 close prices
            volumes: Array of at least 84 volumes
            timestamp: Current timestamp
            
        Returns:
            OrderFlowPrediction
        """
        features = self.compute_features(prices, volumes)
        return self.predict(features, timestamp)


# Singleton instance for global access
_predictor: Optional[OrderFlowPredictor] = None


def get_predictor() -> OrderFlowPredictor:
    """Get the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = OrderFlowPredictor()
    return _predictor


def initialize_predictor(model_path: str, device: str = "cpu") -> bool:
    """Initialize the global predictor with a model."""
    global _predictor
    _predictor = OrderFlowPredictor(model_path=model_path, device=device)
    return _predictor.model is not None


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Order Flow Transformer Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test", action="store_true", help="Run with random test data")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    predictor = OrderFlowPredictor(model_path=args.model)
    
    if args.test:
        # Generate random test data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        volumes = np.random.exponential(1000, 100)
        
        pred = predictor.predict_from_ohlcv(prices, volumes, timestamp=1234567890.0)
        
        print("\n=== Prediction ===")
        print(f"Direction: {pred.direction}")
        print(f"Probabilities: {pred.direction_probs}")
        print(f"Confidence: {pred.confidence:.2%}")
        print(f"Signal Strength: {pred.signal_strength:.4f}")
