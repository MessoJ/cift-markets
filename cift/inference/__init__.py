"""
CIFT Markets - Inference Module

Real-time inference pipeline for ML-based order flow prediction.

Components:
- InferencePipeline: Main pipeline connecting data → features → models → signals
- FeatureExtractor: Real-time feature extraction from market data
- InferenceEngine: Batched GPU inference for efficiency
- WebSocketBroadcaster: Real-time prediction broadcasting

Usage:
    from cift.inference import InferencePipeline, PipelineConfig
    
    config = PipelineConfig(
        polygon_api_key="your-key",
        symbols=["SPY", "QQQ"],
    )
    
    pipeline = InferencePipeline(config)
    
    @pipeline.on_prediction
    def handle_prediction(result):
        if result.prediction.should_trade:
            print(f"{result.symbol}: {result.prediction.direction}")
    
    await pipeline.start()
"""

from cift.inference.pipeline import (
    InferencePipeline,
    PipelineConfig,
    InferenceResult,
    InferenceEngine,
    FeatureExtractor,
    WebSocketBroadcaster,
)

__all__ = [
    "InferencePipeline",
    "PipelineConfig",
    "InferenceResult",
    "InferenceEngine",
    "FeatureExtractor",
    "WebSocketBroadcaster",
]
