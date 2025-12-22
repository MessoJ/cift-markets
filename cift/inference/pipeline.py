"""
CIFT Markets - Real-time Inference Pipeline

End-to-end pipeline from market data to trade signals:
1. Data ingestion (Polygon/Databento)
2. Feature extraction
3. Model inference (5 models + ensemble)
4. Signal generation
5. WebSocket broadcast

Architecture:
- Async event loop for concurrent processing
- Ring buffers for rolling windows
- GPU batching for model inference
- NATS JetStream for signal distribution

Performance targets:
- End-to-end latency: <50ms
- Throughput: 10,000+ messages/second
- GPU utilization: Batch every 100 events or 10ms

References:
- Real-time ML systems design patterns
- Event-driven architecture for trading systems
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from cift.data.data_aggregator import MarketDataAggregator, MarketTick
from cift.data.order_book_processor import OrderBookProcessor

# Data connectors
# ML models
from cift.ml.ensemble import EnsembleMetaModel, EnsemblePrediction, build_ensemble

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""

    # Data sources
    polygon_api_key: str = ""
    databento_api_key: str = ""
    symbols: list[str] = field(default_factory=lambda: ["SPY"])

    # Feature extraction
    tick_window: int = 100  # Ticks for feature rolling window
    second_window: int = 60  # Seconds for aggregation
    minute_window: int = 30  # Minutes for trend features

    # Model inference
    batch_size: int = 100  # Max events before batch inference
    batch_timeout_ms: float = 10.0  # Max time before inference

    # Ensemble settings
    min_agreement: int = 3
    confidence_threshold: float = 0.65

    # Performance
    device: str = "cuda"
    num_workers: int = 4

    # Signal output
    enable_websocket: bool = True
    websocket_port: int = 8765
    nats_url: str = "nats://localhost:4222"


@dataclass
class InferenceResult:
    """Result of inference pipeline."""

    timestamp: float
    symbol: str
    prediction: EnsemblePrediction

    # Timing
    data_latency_ms: float  # Time from market event to pipeline
    feature_latency_ms: float  # Feature extraction time
    model_latency_ms: float  # Model inference time
    total_latency_ms: float  # Total pipeline latency


@dataclass
class FeatureBuffer:
    """Rolling window buffer for features."""

    tick_features: deque = field(default_factory=lambda: deque(maxlen=100))
    second_features: deque = field(default_factory=lambda: deque(maxlen=60))
    minute_features: deque = field(default_factory=lambda: deque(maxlen=30))

    regime_features: deque = field(default_factory=lambda: deque(maxlen=100))

    last_tick_time: float = 0
    last_second_time: float = 0
    last_minute_time: float = 0


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


class FeatureExtractor:
    """
    Real-time feature extraction from market data.

    Transforms raw L2/L3 data into model-ready features.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Order book processor
        self.ob_processor = OrderBookProcessor(
            price_levels=10,
            history_size=1000,
        )

        # Feature buffers per symbol
        self.buffers: dict[str, FeatureBuffer] = {}

        # Running statistics
        self.price_history: dict[str, deque] = {}
        self.volume_history: dict[str, deque] = {}

    def _get_buffer(self, symbol: str) -> FeatureBuffer:
        if symbol not in self.buffers:
            self.buffers[symbol] = FeatureBuffer()
            self.price_history[symbol] = deque(maxlen=10000)
            self.volume_history[symbol] = deque(maxlen=10000)
        return self.buffers[symbol]

    def process_tick(
        self,
        symbol: str,
        tick: MarketTick,
    ) -> dict[str, np.ndarray] | None:
        """
        Process a single market tick.

        Returns features if enough data accumulated.
        """
        buffer = self._get_buffer(symbol)
        now = time.time()

        # Update price/volume history
        self.price_history[symbol].append(tick.price)
        self.volume_history[symbol].append(tick.volume)

        # Extract tick-level features
        tick_features = self._extract_tick_features(tick, symbol)
        buffer.tick_features.append(tick_features)
        buffer.last_tick_time = now

        # Aggregate to second bars
        if now - buffer.last_second_time >= 1.0:
            second_features = self._aggregate_to_second(symbol, buffer)
            if second_features is not None:
                buffer.second_features.append(second_features)
            buffer.last_second_time = now

        # Aggregate to minute bars
        if now - buffer.last_minute_time >= 60.0:
            minute_features = self._aggregate_to_minute(symbol, buffer)
            if minute_features is not None:
                buffer.minute_features.append(minute_features)
            buffer.last_minute_time = now

        # Extract regime features
        regime_features = self._extract_regime_features(symbol, buffer)
        if regime_features is not None:
            buffer.regime_features.append(regime_features)

        # Check if we have enough data
        if len(buffer.tick_features) < 10:
            return None

        return self._build_feature_dict(symbol, buffer)

    def _extract_tick_features(self, tick: MarketTick, symbol: str) -> np.ndarray:
        """Extract features from single tick."""
        features = np.zeros(32, dtype=np.float32)

        # Price features
        features[0] = tick.price
        features[1] = tick.bid_price
        features[2] = tick.ask_price
        features[3] = tick.ask_price - tick.bid_price  # Spread

        # Volume features
        features[4] = tick.volume
        features[5] = tick.bid_size
        features[6] = tick.ask_size

        # Imbalance
        total_size = tick.bid_size + tick.ask_size
        if total_size > 0:
            features[7] = (tick.bid_size - tick.ask_size) / total_size

        # Returns
        prices = self.price_history.get(symbol, [])
        if len(prices) > 1:
            features[8] = (tick.price - prices[-1]) / (prices[-1] + 1e-8)  # 1-tick return
        if len(prices) > 10:
            features[9] = (tick.price - prices[-10]) / (prices[-10] + 1e-8)

        # Timestamp
        features[10] = tick.timestamp % (24 * 3600)  # Time of day

        return features

    def _aggregate_to_second(
        self,
        symbol: str,
        buffer: FeatureBuffer,
    ) -> np.ndarray | None:
        """Aggregate tick features to second bar."""
        if len(buffer.tick_features) < 5:
            return None

        # Get recent ticks
        recent_ticks = list(buffer.tick_features)[-100:]
        tick_array = np.array(recent_ticks)

        features = np.zeros(16, dtype=np.float32)

        # OHLC from prices
        prices = tick_array[:, 0]
        features[0] = prices[0]  # Open
        features[1] = prices.max()  # High
        features[2] = prices.min()  # Low
        features[3] = prices[-1]  # Close

        # Volume
        features[4] = tick_array[:, 4].sum()  # Total volume

        # Spread stats
        spreads = tick_array[:, 3]
        features[5] = spreads.mean()
        features[6] = spreads.std()

        # Imbalance stats
        imbalances = tick_array[:, 7]
        features[7] = imbalances.mean()
        features[8] = imbalances.std()

        # Tick count
        features[9] = len(recent_ticks)

        # Return
        if len(prices) > 1:
            features[10] = (prices[-1] - prices[0]) / (prices[0] + 1e-8)

        # Volatility
        if len(prices) > 5:
            returns = np.diff(prices) / (prices[:-1] + 1e-8)
            features[11] = returns.std() * np.sqrt(len(returns))

        return features

    def _aggregate_to_minute(
        self,
        symbol: str,
        buffer: FeatureBuffer,
    ) -> np.ndarray | None:
        """Aggregate second features to minute bar."""
        if len(buffer.second_features) < 10:
            return None

        recent_seconds = list(buffer.second_features)[-60:]
        second_array = np.array(recent_seconds)

        features = np.zeros(8, dtype=np.float32)

        # Minute OHLC
        closes = second_array[:, 3]
        features[0] = second_array[0, 0]  # Open
        features[1] = second_array[:, 1].max()  # High
        features[2] = second_array[:, 2].min()  # Low
        features[3] = closes[-1]  # Close

        # Volume
        features[4] = second_array[:, 4].sum()

        # Return
        features[5] = (closes[-1] - closes[0]) / (closes[0] + 1e-8)

        # Volatility
        features[6] = second_array[:, 11].mean()  # Avg second volatility

        return features

    def _extract_regime_features(
        self,
        symbol: str,
        buffer: FeatureBuffer,
    ) -> np.ndarray | None:
        """Extract features for regime detection."""
        if len(buffer.second_features) < 30:
            return None

        recent_seconds = np.array(list(buffer.second_features))

        features = np.zeros(16, dtype=np.float32)

        # Volatility features
        if len(recent_seconds) >= 60:
            features[0] = recent_seconds[-60:, 11].mean()  # 1m vol
        if len(recent_seconds) >= 300:
            features[1] = recent_seconds[-300:, 11].mean()  # 5m vol

        features[2] = features[1]  # 30m vol placeholder
        features[3] = recent_seconds[:, 11].std()  # Vol of vol

        # Spread features
        spreads = recent_seconds[:, 5]
        features[4] = spreads.mean()
        features[5] = spreads.std()
        features[6] = np.percentile(spreads, 90) / (spreads.mean() + 1e-8)

        # Volume features
        volumes = recent_seconds[:, 4]
        features[7] = volumes[-1] / (volumes.mean() + 1e-8)

        # Return features
        returns = recent_seconds[:, 10]
        features[9] = returns[-1]  # 1s return
        if len(returns) >= 5:
            features[10] = returns[-5:].sum()  # 5s return
        if len(returns) >= 30:
            features[11] = returns[-30:].sum()  # 30s return

        # Autocorrelation
        if len(returns) > 10:
            features[12] = np.corrcoef(returns[:-1], returns[1:])[0, 1]

        # Order flow
        imbalances = recent_seconds[:, 7]
        features[13] = imbalances.mean()
        features[14] = (imbalances > 0).mean()  # VPIN proxy

        return features

    def _build_feature_dict(
        self,
        symbol: str,
        buffer: FeatureBuffer,
    ) -> dict[str, np.ndarray]:
        """Build feature dictionary for all models."""
        return {
            "tick_features": np.array(list(buffer.tick_features)),
            "second_features": (
                np.array(list(buffer.second_features)) if buffer.second_features else None
            ),
            "minute_features": (
                np.array(list(buffer.minute_features)) if buffer.minute_features else None
            ),
            "regime_features": (
                np.array(list(buffer.regime_features)[-1]) if buffer.regime_features else None
            ),
        }


# ============================================================================
# INFERENCE ENGINE
# ============================================================================


class InferenceEngine:
    """
    Batched model inference engine.

    Collects events and runs inference in batches for GPU efficiency.
    """

    def __init__(
        self,
        ensemble: EnsembleMetaModel,
        config: PipelineConfig,
    ):
        self.ensemble = ensemble
        self.config = config

        # Pending inference queue
        self.pending: list[dict[str, Any]] = []
        self.last_inference_time = time.time()

        # Callbacks for results
        self.callbacks: list[Callable[[InferenceResult], None]] = []

    def add_callback(self, callback: Callable[[InferenceResult], None]):
        """Register callback for inference results."""
        self.callbacks.append(callback)

    async def submit(
        self,
        symbol: str,
        features: dict[str, np.ndarray],
        data_timestamp: float,
    ):
        """
        Submit features for inference.

        May trigger batch inference if conditions met.
        """
        self.pending.append(
            {
                "symbol": symbol,
                "features": features,
                "data_timestamp": data_timestamp,
                "submit_time": time.time(),
            }
        )

        # Check if we should run inference
        should_infer = (
            len(self.pending) >= self.config.batch_size
            or (time.time() - self.last_inference_time) * 1000 >= self.config.batch_timeout_ms
        )

        if should_infer:
            await self._run_batch_inference()

    async def _run_batch_inference(self):
        """Run inference on pending batch."""
        if not self.pending:
            return

        batch = self.pending.copy()
        self.pending.clear()
        self.last_inference_time = time.time()

        time.time()

        for item in batch:
            try:
                result = await self._infer_single(item)

                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Inference error for {item['symbol']}: {e}")

    async def _infer_single(self, item: dict[str, Any]) -> InferenceResult:
        """Run inference for single item."""
        start_time = time.time()

        features = item["features"]

        # Prepare inputs for ensemble
        # Hawkes events (from tick features)
        hawkes_events = None
        if features.get("tick_features") is not None:
            tick_data = features["tick_features"]
            # Convert to event format: [timestamp, type, value]
            hawkes_events = np.zeros((len(tick_data), 3))
            for i, t in enumerate(tick_data):
                hawkes_events[i, 0] = i  # Relative timestamp
                hawkes_events[i, 1] = 0 if t[7] > 0 else 1  # Buy/sell based on imbalance
                hawkes_events[i, 2] = abs(t[4])  # Volume

        # Transformer features
        transformer_features = None
        if features.get("tick_features") is not None:
            # Combine multi-timeframe features
            tick_f = (
                features["tick_features"][-50:]
                if len(features["tick_features"]) > 50
                else features["tick_features"]
            )

            # Pad if needed
            if len(tick_f) < 50:
                pad = np.zeros((50 - len(tick_f), tick_f.shape[1]))
                tick_f = np.vstack([pad, tick_f])

            transformer_features = {
                "tick": tick_f[:, :32],  # First 32 features
                "second": (
                    features.get("second_features", np.zeros((60, 16)))[-60:]
                    if features.get("second_features") is not None
                    else np.zeros((60, 16))
                ),
                "minute": (
                    features.get("minute_features", np.zeros((30, 8)))[-30:]
                    if features.get("minute_features") is not None
                    else np.zeros((30, 8))
                ),
            }

        # HMM features
        hmm_features = features.get("regime_features")

        # XGBoost features
        xgboost_features = None
        if features.get("tick_features") is not None:
            # Create alt-data feature vector (27 features)
            xgboost_features = np.zeros(27, dtype=np.float32)

            # Fill in microstructure features from tick data
            recent = features["tick_features"][-10:]
            xgboost_features[21] = np.mean([t[7] for t in recent])  # Order flow imbalance
            xgboost_features[22] = 0.5  # VPIN placeholder
            xgboost_features[23] = 0.1  # Kyle lambda placeholder
            xgboost_features[24] = (
                np.mean([t[3] for t in recent]) / (recent[-1][0] + 1e-8) * 10000
            )  # Spread percentile
            xgboost_features[25] = 1.0  # Volume ratio placeholder
            xgboost_features[26] = (
                np.std([t[8] for t in recent]) if len(recent) > 1 else 0
            )  # Realized vol

        feature_time = time.time()

        # Run ensemble prediction
        prediction = self.ensemble.predict(
            hawkes_events=hawkes_events,
            transformer_features=transformer_features,
            hmm_features=hmm_features,
            xgboost_features=xgboost_features,
            target_symbol=item["symbol"],
            timestamp=time.time(),
        )

        model_time = time.time()

        return InferenceResult(
            timestamp=time.time(),
            symbol=item["symbol"],
            prediction=prediction,
            data_latency_ms=(item["submit_time"] - item["data_timestamp"]) * 1000,
            feature_latency_ms=(feature_time - start_time) * 1000,
            model_latency_ms=(model_time - feature_time) * 1000,
            total_latency_ms=(model_time - item["data_timestamp"]) * 1000,
        )


# ============================================================================
# MAIN PIPELINE
# ============================================================================


class InferencePipeline:
    """
    Complete real-time inference pipeline.

    Connects data sources to feature extraction to model inference.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Build ensemble
        self.ensemble = build_ensemble(
            config={
                "min_agreement": config.min_agreement,
                "confidence_threshold": config.confidence_threshold,
            },
            device=config.device,
        )

        # Feature extractor
        self.feature_extractor = FeatureExtractor(config)

        # Inference engine
        self.inference_engine = InferenceEngine(self.ensemble, config)

        # Data aggregator
        self.data_aggregator = None

        # State
        self.running = False
        self._tasks: list[asyncio.Task] = []

        logger.info("InferencePipeline initialized")

    async def start(self):
        """Start the pipeline."""
        logger.info("Starting inference pipeline...")
        self.running = True

        # Initialize data aggregator
        self.data_aggregator = MarketDataAggregator(
            polygon_api_key=self.config.polygon_api_key,
            databento_api_key=self.config.databento_api_key,
            symbols=self.config.symbols,
        )

        # Register tick callback
        self.data_aggregator.on_tick(self._handle_tick)

        # Start data connection
        self._tasks.append(asyncio.create_task(self.data_aggregator.start()))

        logger.info(f"Pipeline started for symbols: {self.config.symbols}")

    async def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping inference pipeline...")
        self.running = False

        if self.data_aggregator:
            await self.data_aggregator.stop()

        for task in self._tasks:
            task.cancel()

        logger.info("Pipeline stopped")

    async def _handle_tick(self, tick: MarketTick):
        """Handle incoming market tick."""
        if not self.running:
            return

        try:
            # Extract features
            features = self.feature_extractor.process_tick(tick.symbol, tick)

            if features is not None:
                # Submit for inference
                await self.inference_engine.submit(
                    symbol=tick.symbol,
                    features=features,
                    data_timestamp=tick.timestamp,
                )

        except Exception as e:
            logger.error(f"Error handling tick: {e}")

    def on_prediction(self, callback: Callable[[InferenceResult], None]):
        """Register callback for predictions."""
        self.inference_engine.add_callback(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "running": self.running,
            "symbols": self.config.symbols,
            "pending_inferences": len(self.inference_engine.pending),
        }


# ============================================================================
# WEBSOCKET BROADCASTER (optional)
# ============================================================================


class WebSocketBroadcaster:
    """
    WebSocket server for broadcasting predictions.

    Clients can subscribe to predictions for specific symbols.
    """

    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: set = set()
        self.server = None

    async def start(self):
        """Start WebSocket server."""
        try:
            import websockets

            async def handler(websocket, path):
                self.clients.add(websocket)
                try:
                    async for _message in websocket:
                        # Handle subscription messages
                        pass
                finally:
                    self.clients.discard(websocket)

            self.server = await websockets.serve(handler, "0.0.0.0", self.port)
            logger.info(f"WebSocket server started on port {self.port}")

        except ImportError:
            logger.warning("websockets not installed, WebSocket broadcasting disabled")

    async def broadcast(self, result: InferenceResult):
        """Broadcast prediction to all clients."""
        if not self.clients:
            return

        import json

        message = json.dumps(
            {
                "type": "prediction",
                "timestamp": result.timestamp,
                "symbol": result.symbol,
                "direction": result.prediction.direction,
                "probability": result.prediction.direction_probability,
                "confidence": result.prediction.confidence,
                "should_trade": result.prediction.should_trade,
                "position_size": result.prediction.position_size,
                "latency_ms": result.total_latency_ms,
            }
        )

        for client in self.clients.copy():
            try:
                await client.send(message)
            except Exception:
                self.clients.discard(client)


__all__ = [
    "InferencePipeline",
    "PipelineConfig",
    "InferenceResult",
    "InferenceEngine",
    "FeatureExtractor",
    "WebSocketBroadcaster",
]
