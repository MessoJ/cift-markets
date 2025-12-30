"""
CIFT Markets - Inference API Routes

FastAPI routes for ML inference endpoints.

Endpoints:
- POST /api/v1/predict: Single prediction request
- GET /api/v1/predict/stream: WebSocket for streaming predictions
- GET /api/v1/models/status: Model status and health
- POST /api/v1/models/reload: Reload models from disk

Integration:
- Connects to InferencePipeline for real-time predictions
- Provides REST API for on-demand predictions
- WebSocket for streaming predictions to frontend
"""

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, Field

from cift.inference.pipeline import (
    InferencePipeline,
    InferenceResult,
    PipelineConfig,
)
from cift.ml.ensemble import build_ensemble

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for single prediction."""
    symbol: str = Field(..., description="Symbol to predict")
    tick_features: list[list[float]] | None = Field(None, description="Recent tick features")
    second_features: list[list[float]] | None = Field(None, description="Second bar features")
    regime_features: list[float] | None = Field(None, description="Regime detection features")
    xgboost_features: list[float] | None = Field(None, description="Alternative data features")


class PredictionResponse(BaseModel):
    """Response with prediction."""
    timestamp: float
    symbol: str

    # Primary signal
    direction: str                   # "long", "short", "neutral"
    direction_probability: float
    magnitude: float

    # Confidence
    confidence: float
    model_agreement: int

    # Regime
    current_regime: str
    regime_probability: float

    # Trade recommendation
    should_trade: bool
    position_size: float
    stop_loss_bps: float
    take_profit_bps: float

    # Model contributions
    model_weights: dict[str, float]

    # Latency
    inference_latency_ms: float


class ModelStatus(BaseModel):
    """Model status information."""
    model_name: str
    loaded: bool
    last_prediction_time: float | None
    total_predictions: int
    avg_latency_ms: float


class SystemStatus(BaseModel):
    """Overall system status."""
    status: str  # "healthy", "degraded", "error"
    models: list[ModelStatus]
    pipeline_running: bool
    active_symbols: list[str]
    total_predictions: int
    uptime_seconds: float


# ============================================================================
# API STATE
# ============================================================================

class InferenceAPIState:
    """Singleton state for inference API."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.pipeline: InferencePipeline | None = None
        self.ensemble = None
        self.start_time = time.time()
        self.total_predictions = 0
        self.model_stats = {
            "hawkes": {"predictions": 0, "latencies": []},
            "transformer": {"predictions": 0, "latencies": []},
            "hmm": {"predictions": 0, "latencies": []},
            "gnn": {"predictions": 0, "latencies": []},
            "xgboost": {"predictions": 0, "latencies": []},
        }

        # WebSocket connections
        self.ws_connections: set = set()

        self._initialized = True

    async def initialize(self, config: PipelineConfig | None = None):
        """Initialize the inference system."""
        if self.ensemble is None:
            logger.info("Initializing inference ensemble...")
            self.ensemble = build_ensemble()
            logger.info("Ensemble initialized")

        if config and self.pipeline is None:
            logger.info("Initializing inference pipeline...")
            self.pipeline = InferencePipeline(config)
            self.pipeline.on_prediction(self._on_prediction)
            logger.info("Pipeline initialized")

    async def _on_prediction(self, result: InferenceResult):
        """Handle prediction from pipeline."""
        self.total_predictions += 1

        # Broadcast to WebSocket clients
        await self._broadcast_prediction(result)

    async def _broadcast_prediction(self, result: InferenceResult):
        """Broadcast prediction to all WebSocket clients."""
        import json

        message = json.dumps({
            "type": "prediction",
            "data": {
                "timestamp": result.timestamp,
                "symbol": result.symbol,
                "direction": result.prediction.direction,
                "probability": result.prediction.direction_probability,
                "confidence": result.prediction.confidence,
                "should_trade": result.prediction.should_trade,
                "latency_ms": result.total_latency_ms,
            }
        })

        for ws in self.ws_connections.copy():
            try:
                await ws.send_text(message)
            except Exception:
                self.ws_connections.discard(ws)


# Global state instance
api_state = InferenceAPIState()


def get_api_state() -> InferenceAPIState:
    """Dependency to get API state."""
    return api_state


# ============================================================================
# API ROUTER
# ============================================================================

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    state: InferenceAPIState = Depends(get_api_state),
):
    """
    Make a single prediction for a symbol.

    Accepts feature arrays and returns ensemble prediction.
    """
    import numpy as np

    if state.ensemble is None:
        await state.initialize()

    start_time = time.time()

    try:
        # Convert request to numpy arrays
        tick_features = None
        if request.tick_features:
            tick_features = np.array(request.tick_features, dtype=np.float32)

        transformer_features = None
        if tick_features is not None and len(tick_features) >= 10:
            transformer_features = {
                "tick": tick_features[-50:] if len(tick_features) > 50 else tick_features,
                "second": np.zeros((60, 16)),
                "minute": np.zeros((30, 8)),
            }
            if request.second_features:
                second_arr = np.array(request.second_features, dtype=np.float32)
                transformer_features["second"] = second_arr[-60:]

        hmm_features = None
        if request.regime_features:
            hmm_features = np.array(request.regime_features, dtype=np.float32)

        xgboost_features = None
        if request.xgboost_features:
            xgboost_features = np.array(request.xgboost_features, dtype=np.float32)

        # Prepare hawkes events
        hawkes_events = None
        if tick_features is not None:
            hawkes_events = np.zeros((len(tick_features), 3))
            for i, t in enumerate(tick_features):
                hawkes_events[i, 0] = i
                hawkes_events[i, 1] = 0 if t[7] > 0 else 1
                hawkes_events[i, 2] = abs(t[4])

        # Run prediction
        prediction = state.ensemble.predict(
            hawkes_events=hawkes_events,
            transformer_features=transformer_features,
            hmm_features=hmm_features,
            xgboost_features=xgboost_features,
            target_symbol=request.symbol,
            timestamp=time.time(),
        )

        state.total_predictions += 1
        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            timestamp=prediction.timestamp,
            symbol=request.symbol,
            direction=prediction.direction,
            direction_probability=prediction.direction_probability,
            magnitude=prediction.magnitude,
            confidence=prediction.confidence,
            model_agreement=prediction.model_agreement,
            current_regime=prediction.current_regime.name,
            regime_probability=prediction.regime_probability,
            should_trade=prediction.should_trade,
            position_size=prediction.position_size,
            stop_loss_bps=prediction.stop_loss_bps,
            take_profit_bps=prediction.take_profit_bps,
            model_weights=prediction.model_weights,
            inference_latency_ms=inference_time,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.websocket("/predict/stream")
async def predict_stream(
    websocket: WebSocket,
    state: InferenceAPIState = Depends(get_api_state),
):
    """
    WebSocket endpoint for streaming predictions.

    Clients receive real-time predictions as they're generated.
    """
    await websocket.accept()
    state.ws_connections.add(websocket)

    try:
        logger.info("WebSocket client connected for prediction streaming")

        # Keep connection alive
        while True:
            try:
                # Handle any incoming messages (e.g., subscription changes)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Process subscription message
                import json
                data = json.loads(message)

                if data.get("type") == "subscribe":
                    symbols = data.get("symbols", [])
                    logger.info(f"Client subscribed to: {symbols}")

            except TimeoutError:
                # Send heartbeat
                await websocket.send_text('{"type": "heartbeat"}')

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        state.ws_connections.discard(websocket)


@router.get("/status", response_model=SystemStatus)
async def get_model_status(
    state: InferenceAPIState = Depends(get_api_state),
):
    """Get status of all models and the inference system."""
    models = []

    model_names = ["hawkes", "transformer", "hmm", "gnn", "xgboost"]

    for name in model_names:
        stats = state.model_stats.get(name, {"predictions": 0, "latencies": []})
        latencies = stats["latencies"]

        models.append(ModelStatus(
            model_name=name,
            loaded=state.ensemble is not None,
            last_prediction_time=latencies[-1] if latencies else None,
            total_predictions=stats["predictions"],
            avg_latency_ms=sum(latencies[-100:]) / len(latencies[-100:]) if latencies else 0,
        ))

    # Determine overall status
    if state.ensemble is None:
        status = "error"
    elif state.pipeline and state.pipeline.running:
        status = "healthy"
    else:
        status = "degraded"

    return SystemStatus(
        status=status,
        models=models,
        pipeline_running=state.pipeline.running if state.pipeline else False,
        active_symbols=state.pipeline.config.symbols if state.pipeline else [],
        total_predictions=state.total_predictions,
        uptime_seconds=time.time() - state.start_time,
    )


@router.post("/models/reload")
async def reload_models(
    state: InferenceAPIState = Depends(get_api_state),
):
    """Reload models from disk."""
    try:
        logger.info("Reloading models...")

        # Rebuild ensemble
        state.ensemble = build_ensemble()

        # Reconnect to pipeline if running
        if state.pipeline:
            state.pipeline.ensemble = state.ensemble

        logger.info("Models reloaded successfully")

        return {"status": "success", "message": "Models reloaded"}

    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class PipelineStartRequest(BaseModel):
    """Request to start the pipeline."""
    symbols: list[str] = Field(default=["SPY"], description="Symbols to trade")


@router.post("/pipeline/start")
async def start_pipeline(
    request: PipelineStartRequest = None,
    state: InferenceAPIState = Depends(get_api_state),
):
    """Start the inference pipeline."""
    from cift.core.config import settings

    symbols = request.symbols if request else ["SPY"]
    try:
        config = PipelineConfig(
            polygon_api_key=settings.polygon_api_key,
            symbols=symbols,
        )

        await state.initialize(config)
        await state.pipeline.start()

        return {"status": "success", "message": f"Pipeline started for {symbols}"}

    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/pipeline/stop")
async def stop_pipeline(
    state: InferenceAPIState = Depends(get_api_state),
):
    """Stop the inference pipeline."""
    try:
        if state.pipeline:
            await state.pipeline.stop()
            return {"status": "success", "message": "Pipeline stopped"}
        else:
            return {"status": "warning", "message": "Pipeline not running"}

    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


__all__ = [
    "router",
    "PredictionRequest",
    "PredictionResponse",
    "SystemStatus",
    "ModelStatus",
]
