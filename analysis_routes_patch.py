# Stock Analysis Engine (AI-powered recommendations)
try:
    from cift.api.routes import analysis
    app.include_router(analysis.router, prefix="/api/v1")
    logger.info("Stock analysis routes loaded")
except ImportError as e:
    logger.warning(f"Stock analysis routes not available: {e}")

