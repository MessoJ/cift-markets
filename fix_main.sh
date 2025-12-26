#!/bin/bash
# Fix the analysis routes in main.py

# Check current state
grep -n 'analysis' /app/cift/api/main.py

# If no analysis routes, inject them
if ! grep -q 'Stock analysis routes' /app/cift/api/main.py; then
    # Find line number of "# ML Inference routes" and insert before it
    LINE=$(grep -n '# ML Inference routes' /app/cift/api/main.py | head -1 | cut -d: -f1)
    if [ -n "$LINE" ]; then
        # Insert the analysis routes block before ML Inference routes
        sed -i "${LINE}i\\
# Stock Analysis Engine (AI-powered recommendations)\\
try:\\
    from cift.api.routes import analysis\\
    app.include_router(analysis.router, prefix=\"/api/v1\")\\
    logger.info(\"Stock analysis routes loaded\")\\
except ImportError as e:\\
    logger.warning(f\"Stock analysis routes not available: {e}\")\\
" /app/cift/api/main.py
        echo "Analysis routes injected at line $LINE"
    fi
fi

# Verify
grep -n 'analysis' /app/cift/api/main.py
