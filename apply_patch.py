import re

# Read main.py
with open('/app/cift/api/main.py', 'r') as f:
    content = f.read()

# The code to inject
patch = '''
# Stock Analysis Engine (AI-powered recommendations)
try:
    from cift.api.routes import analysis
    app.include_router(analysis.router, prefix="/api/v1")
    logger.info("Stock analysis routes loaded")
except ImportError as e:
    logger.warning(f"Stock analysis routes not available: {e}")

'''

# Find the ML Inference routes marker and insert before it
if '# Stock Analysis Engine' not in content:
    content = content.replace(
        '# ML Inference routes (Phase 8 - ML Implementation)',
        patch + '# ML Inference routes (Phase 8 - ML Implementation)'
    )
    
    with open('/app/cift/api/main.py', 'w') as f:
        f.write(content)
    print("Patch applied successfully!")
else:
    print("Analysis routes already present")

# Verify
import subprocess
result = subprocess.run(['grep', '-c', 'analysis', '/app/cift/api/main.py'], capture_output=True, text=True)
print(f"Lines containing 'analysis': {result.stdout.strip()}")
