import sys
import os
sys.path.append(os.getcwd())

print("Importing cift.core.execution_engine...")
try:
    import cift.core.execution_engine
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print("Importing cift.api.routes.funding...")
try:
    import cift.api.routes.funding
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
