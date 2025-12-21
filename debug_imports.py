import os
import sys

sys.path.append(os.getcwd())

print("Importing cift.core.execution_engine...")
try:
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print("Importing cift.api.routes.funding...")
try:
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
