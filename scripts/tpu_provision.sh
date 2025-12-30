#!/bin/bash

# Option 1: Try TPU v6 (Trillium)
# Note: This is likely to fail without specific allowlisting/quota.
echo "Attempting to create TPU v6 (Trillium)..."
gcloud compute tpus tpu-vm create cift-tpu-v6 \
  --zone=us-central1-a \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base

# Option 2: The Reliable Fallback (TPU v3-8 + Python 3.10)
# If v6 fails, run this command. It uses the Ubuntu 22.04 image to support
# the Python 3.10+ syntax in our codebase, running on widely available v3 hardware.
# echo "Creating TPU v3-8 with Python 3.10 support..."
# gcloud compute tpus tpu-vm create cift-tpu-v3 \
#   --zone=us-central1-b \
#   --accelerator-type=v3-8 \
#   --version=tpu-ubuntu2204-base
