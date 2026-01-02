#!/bin/bash

# 1. Configuration
TPU_NAME="tpu-v4-32"
ZONE="us-central2-b"
PROJECT_ID="cift-447716" # Replace if different, usually inferred
REPO_URL="https://github.com/mesof/cift-markets.git"

# 2. Delete existing TPU (if any)
echo "Deleting existing TPU..."
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet

# 3. Create new TPU
echo "Creating new TPU..."
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=v4-32 \
  --version=tpu-vm-pt-2.0 \
  --quiet

# 4. Wait for it to be ready (create command usually waits, but just in case)
echo "Waiting for TPU to be ready..."
sleep 30

# 5. SSH and Setup
echo "Setting up TPU workers..."
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "
  # Kill any lingering processes
  pkill -9 python || true
  
  # Clean directory
  rm -rf cift-markets
  
  # Clone Repo
  git clone $REPO_URL
  cd cift-markets
  
  # Install Dependencies
  # Uninstall mismatched versions first
  pip uninstall -y torch torch-xla libtpu-nightly
  
  # Install correct PyTorch/XLA 2.4.0 stack
  pip install torch~=2.4.0 torch_xla[tpu]~=2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html
  
  # Install missing python dependencies
  pip install pydantic loguru
  
  # Verify installation
  python3 -c 'import torch; import torch_xla; import torch_xla.core.xla_model as xm; print(\"Torch:\", torch.__version__); print(\"XLA:\", torch_xla.__version__); print(\"Device:\", xm.xla_device())'
"

echo "Setup Complete! You can now run training."
