# GCP SETUP SCRIPT (Run in Cloud Shell)
# Based on your provided availability list

# OPTION 1: TPU v3-8 (Recommended - High Performance)
# Zone: us-central1-a
export ZONE="us-central1-a"
export TPU_TYPE="v3-8"
export TPU_NAME="cift-tpu-v3"

# OPTION 2: TPU v2-8 (Cheaper, good for testing)
# Uncomment lines below to use this instead if v3 is unavailable
# export ZONE="us-central1-b"
# export TPU_TYPE="v2-8"
# export TPU_NAME="cift-tpu-v2"

# OPTION 3: TPU v5e (Newer, Efficient)
# Uncomment lines below to use this instead
# export ZONE="us-central1-a"
# export TPU_TYPE="v5litepod-8"
# export TPU_NAME="cift-tpu-v5"

# ---------------------------------------------------------
# EXECUTION COMMANDS
# ---------------------------------------------------------

# 1. Set Project
gcloud config set project gen-lang-client-0197406861

# 2. Enable APIs
echo "Enabling APIs..."
gcloud services enable compute.googleapis.com storage.googleapis.com tpu.googleapis.com

# 3. Create Bucket (if not exists)
export BUCKET_NAME="cift-data-gen-lang-client-0197406861"
echo "Creating Bucket gs://$BUCKET_NAME..."
gcloud storage buckets create gs://$BUCKET_NAME --location=us-central1 || echo "Bucket might already exist, continuing..."

# 4. Create TPU VM
# ATTEMPT 2: Try us-central1-b (since us-central1-a was full)
echo "Creating TPU VM (v3-8) in us-central1-b..."
gcloud compute tpus tpu-vm create cift-tpu-v3 \
  --zone=us-central1-b \
  --accelerator-type=v3-8 \
  --version=tpu-vm-pt-2.0

# FALLBACK: If v3-8 is full in both zones, use v2-8 (older but available)
# gcloud compute tpus tpu-vm create cift-tpu-v2 \
#   --zone=us-central1-b \
#   --accelerator-type=v2-8 \
#   --version=tpu-vm-pt-2.0

echo "----------------------------------------------------------------"
echo "SETUP COMPLETE!"
echo "To connect to your TPU VM, run:"
echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
echo "----------------------------------------------------------------"
