#!/bin/bash
# Script to check available TPU accelerator types in common zones

echo "Checking available TPU types in us-central1-a..."
gcloud compute tpus accelerator-types list --zone=us-central1-a

echo "Checking available TPU types in us-central1-b..."
gcloud compute tpus accelerator-types list --zone=us-central1-b

echo "Checking available TPU types in us-central1-c..."
gcloud compute tpus accelerator-types list --zone=us-central1-c

echo "Checking available TPU types in us-east1-c..."
gcloud compute tpus accelerator-types list --zone=us-east1-c

echo "Checking available TPU types in us-east1-d..."
gcloud compute tpus accelerator-types list --zone=us-east1-d

echo "Checking available TPU types in us-west4-a..."
gcloud compute tpus accelerator-types list --zone=us-west4-a

echo "----------------------------------------------------------------"
echo "RECOMMENDATION:"
echo "If v6/v5 are not available or have quota issues, run this command"
echo "to create a v3-8 TPU with Python 3.10 support:"
echo ""
echo "gcloud compute tpus tpu-vm create cift-tpu-v3 \\"
echo "  --zone=us-central1-b \\"
echo "  --accelerator-type=v3-8 \\"
echo "  --version=tpu-ubuntu2204-base"
