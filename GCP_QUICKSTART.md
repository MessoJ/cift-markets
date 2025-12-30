# GCP Training Quick Start 🚀

## 5-MINUTE SETUP

### Step 1: Download Data Locally (Run Now)
```powershell
cd c:\Users\mesof\cift-markets
pip install yfinance pandas numpy requests pyarrow
python scripts/download_data.py
```
**Wait for:** "✅ ALL DATA DOWNLOADED!" message

---

### Step 2: Create GCP Resources (One-Time)
Go to: https://console.cloud.google.com

```bash
# In Google Cloud Shell (click terminal icon top-right)
gcloud projects create cift-stat-arb --name="CIFT Stat Arb"
gcloud config set project cift-stat-arb
gcloud services enable compute.googleapis.com storage.googleapis.com

# Create bucket
gsutil mb -l us-central1 gs://cift-stat-arb-data

# Create VM (e2-standard-8 = 8 vCPU, 32GB RAM, ~$0.27/hr)
gcloud compute instances create stat-arb-trainer \
  --zone=us-central1-a \
  --machine-type=e2-standard-8 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB
```

---

### Step 3: Upload Data
```powershell
# In your local terminal
gcloud auth login
gcloud config set project cift-stat-arb

# Upload all data
gsutil -m cp -r data/* gs://cift-stat-arb-data/data/

# Upload training script
gsutil cp scripts/gcp_train.py gs://cift-stat-arb-data/code/
```

---

### Step 4: SSH to VM and Run Training
```bash
# SSH into VM
gcloud compute ssh stat-arb-trainer --zone=us-central1-a

# On the VM - setup environment
sudo apt update
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/stat_arb_env
source ~/stat_arb_env/bin/activate
pip install numpy pandas yfinance google-cloud-storage pyarrow

# Download data and code from bucket
mkdir -p data
gsutil -m cp -r gs://cift-stat-arb-data/data/* data/
gsutil cp gs://cift-stat-arb-data/code/gcp_train.py .

# RUN TRAINING
python gcp_train.py
```

---

### Step 5: Monitor Progress
```bash
# In another terminal, SSH to VM
gcloud compute ssh stat-arb-trainer --zone=us-central1-a

# Watch logs in real-time
tail -f training.log

# Or check periodically
cat training.log
```

---

## What to Look For ✅

### GOOD Signs:
| Metric | Target |
|--------|--------|
| Train Sharpe | > 0.7 |
| Val Sharpe | > 0.6 |
| Test Sharpe | > 0.5 |
| Train/Val Ratio | < 1.5 |
| Cointegrated Pairs | 30+ |

### ⚠️ WARNING Signs:
- Train Sharpe >> Val Sharpe (overfitting!)
- < 20 cointegrated pairs (expand universe)
- Test Sharpe < 0.3 (poor generalization)

---

## After Training

### Download Results
```powershell
# On local machine
gsutil cp gs://cift-stat-arb-data/models/training_results_*.json .
```

### Apply Best Parameters
Edit `cift/ml/production_stat_arb.py`:
```python
CONFIG = {
    'ENTRY_Z': <from results>,
    'EXIT_Z': <from results>,
    'LOOKBACK': <from results>,
    'MAX_PAIRS': <from results>,
}
```

### Stop VM (Save Money!)
```bash
gcloud compute instances stop stat-arb-trainer --zone=us-central1-a
```

---

## Cost Estimate
| Resource | Cost |
|----------|------|
| VM (e2-standard-8) | ~$0.27/hour |
| Storage (10GB) | ~$0.20/month |
| **3-hour training** | **~$1.00** |

---

## Files You Need

| File | What It Does |
|------|--------------|
| `scripts/download_data.py` | Downloads equity/crypto data |
| `scripts/gcp_train.py` | Main training script for GCP |
| `docs/GCP_TRAINING_GUIDE.md` | Full detailed guide |
| `cift/ml/production_stat_arb.py` | Update with results |

---

**Total Time: ~30 minutes** (mostly waiting for data download and training)
