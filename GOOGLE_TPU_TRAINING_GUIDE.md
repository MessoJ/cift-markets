# 🚀 Migrating ML Training to Google Cloud TPUs

This guide details how to move your heavy ML training workloads from Azure (CPU-only) to Google Cloud TPUs for massive performance gains and cost efficiency.

## 1. Why Move?
- **Azure VM:** Your current `Standard_D2s_v3` has no GPU. Training deep learning models (Transformers, LSTMs) will take **days or weeks**.
- **Google TPU:** Tensor Processing Units are designed specifically for matrix math. A TPU v2-8 or v3-8 can train the same model in **minutes or hours**.

## 2. Prerequisites
1.  **Google Cloud Account:** Create one at [console.cloud.google.com](https://console.cloud.google.com).
2.  **Enable APIs:** Enable "Compute Engine API" and "TPU API".
3.  **Storage Bucket:** Create a GCS bucket (e.g., `gs://cift-ml-data`) to store your dataset and model checkpoints.

## 3. Step-by-Step Migration Guide

### Phase 1: Data Export (From Azure)
You need to get your training data out of the Azure VM and into a format Google Cloud can access.

1.  **Export Data to CSV/Parquet:**
    SSH into your Azure VM and run a script to dump your training data.
    ```bash
    # Example: Dump market data from QuestDB/Postgres to Parquet
    python scripts/export_training_data.py --output data/training_set.parquet
    ```

2.  **Upload to Google Cloud Storage (GCS):**
    The fastest way is to install `gsutil` or use the web console.
    ```bash
    # Install Google Cloud SDK
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh
    
    # Authenticate
    ./google-cloud-sdk/bin/gcloud auth login
    
    # Upload
    ./google-cloud-sdk/bin/gsutil cp data/training_set.parquet gs://cift-ml-data/
    ```

### Phase 2: Training on Google Colab (Easiest Start)
For immediate access to TPUs without setting up complex infrastructure, use Google Colab Pro.

1.  **Open Colab:** Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Runtime Settings:** Click `Runtime` -> `Change runtime type` -> Select `TPU`.
3.  **Install Dependencies:**
    ```python
    !pip install torch torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/tpu-pytorch/wheels/cuda/117/torch_xla-2.1.0-cp310-cp310-linux_x86_64.whl
    !pip install transformers polars
    ```
4.  **Mount GCS:**
    ```python
    from google.colab import auth
    auth.authenticate_user()
    ```
5.  **Training Script (PyTorch XLA):**
    You must modify your PyTorch code to use `torch_xla`.

    ```python
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

    def train_loop(index, flags):
        # 1. Set Device
        device = xm.xla_device()
        
        # 2. Load Data (Use DistributedSampler)
        dataset = MyDataset("gs://cift-ml-data/training_set.parquet")
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=sampler)
        
        # 3. Model to Device
        model = MyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 4. Loop
        model.train()
        for epoch in range(flags['epochs']):
            para_loader = pl.ParallelLoader(loader, [device])
            for batch in para_loader.per_device_loader(device):
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                
                # Optimizer Step (XLA specific)
                xm.optimizer_step(optimizer)
                
            # Print metrics (only on master node)
            if xm.is_master_ordinal():
                print(f"Epoch {epoch} complete. Loss: {loss.item()}")
                
        # 5. Save Model
        xm.save(model.state_dict(), "model_tpu.pth")

    # Run on all 8 TPU cores
    xmp.spawn(train_loop, args=(flags,), nprocs=8, start_method='fork')
    ```

### Phase 3: Metrics & Monitoring

**What to Watch:**
1.  **Loss (Training & Validation):** Should decrease steadily. If it flatlines immediately, check your learning rate.
2.  **Throughput (samples/sec):** TPUs shine here. You should see thousands of samples processed per second.
3.  **XLA Compilation:** The first few steps will be slow as XLA compiles the graph. This is normal. Subsequent steps should be lightning fast.

**Success Criteria:**
- **Convergence:** Loss reaches a stable minimum.
- **Generalization:** Validation loss is close to training loss (not overfitting).
- **Speed:** Training completes in < 10% of the time it took on CPU.

### Phase 4: Deployment (Back to Azure)

Once training is done:
1.  **Download Model:**
    ```python
    # In Colab
    from google.colab import files
    files.download('model_tpu.pth')
    ```
    Or copy to GCS: `gsutil cp model_tpu.pth gs://cift-ml-data/`

2.  **Upload to Azure:**
    SCP the model back to your Azure VM.
    ```bash
    scp model_tpu.pth azureuser@20.250.40.67:/home/azureuser/cift-markets/models/
    ```

3.  **Inference:**
    Your Azure VM (CPU) can run *inference* reasonably well for single predictions. Load the model using standard PyTorch:
    ```python
    model = MyModel()
    model.load_state_dict(torch.load("models/model_tpu.pth", map_location=torch.device('cpu')))
    model.eval()
    ```

## 4. Cost Savings
- **Azure:** You stopped the heavy training on the always-on VM.
- **Google:** You only pay for the TPU seconds you use. Colab Free/Pro is a flat fee, which is extremely cost-effective.

## 5. Next Steps
1.  **Export your data** from Azure.
2.  **Set up the Colab notebook** using the template above.
3.  **Run the training.**
4.  **Deploy the `.pth` file** back to Azure for the live API.
