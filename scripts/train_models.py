import argparse
import logging
import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cift.ml.gnn import CrossAssetGNN
from cift.ml.hawkes import HawkesOrderFlowModel
from cift.ml.transformer import OrderFlowTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000, n_assets=10, seq_len=60):
    """Generates synthetic data for testing the training pipeline."""
    logger.info("Generating synthetic market data...")

    # Transformer Data: (Batch, Seq, Features)
    X_trans = torch.randn(n_samples, seq_len, 16)
    y_trans = torch.randint(0, 2, (n_samples,)) # Binary classification (Up/Down)

    # GNN Data: (Batch, Seq, Nodes, Features)
    X_gnn = torch.randn(n_samples, seq_len, n_assets, 8)

    # Hawkes Data: List of tensors (event times)
    X_hawkes = [torch.sort(torch.rand(np.random.randint(10, 50))).values for _ in range(n_samples)]

    return X_trans, y_trans, X_gnn, X_hawkes

def train_hawkes(data, save_path):
    logger.info("Training Hawkes Process (Power-Law Kernel)...")
    model = HawkesOrderFlowModel(decay_type="power_law", n_types=2)

    # Mock training loop (Hawkes usually uses MLE on event timestamps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for _seq in data[:100]: # Mini-batch
            optimizer.zero_grad()
            # Dummy forward pass for structure - real Hawkes needs event types + times
            # This is a placeholder for the actual MLE implementation
            loss = torch.tensor(0.5, requires_grad=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}: Loss = {total_loss/100:.4f}")

    torch.save(model.state_dict(), os.path.join(save_path, "hawkes.pt"))
    logger.info("Hawkes model saved.")

def train_transformer(X, y, save_path):
    logger.info("Training Temporal Fusion Transformer...")
    model = OrderFlowTransformer(
        input_dim=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        use_vsn=True # Enable Variable Selection Network
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X) # (Batch, 2)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(save_path, "transformer.pt"))
    logger.info("Transformer model saved.")

def train_gnn(X, save_path):
    logger.info("Training Cross-Asset GNN (Dynamic Graph Learning)...")
    n_assets = X.shape[2]
    model = CrossAssetGNN(num_assets=n_assets, in_channels=8, out_channels=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(5):
        # GNN training usually involves predicting next step or masked nodes
        # Here we just run the forward pass to ensure connectivity
        optimizer.zero_grad()
        # X shape: (Batch, Seq, Nodes, Feats) -> (Batch, Nodes, Feats) for simple GNN
        # Taking last time step for simplicity
        batch_input = X[:, -1, :, :]
        output, adj = model(batch_input)

        loss = output.mean() # Dummy loss
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(save_path, "gnn.pt"))
    logger.info("GNN model saved.")

def main():
    parser = argparse.ArgumentParser(description="CIFT Markets - Model Training Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["data_prep", "train", "eval"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "hawkes", "transformer", "gnn", "xgboost"])
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        # 1. Load Data
        X_trans, y_trans, X_gnn, X_hawkes = generate_synthetic_data()

        # 2. Train Models
        if args.model in ["all", "hawkes"]:
            train_hawkes(X_hawkes, args.output_dir)

        if args.model in ["all", "transformer"]:
            train_transformer(X_trans, y_trans, args.output_dir)

        if args.model in ["all", "gnn"]:
            train_gnn(X_gnn, args.output_dir)

        if args.model in ["all", "xgboost"]:
            logger.info("Training XGBoost Fusion Layer...")
            # XGBoost training logic here (requires sklearn/xgboost installed)
            pass

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
