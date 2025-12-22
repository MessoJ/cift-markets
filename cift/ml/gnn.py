"""
CIFT Markets - Graph Neural Network for Cross-Asset Correlation Modeling

Models relationships between assets using graph structure:
- Nodes: Individual assets (stocks, ETFs, futures)
- Edges: Correlation/causality links between assets
- Features: Order flow, price movements, microstructure signals

Key Insights:
- Market impact propagates through correlated assets
- Lead-lag relationships exist between liquid and illiquid assets
- Sector/industry clustering affects price dynamics
- ETF arbitrage creates mechanical relationships

Architecture:
- Graph Attention Network (GAT) for attention-weighted message passing
- Temporal graph convolutions for time-varying relationships
- Edge prediction head for detecting new correlations
- Node regression head for directional prediction

Applications:
- Cross-asset alpha: Trade correlated assets together
- Risk hedging: Identify hedge relationships dynamically
- Market impact: Predict spillover effects
- Pair trading: Dynamic pair selection

References:
- Veličković et al. (2018): "Graph Attention Networks"
- Feng et al. (2019): "Temporal Relational Ranking for Stock Prediction"
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class AssetNode:
    """Single asset in the graph."""

    symbol: str
    node_id: int

    # Node features
    order_flow_imbalance: float
    vpin: float
    realized_vol: float
    return_1m: float
    return_5m: float
    spread: float
    volume_ratio: float
    kyle_lambda: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.order_flow_imbalance,
                self.vpin,
                self.realized_vol,
                self.return_1m,
                self.return_5m,
                self.spread,
                self.volume_ratio,
                self.kyle_lambda,
            ],
            dtype=np.float32,
        )


@dataclass
class CrossAssetPrediction:
    """Prediction output from GNN model."""

    timestamp: float

    # Per-asset predictions
    asset_predictions: dict[str, dict[str, float]]  # symbol -> {direction, magnitude, confidence}

    # Cross-asset signals
    lead_lag_pairs: list[tuple[str, str, float]]  # (leader, lagger, confidence)
    correlation_changes: dict[tuple[str, str], float]  # Correlation delta predictions

    # Portfolio signals
    suggested_hedge_pairs: list[tuple[str, str, float]]  # (long, short, ratio)
    contagion_risk: float  # Risk of cross-asset spillover


# ============================================================================
# GRAPH ATTENTION LAYER
# ============================================================================


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Computes attention-weighted message passing between nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        # Linear transformations for each head
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))

        # Attention parameters
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain("leaky_relu", self.negative_slope)
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.a_src, gain=gain)
        nn.init.xavier_uniform_(self.a_dst, gain=gain)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Updated node features [num_nodes, num_heads * out_features] or [num_nodes, out_features]
        """
        num_nodes = x.shape[0]

        # Transform node features for each head
        # x: [num_nodes, in_features]
        # W: [num_heads, in_features, out_features]
        # h: [num_heads, num_nodes, out_features]
        h = torch.einsum("ni,hio->hno", x, self.W)

        # Compute attention scores
        # Source attention: [num_heads, num_nodes, 1]
        attn_src = torch.einsum("hno,hod->hnd", h, self.a_src)
        attn_dst = torch.einsum("hno,hod->hnd", h, self.a_dst)

        # Get source and target indices
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Compute edge attention: [num_heads, num_edges]
        edge_attn = attn_src[:, src_idx, 0] + attn_dst[:, dst_idx, 0]
        edge_attn = F.leaky_relu(edge_attn, negative_slope=self.negative_slope)

        # Apply edge weights if provided
        if edge_weight is not None:
            edge_attn = edge_attn * edge_weight.unsqueeze(0)

        # Softmax over neighbors (custom sparse softmax)
        # Group edges by destination node
        edge_attn_exp = torch.exp(edge_attn - edge_attn.max())

        # Sum of attention for each destination
        attention_sum = torch.zeros(self.num_heads, num_nodes, device=x.device)
        attention_sum.scatter_add_(
            1, dst_idx.unsqueeze(0).expand(self.num_heads, -1), edge_attn_exp
        )

        # Normalize
        alpha = edge_attn_exp / (attention_sum[:, dst_idx] + 1e-10)
        alpha = self.dropout(alpha)

        # Aggregate messages
        # h: [num_heads, num_nodes, out_features]
        # alpha: [num_heads, num_edges]
        msg = h[:, src_idx] * alpha.unsqueeze(-1)  # [num_heads, num_edges, out_features]

        # Aggregate to destination nodes
        out = torch.zeros(self.num_heads, num_nodes, self.out_features, device=x.device)
        out.scatter_add_(
            1, dst_idx.unsqueeze(0).unsqueeze(-1).expand(self.num_heads, -1, self.out_features), msg
        )

        # Concat or average heads
        if self.concat:
            out = out.transpose(0, 1).reshape(
                num_nodes, -1
            )  # [num_nodes, num_heads * out_features]
        else:
            out = out.mean(dim=0)  # [num_nodes, out_features]

        return out


class TemporalGraphLayer(nn.Module):
    """
    Temporal graph layer for time-varying relationships.

    Combines graph attention with temporal dynamics.
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gat = GraphAttentionLayer(node_features, hidden_dim, num_heads, dropout, concat=True)

        # Temporal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim * num_heads,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for temporal graph.

        Args:
            x_seq: Node features over time [num_nodes, seq_len, features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes, seq_len, _ = x_seq.shape

        # Apply GAT at each timestep
        gat_outputs = []
        for t in range(seq_len):
            h = self.gat(x_seq[:, t, :], edge_index, edge_weight)
            gat_outputs.append(h)

        # Stack for GRU: [num_nodes, seq_len, gat_dim]
        gat_seq = torch.stack(gat_outputs, dim=1)

        # Apply GRU
        _, h_n = self.gru(gat_seq)  # h_n: [1, num_nodes, hidden_dim]

        out = self.norm(h_n.squeeze(0))

        return out


class DynamicGraphLearning(nn.Module):
    """
    Dynamic Graph Learning Module.

    RESEARCH-VALIDATED: Based on Graph WaveNet (arXiv:1906.00121)

    Learns an adaptive adjacency matrix from node embeddings:
        A_adaptive = softmax(ReLU(E1 @ E2.T))

    Key insight: Financial correlations are non-stationary. Rather than
    using static correlation-based graphs, learn edge weights dynamically
    from node features. This allows the model to discover hidden relationships
    and adapt to regime changes.

    Features:
    - Learnable node embeddings E1, E2 of dimension (num_nodes, embed_dim)
    - Combines static graph with learned graph: A = α*A_static + (1-α)*A_learned
    - Supports fully connected or top-k sparsification
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 16,
        static_graph_weight: float = 0.3,  # α in combination formula
        top_k: int | None = None,  # Sparsify to top-k edges per node
        learn_from_features: bool = True,  # Also condition on node features
        feature_dim: int | None = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.static_graph_weight = static_graph_weight
        self.top_k = top_k
        self.learn_from_features = learn_from_features

        # Learnable node embeddings (Graph WaveNet style)
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        # Optional: Feature-based edge weights
        if learn_from_features and feature_dim is not None:
            self.feature_edge_net = nn.Sequential(
                nn.Linear(feature_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.feature_edge_net = None

        logger.info(
            f"DynamicGraphLearning: {num_nodes} nodes, embed_dim={embed_dim}, static_weight={static_graph_weight}"
        )

    def forward(
        self,
        static_adj: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix.

        Args:
            static_adj: Optional static adjacency matrix [num_nodes, num_nodes]
            node_features: Optional node features [num_nodes, feature_dim]

        Returns:
            Adaptive adjacency matrix [num_nodes, num_nodes]
        """
        # Learned adjacency: A = softmax(ReLU(E1 @ E2.T))
        adaptive_adj = F.relu(self.E1 @ self.E2.T)

        # Optional: Incorporate node features
        if self.feature_edge_net is not None and node_features is not None:
            # Compute pairwise feature edges
            # [num_nodes, feature_dim] -> [num_nodes, num_nodes, feature_dim * 2]
            feat_i = node_features.unsqueeze(1).expand(-1, self.num_nodes, -1)
            feat_j = node_features.unsqueeze(0).expand(self.num_nodes, -1, -1)
            feat_pairs = torch.cat([feat_i, feat_j], dim=-1)

            feature_adj = self.feature_edge_net(feat_pairs).squeeze(-1)
            adaptive_adj = adaptive_adj + F.relu(feature_adj)

        # Normalize (softmax over neighbors)
        adaptive_adj = F.softmax(adaptive_adj, dim=-1)

        # Top-k sparsification
        if self.top_k is not None:
            # Keep only top-k edges per node
            values, indices = adaptive_adj.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(adaptive_adj)
            mask.scatter_(-1, indices, 1.0)
            adaptive_adj = adaptive_adj * mask
            # Renormalize
            adaptive_adj = adaptive_adj / (adaptive_adj.sum(dim=-1, keepdim=True) + 1e-8)

        # Combine with static graph
        if static_adj is not None:
            # Normalize static adj
            static_norm = static_adj / (static_adj.sum(dim=-1, keepdim=True) + 1e-8)
            combined_adj = (
                self.static_graph_weight * static_norm
                + (1 - self.static_graph_weight) * adaptive_adj
            )
        else:
            combined_adj = adaptive_adj

        return combined_adj

    def to_edge_index(
        self, adj: torch.Tensor, threshold: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert adjacency matrix to edge_index format.

        Args:
            adj: Adjacency matrix [num_nodes, num_nodes]
            threshold: Minimum weight to include edge

        Returns:
            Tuple of (edge_index [2, num_edges], edge_weight [num_edges])
        """
        # Find non-zero edges above threshold
        mask = adj > threshold
        edge_index = mask.nonzero(as_tuple=False).T  # [2, num_edges]
        edge_weight = adj[mask]  # [num_edges]

        return edge_index, edge_weight


# ============================================================================
# GRAPH NEURAL NETWORK MODEL
# ============================================================================


class CrossAssetGNN(nn.Module):
    """
    Graph Neural Network for Cross-Asset Correlation Modeling.

    Architecture:
    1. Node embedding from asset features
    2. Multi-layer graph attention
    3. Temporal aggregation across snapshots
    4. Prediction heads for various outputs

    RESEARCH-VALIDATED ENHANCEMENT (arXiv:1906.00121):
    - Dynamic Graph Learning: Learns adaptive adjacency matrix
    - Combines static correlation graph with learned relationships
    - Adapts to regime changes and non-stationary correlations

    Outputs:
    - Direction prediction per asset
    - Magnitude prediction per asset
    - Edge correlation prediction
    - Lead-lag relationship detection
    """

    def __init__(
        self,
        node_features: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_temporal: bool = True,
        use_dynamic_graph: bool = True,  # NEW: Enable dynamic graph learning
        num_nodes: int = 50,  # Number of assets for dynamic graph
        dynamic_graph_embed_dim: int = 16,  # Embedding dimension for graph learning
        static_graph_weight: float = 0.3,  # Weight for static vs learned graph
    ):
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_temporal = use_temporal
        self.use_dynamic_graph = use_dynamic_graph

        # Dynamic Graph Learning (Graph WaveNet style)
        if use_dynamic_graph:
            self.graph_learner = DynamicGraphLearning(
                num_nodes=num_nodes,
                embed_dim=dynamic_graph_embed_dim,
                static_graph_weight=static_graph_weight,
                top_k=10,  # Keep top-10 edges per node
                learn_from_features=True,
                feature_dim=hidden_dim,
            )
        else:
            self.graph_learner = None

        # Node feature embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    concat=(i < num_layers - 1),  # Don't concat at last layer
                )
            )

        # Temporal layer
        if use_temporal:
            self.temporal = TemporalGraphLayer(hidden_dim, hidden_dim, num_heads, dropout)

        # Node prediction head (direction + magnitude)
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # direction_prob, magnitude, confidence
        )

        # Edge prediction head (correlation change)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # correlation_delta, confidence
        )

        # Lead-lag detection head
        self.leadlag_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # leader_prob, lagger_prob, neutral_prob
        )

        logger.info(
            f"CrossAssetGNN initialized ({num_layers} layers, {num_heads} heads, dynamic_graph={use_dynamic_graph})"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for single snapshot.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Tuple of:
            - node_pred: [num_nodes, 3]
            - edge_pred: [num_edges, 2]
            - leadlag_pred: [num_edges, 3]
        """
        # Dynamic Graph Learning
        if self.use_dynamic_graph and self.graph_learner is not None:
            # Convert static edge_index to adjacency matrix for combination
            num_nodes = x.shape[0]
            static_adj = torch.zeros(num_nodes, num_nodes, device=x.device)
            if edge_weight is not None:
                static_adj[edge_index[0], edge_index[1]] = edge_weight
            else:
                static_adj[edge_index[0], edge_index[1]] = 1.0

            # Learn dynamic adjacency
            dynamic_adj = self.graph_learner(static_adj, x)

            # Convert back to edge_index/weight for GAT
            # We use a threshold to keep the graph sparse
            edge_index, edge_weight = self.graph_learner.to_edge_index(dynamic_adj, threshold=0.01)

        # Embed nodes
        h = self.node_embed(x)

        # Apply GAT layers
        for gat in self.gat_layers:
            h = gat(h, edge_index, edge_weight)
            h = F.elu(h)

        # Node predictions
        node_pred = self.node_predictor(h)

        # Edge predictions
        # Note: edge_index might have changed, so we predict on the *active* edges
        src_idx, dst_idx = edge_index[0], edge_index[1]
        edge_features = torch.cat([h[src_idx], h[dst_idx]], dim=-1)

        edge_pred = self.edge_predictor(edge_features)
        leadlag_pred = self.leadlag_predictor(edge_features)

        return node_pred, edge_pred, leadlag_pred

    def forward_temporal(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal aggregation.

        Args:
            x_seq: Node features over time [num_nodes, seq_len, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            Same as forward()
        """
        if not self.use_temporal:
            # Use last timestep only
            return self.forward(x_seq[:, -1, :], edge_index, edge_weight)

        num_nodes, seq_len, _ = x_seq.shape

        # Dynamic Graph Learning (using last timestep features for graph structure)
        if self.use_dynamic_graph and self.graph_learner is not None:
            # Use features from last timestep to learn current graph structure
            current_features = x_seq[:, -1, :]

            static_adj = torch.zeros(num_nodes, num_nodes, device=x_seq.device)
            if edge_weight is not None:
                static_adj[edge_index[0], edge_index[1]] = edge_weight
            else:
                static_adj[edge_index[0], edge_index[1]] = 1.0

            dynamic_adj = self.graph_learner(static_adj, current_features)
            edge_index, edge_weight = self.graph_learner.to_edge_index(dynamic_adj, threshold=0.01)

        # Embed all timesteps
        x_embedded = self.node_embed(x_seq.reshape(-1, x_seq.shape[-1]))
        x_embedded = x_embedded.reshape(num_nodes, seq_len, -1)

        # Temporal graph layer
        h = self.temporal(x_embedded, edge_index, edge_weight)

        # Predictions
        node_pred = self.node_predictor(h)

        src_idx, dst_idx = edge_index[0], edge_index[1]
        edge_features = torch.cat([h[src_idx], h[dst_idx]], dim=-1)

        edge_pred = self.edge_predictor(edge_features)
        leadlag_pred = self.leadlag_predictor(edge_features)

        return node_pred, edge_pred, leadlag_pred

    def build_correlation_graph(
        self,
        returns: np.ndarray,
        correlation_threshold: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build graph from return correlations.

        Args:
            returns: Return series [seq_len, num_assets]
            correlation_threshold: Minimum correlation for edge

        Returns:
            Tuple of (edge_index [2, num_edges], edge_weight [num_edges])
        """
        num_assets = returns.shape[1]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(returns.T)

        # Build edge list
        edges_src = []
        edges_dst = []
        weights = []

        for i in range(num_assets):
            for j in range(num_assets):
                if i != j and abs(corr_matrix[i, j]) >= correlation_threshold:
                    edges_src.append(i)
                    edges_dst.append(j)
                    weights.append(abs(corr_matrix[i, j]))

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

        return edge_index, edge_weight

    def predict(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        symbol_map: dict[int, str],
        edge_weight: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> CrossAssetPrediction:
        """
        Predict cross-asset relationships.

        Args:
            node_features: [num_nodes, features] or [num_nodes, seq_len, features]
            edge_index: [2, num_edges]
            symbol_map: node_id -> symbol mapping
            edge_weight: Optional edge weights
            timestamp: Current timestamp

        Returns:
            CrossAssetPrediction
        """
        self.eval()

        with torch.no_grad():
            device = next(self.parameters()).device

            x = torch.tensor(node_features, dtype=torch.float32, device=device)
            ei = torch.tensor(edge_index, dtype=torch.long, device=device)
            ew = None
            if edge_weight is not None:
                ew = torch.tensor(edge_weight, dtype=torch.float32, device=device)

            # Forward pass
            if x.ndim == 3:
                node_pred, edge_pred, leadlag_pred = self.forward_temporal(x, ei, ew)
            else:
                node_pred, edge_pred, leadlag_pred = self.forward(x, ei, ew)

            # Process node predictions
            asset_predictions = {}
            for node_id, symbol in symbol_map.items():
                pred = node_pred[node_id].cpu().numpy()
                direction_prob = torch.sigmoid(torch.tensor(pred[0])).item()
                asset_predictions[symbol] = {
                    "direction": direction_prob,  # >0.5 means up
                    "magnitude": float(pred[1]),
                    "confidence": torch.sigmoid(torch.tensor(pred[2])).item(),
                }

            # Process lead-lag predictions
            lead_lag_pairs = []
            leadlag_probs = F.softmax(leadlag_pred, dim=-1).cpu().numpy()

            for edge_idx in range(ei.shape[1]):
                src = ei[0, edge_idx].item()
                dst = ei[1, edge_idx].item()
                probs = leadlag_probs[edge_idx]

                src_symbol = symbol_map.get(src, f"asset_{src}")
                dst_symbol = symbol_map.get(dst, f"asset_{dst}")

                # Leader probability for source
                if probs[0] > 0.6:  # Source leads
                    lead_lag_pairs.append((src_symbol, dst_symbol, float(probs[0])))

            # Process correlation changes
            correlation_changes = {}
            for edge_idx in range(ei.shape[1]):
                src = ei[0, edge_idx].item()
                dst = ei[1, edge_idx].item()

                src_symbol = symbol_map.get(src, f"asset_{src}")
                dst_symbol = symbol_map.get(dst, f"asset_{dst}")

                corr_delta = edge_pred[edge_idx, 0].item()
                correlation_changes[(src_symbol, dst_symbol)] = corr_delta

            # Compute hedge pairs (opposite direction, high correlation)
            suggested_hedge_pairs = []
            for node_id_i, symbol_i in symbol_map.items():
                for node_id_j, symbol_j in symbol_map.items():
                    if node_id_i >= node_id_j:
                        continue

                    pred_i = asset_predictions[symbol_i]
                    pred_j = asset_predictions[symbol_j]

                    # Opposite directions with high confidence
                    if (pred_i["direction"] > 0.6 and pred_j["direction"] < 0.4) or (
                        pred_i["direction"] < 0.4 and pred_j["direction"] > 0.6
                    ):
                        conf = min(pred_i["confidence"], pred_j["confidence"])
                        if conf > 0.5:
                            # Ratio based on volatility
                            ratio = abs(pred_j["magnitude"]) / (abs(pred_i["magnitude"]) + 1e-8)
                            suggested_hedge_pairs.append((symbol_i, symbol_j, ratio))

            # Contagion risk
            avg_corr = (
                np.mean([abs(v) for v in correlation_changes.values()])
                if correlation_changes
                else 0.5
            )
            contagion_risk = float(avg_corr)

            return CrossAssetPrediction(
                timestamp=timestamp,
                asset_predictions=asset_predictions,
                lead_lag_pairs=lead_lag_pairs[:10],  # Top 10
                correlation_changes=correlation_changes,
                suggested_hedge_pairs=suggested_hedge_pairs[:5],  # Top 5
                contagion_risk=contagion_risk,
            )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class GNNLoss(nn.Module):
    """Combined loss for GNN training."""

    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        correlation_weight: float = 0.3,
        leadlag_weight: float = 0.2,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.correlation_weight = correlation_weight
        self.leadlag_weight = leadlag_weight

    def forward(
        self,
        node_pred: torch.Tensor,
        edge_pred: torch.Tensor,
        leadlag_pred: torch.Tensor,
        node_targets: torch.Tensor,
        edge_targets: torch.Tensor,
        leadlag_targets: torch.Tensor,
    ) -> torch.Tensor:
        # Direction loss (binary cross-entropy)
        direction_loss = F.binary_cross_entropy_with_logits(node_pred[:, 0], node_targets[:, 0])

        # Magnitude loss (MSE)
        magnitude_loss = F.mse_loss(node_pred[:, 1], node_targets[:, 1])

        # Correlation loss
        correlation_loss = F.mse_loss(edge_pred[:, 0], edge_targets[:, 0])

        # Lead-lag loss (cross-entropy)
        leadlag_loss = F.cross_entropy(leadlag_pred, leadlag_targets)

        total_loss = (
            self.direction_weight * direction_loss
            + self.magnitude_weight * magnitude_loss
            + self.correlation_weight * correlation_loss
            + self.leadlag_weight * leadlag_loss
        )

        return total_loss


class GNNTrainer:
    """Training loop for GNN model."""

    def __init__(
        self,
        model: CrossAssetGNN,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = GNNLoss()

    def train_step(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray | None,
        node_targets: np.ndarray,
        edge_targets: np.ndarray,
        leadlag_targets: np.ndarray,
    ) -> float:
        """Single training step."""
        self.model.train()

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        ei_t = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        ew_t = None
        if edge_weight is not None:
            ew_t = torch.tensor(edge_weight, dtype=torch.float32, device=self.device)

        nt_t = torch.tensor(node_targets, dtype=torch.float32, device=self.device)
        et_t = torch.tensor(edge_targets, dtype=torch.float32, device=self.device)
        ll_t = torch.tensor(leadlag_targets, dtype=torch.long, device=self.device)

        if x_t.ndim == 3:
            node_pred, edge_pred, leadlag_pred = self.model.forward_temporal(x_t, ei_t, ew_t)
        else:
            node_pred, edge_pred, leadlag_pred = self.model.forward(x_t, ei_t, ew_t)

        loss = self.loss_fn(node_pred, edge_pred, leadlag_pred, nt_t, et_t, ll_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


__all__ = [
    "CrossAssetGNN",
    "CrossAssetPrediction",
    "AssetNode",
    "GraphAttentionLayer",
    "TemporalGraphLayer",
    "GNNTrainer",
    "GNNLoss",
]
