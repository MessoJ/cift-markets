"""
Hierarchical Risk Parity (HRP) Portfolio Construction

Implementation of Lopez de Prado's HRP algorithm:
1. Tree clustering based on correlation structure
2. Quasi-diagonalization (ordering assets by cluster)
3. Recursive bisection for weight allocation

Benefits over Mean-Variance:
- No matrix inversion (avoids instability)
- Out-of-sample robustness
- Works with singular covariance matrices
- More stable weights over time

Reference:
- Lopez de Prado (2016), "Building Diversified Portfolios that Outperform Out-of-Sample"
- De Prado (2018), "Advances in Financial Machine Learning", Chapter 16
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform


# =============================================================================
# CORRELATION DISTANCE METRICS
# =============================================================================

def correlation_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.
    
    Distance = sqrt(0.5 * (1 - correlation))
    
    This satisfies metric properties:
    - d(i,i) = 0
    - d(i,j) = d(j,i)
    - Triangle inequality
    
    Args:
        corr_matrix: Correlation matrix
        
    Returns:
        Distance matrix
    """
    # Ensure valid correlations
    corr = np.clip(corr_matrix, -1, 1)
    
    # Distance formula
    dist = np.sqrt(0.5 * (1 - corr))
    
    # Ensure diagonal is 0
    np.fill_diagonal(dist, 0)
    
    return dist


def angular_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Alternative: Angular distance.
    
    Distance = arccos(correlation) / pi
    
    More sensitive to differences at high correlations.
    """
    corr = np.clip(corr_matrix, -1, 1)
    dist = np.arccos(corr) / np.pi
    np.fill_diagonal(dist, 0)
    return dist


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def compute_hrp_linkage(
    corr_matrix: np.ndarray,
    method: str = "single"
) -> np.ndarray:
    """
    Compute hierarchical clustering linkage matrix.
    
    Args:
        corr_matrix: Correlation matrix
        method: Linkage method ('single', 'complete', 'average', 'ward')
                'single' is recommended for HRP
        
    Returns:
        Linkage matrix (scipy format)
    """
    # Convert to distance
    dist_matrix = correlation_distance(corr_matrix)
    
    # Condensed distance matrix for scipy
    condensed = squareform(dist_matrix, checks=False)
    
    # Hierarchical clustering
    link = linkage(condensed, method=method)
    
    return link


def get_quasi_diagonal_order(link: np.ndarray) -> List[int]:
    """
    Get asset ordering that makes covariance matrix quasi-diagonal.
    
    This ordering groups correlated assets together,
    making the covariance matrix block-diagonal-like.
    
    Args:
        link: Linkage matrix from hierarchical clustering
        
    Returns:
        Ordered list of asset indices
    """
    return list(leaves_list(link))


# =============================================================================
# RECURSIVE BISECTION
# =============================================================================

def get_cluster_variance(
    cov_matrix: np.ndarray,
    indices: List[int]
) -> float:
    """
    Calculate cluster variance using inverse-variance weighting.
    
    Within a cluster, allocate inversely proportional to variance.
    Cluster variance = 1 / sum(1/var_i)
    
    Args:
        cov_matrix: Covariance matrix
        indices: Asset indices in cluster
        
    Returns:
        Cluster variance
    """
    # Individual variances
    variances = np.array([cov_matrix[i, i] for i in indices])
    
    # Avoid division by zero
    variances = np.maximum(variances, 1e-10)
    
    # Inverse variance weights within cluster
    inv_var = 1 / variances
    cluster_var = 1 / np.sum(inv_var)
    
    return cluster_var


def recursive_bisection(
    cov_matrix: np.ndarray,
    sorted_indices: List[int]
) -> np.ndarray:
    """
    Recursively bisect clusters to allocate weights.
    
    At each split:
    1. Divide ordered assets into two clusters
    2. Allocate weight inversely proportional to cluster variance
    3. Recurse into each cluster
    
    Args:
        cov_matrix: Covariance matrix
        sorted_indices: Quasi-diagonal ordered asset indices
        
    Returns:
        Array of portfolio weights
    """
    n = len(sorted_indices)
    weights = np.ones(n)
    
    # Clusters to process: (start_idx, end_idx, weight_allocation)
    clusters = [(0, n, 1.0)]
    
    while clusters:
        start, end, allocation = clusters.pop(0)
        
        if end - start == 1:
            # Single asset
            weights[start] = allocation
            continue
        
        # Split in half
        mid = (start + end) // 2
        
        left_indices = sorted_indices[start:mid]
        right_indices = sorted_indices[mid:end]
        
        # Cluster variances
        left_var = get_cluster_variance(cov_matrix, left_indices)
        right_var = get_cluster_variance(cov_matrix, right_indices)
        
        # Allocate inversely proportional to variance
        total_inv_var = 1/left_var + 1/right_var
        left_weight = (1/left_var) / total_inv_var
        right_weight = (1/right_var) / total_inv_var
        
        # Scale by parent allocation
        left_alloc = allocation * left_weight
        right_alloc = allocation * right_weight
        
        # Add child clusters
        clusters.append((start, mid, left_alloc))
        clusters.append((mid, end, right_alloc))
    
    return weights


# =============================================================================
# MAIN HRP ALGORITHM
# =============================================================================

def compute_hrp_weights(
    returns: np.ndarray,
    asset_names: Optional[List[str]] = None,
    linkage_method: str = "single"
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute HRP portfolio weights.
    
    Steps:
    1. Compute correlation and covariance matrices
    2. Hierarchical clustering on correlations
    3. Quasi-diagonalization (reorder assets)
    4. Recursive bisection (allocate weights)
    
    Args:
        returns: Asset returns matrix (T x N)
        asset_names: Optional asset names for labeling
        linkage_method: Clustering method
        
    Returns:
        (weights, sorted_indices)
    """
    # Ensure 2D
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    
    n_assets = returns.shape[1]
    
    if n_assets == 1:
        return np.array([1.0]), [0]
    
    # Compute correlation and covariance
    cov_matrix = np.cov(returns, rowvar=False)
    
    # Handle scalar covariance (single asset)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
    
    std = np.sqrt(np.diag(cov_matrix))
    std[std == 0] = 1e-10  # Avoid division by zero
    
    corr_matrix = cov_matrix / np.outer(std, std)
    np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal is 1
    
    # Step 1: Hierarchical clustering
    link = compute_hrp_linkage(corr_matrix, method=linkage_method)
    
    # Step 2: Quasi-diagonalization
    sorted_indices = get_quasi_diagonal_order(link)
    
    # Step 3: Recursive bisection
    # Reorder covariance matrix
    sorted_cov = cov_matrix[np.ix_(sorted_indices, sorted_indices)]
    
    # Get weights in sorted order
    sorted_weights = recursive_bisection(sorted_cov, list(range(n_assets)))
    
    # Map back to original order
    weights = np.zeros(n_assets)
    for i, sorted_idx in enumerate(sorted_indices):
        weights[sorted_idx] = sorted_weights[i]
    
    # Normalize (should already sum to 1, but ensure)
    weights = weights / weights.sum()
    
    return weights, sorted_indices


def compute_hrp_weights_from_cov(
    cov_matrix: np.ndarray,
    linkage_method: str = "single"
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute HRP weights directly from covariance matrix.
    
    Useful when you already have the covariance estimate.
    
    Args:
        cov_matrix: Covariance matrix
        linkage_method: Clustering method
        
    Returns:
        (weights, sorted_indices)
    """
    n = cov_matrix.shape[0]
    
    if n == 1:
        return np.array([1.0]), [0]
    
    # Convert to correlation
    std = np.sqrt(np.diag(cov_matrix))
    std[std == 0] = 1e-10
    corr_matrix = cov_matrix / np.outer(std, std)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Clustering
    link = compute_hrp_linkage(corr_matrix, method=linkage_method)
    sorted_indices = get_quasi_diagonal_order(link)
    
    # Recursive bisection
    sorted_cov = cov_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_weights = recursive_bisection(sorted_cov, list(range(n)))
    
    # Map back
    weights = np.zeros(n)
    for i, sorted_idx in enumerate(sorted_indices):
        weights[sorted_idx] = sorted_weights[i]
    
    weights = weights / weights.sum()
    
    return weights, sorted_indices


# =============================================================================
# EXTENSIONS
# =============================================================================

def hrp_with_constraints(
    returns: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_volatility: Optional[float] = None
) -> np.ndarray:
    """
    HRP with weight constraints and volatility targeting.
    
    Args:
        returns: Asset returns (T x N)
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        target_volatility: Target portfolio volatility (optional)
        
    Returns:
        Constrained weights
    """
    # Base HRP weights
    weights, _ = compute_hrp_weights(returns)
    
    # Apply bounds
    weights = np.clip(weights, min_weight, max_weight)
    
    # Renormalize
    weights = weights / weights.sum()
    
    # Volatility scaling
    if target_volatility is not None:
        cov = np.cov(returns, rowvar=False)
        portfolio_vol = np.sqrt(weights.T @ cov @ weights)
        
        if portfolio_vol > 0:
            vol_scalar = target_volatility / portfolio_vol
            # Note: This changes leverage, not weights
            weights = weights * vol_scalar
    
    return weights


def rolling_hrp_weights(
    returns: np.ndarray,
    window: int = 252,
    rebalance_freq: int = 21
) -> np.ndarray:
    """
    Compute rolling HRP weights over time.
    
    Args:
        returns: Full return history (T x N)
        window: Rolling window for estimation
        rebalance_freq: How often to rebalance (in bars)
        
    Returns:
        Time series of weights (T x N)
    """
    T, N = returns.shape
    all_weights = np.zeros((T, N))
    
    current_weights = np.ones(N) / N  # Start equal weight
    
    for t in range(T):
        if t < window:
            # Not enough data, use equal weight
            all_weights[t] = np.ones(N) / N
        elif t % rebalance_freq == 0:
            # Rebalance
            window_returns = returns[t - window:t]
            current_weights, _ = compute_hrp_weights(window_returns)
            all_weights[t] = current_weights
        else:
            # Hold previous weights
            all_weights[t] = current_weights
    
    return all_weights


# =============================================================================
# NESTED CLUSTERING OPTIMIZATION (NCO)
# =============================================================================

def nco_weights(
    returns: np.ndarray,
    n_clusters: Optional[int] = None,
    intra_method: str = "hrp",
    inter_method: str = "hrp"
) -> np.ndarray:
    """
    Nested Clustering Optimization (NCO).
    
    Two-level HRP:
    1. Cluster assets
    2. Optimize within each cluster
    3. Optimize across clusters
    
    This provides better diversification for large portfolios.
    
    Args:
        returns: Asset returns (T x N)
        n_clusters: Number of clusters (default: sqrt(N))
        intra_method: Method for intra-cluster optimization
        inter_method: Method for inter-cluster optimization
        
    Returns:
        NCO weights
    """
    T, N = returns.shape
    
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N)))
    
    # Correlation matrix
    corr = np.corrcoef(returns, rowvar=False)
    
    # Cluster using HRP linkage
    link = compute_hrp_linkage(corr)
    
    # Cut tree to get clusters
    from scipy.cluster.hierarchy import fcluster
    cluster_labels = fcluster(link, n_clusters, criterion='maxclust')
    
    # Intra-cluster weights
    cluster_weights = {}
    cluster_returns = {}
    
    for c in range(1, n_clusters + 1):
        mask = cluster_labels == c
        cluster_idx = np.where(mask)[0]
        
        if len(cluster_idx) == 1:
            # Single asset cluster
            cluster_weights[c] = np.array([1.0])
            cluster_returns[c] = returns[:, cluster_idx]
        else:
            # HRP within cluster
            cluster_ret = returns[:, cluster_idx]
            w, _ = compute_hrp_weights(cluster_ret)
            cluster_weights[c] = w
            cluster_returns[c] = cluster_ret @ w.reshape(-1, 1)
    
    # Inter-cluster weights (treat each cluster as an asset)
    cluster_ret_matrix = np.hstack([cluster_returns[c] for c in range(1, n_clusters + 1)])
    inter_weights, _ = compute_hrp_weights(cluster_ret_matrix)
    
    # Combine to get final weights
    final_weights = np.zeros(N)
    for c in range(1, n_clusters + 1):
        mask = cluster_labels == c
        cluster_idx = np.where(mask)[0]
        final_weights[cluster_idx] = inter_weights[c - 1] * cluster_weights[c]
    
    # Normalize
    final_weights = final_weights / final_weights.sum()
    
    return final_weights


# =============================================================================
# RISK CONTRIBUTIONS
# =============================================================================

def marginal_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate marginal risk contribution of each asset.
    
    MRC_i = d(portfolio_vol) / d(w_i)
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        Marginal risk contributions
    """
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    
    if port_vol < 1e-10:
        return np.zeros_like(weights)
    
    mrc = (cov_matrix @ weights) / port_vol
    
    return mrc


def risk_contribution_pct(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate percentage risk contribution of each asset.
    
    RC_i = w_i * MRC_i / portfolio_vol
    
    This shows what fraction of total risk each asset contributes.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        Risk contributions (sum to 1)
    """
    mrc = marginal_risk_contributions(weights, cov_matrix)
    
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    
    if port_vol < 1e-10:
        return np.ones_like(weights) / len(weights)
    
    rc = weights * mrc / port_vol
    
    # Normalize to ensure sum to 1
    rc = rc / rc.sum()
    
    return rc


def effective_number_of_bets(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Calculate Effective Number of Bets (ENB).
    
    ENB = exp(-sum(rc_i * log(rc_i)))
    
    Measures portfolio diversification.
    ENB = N means perfect diversification.
    ENB = 1 means concentrated in single bet.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        ENB value
    """
    rc = risk_contribution_pct(weights, cov_matrix)
    
    # Avoid log(0)
    rc = np.maximum(rc, 1e-10)
    
    # Entropy
    entropy = -np.sum(rc * np.log(rc))
    
    # ENB
    enb = np.exp(entropy)
    
    return enb
