"""
ADVANCED QUANTITATIVE METHODS
=============================

Implementation of Institutional-Grade Algorithms:
1. Copula-Based Pairs Trading (Non-Linear Dependence)
2. ML-Enhanced Residual Filtering (XGBoost)
3. Hierarchical Risk Parity (HRP) Allocation

References:
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
- Xie, W., et al. (2016). Pairs Trading with Copulas.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging
from typing import List

logger = logging.getLogger(__name__)

# =============================================================================
# 1. COPULA PAIRS TRADING
# =============================================================================

class CopulaPairs:
    """
    Models the joint distribution of two assets using a Gaussian Copula.
    Returns a 'Mispricing Index' (conditional probability) instead of Z-score.
    """
    def __init__(self):
        self.u1 = None # CDF of asset 1
        self.u2 = None # CDF of asset 2
        self.rho = None # Correlation of transformed variables
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit the copula to historical returns"""
        # Transform to uniform [0, 1] using empirical CDF
        self.u1 = stats.rankdata(x) / (len(x) + 1)
        self.u2 = stats.rankdata(y) / (len(y) + 1)
        
        # Transform to Gaussian
        z1 = stats.norm.ppf(self.u1)
        z2 = stats.norm.ppf(self.u2)
        
        # Calculate correlation
        self.rho = np.corrcoef(z1, z2)[0, 1]
        
    def get_mispricing_index(self, x_val: float, y_val: float, history_x: np.ndarray, history_y: np.ndarray) -> float:
        """
        Calculate conditional probability P(U1 <= u1 | U2 = u2)
        This is the 'Mispricing Index'.
        0.5 = Fair Value
        < 0.05 = Undervalued (Long)
        > 0.95 = Overvalued (Short)
        """
        # Map current values to quantiles based on history
        u1 = (history_x < x_val).mean()
        u2 = (history_y < y_val).mean()
        
        # Avoid 0/1 for infinity issues
        u1 = np.clip(u1, 0.001, 0.999)
        u2 = np.clip(u2, 0.001, 0.999)
        
        # Conditional probability formula for Gaussian Copula
        # P(U1 <= u1 | U2 = u2) = N( (N^-1(u1) - rho * N^-1(u2)) / sqrt(1-rho^2) )
        
        z1 = stats.norm.ppf(u1)
        z2 = stats.norm.ppf(u2)
        
        numerator = z1 - self.rho * z2
        denominator = np.sqrt(1 - self.rho**2)
        
        mi = stats.norm.cdf(numerator / denominator)
        return mi

# =============================================================================
# 2. ML RESIDUAL FILTER
# =============================================================================

class MLSignalFilter:
    """
    Uses Gradient Boosting to predict if a mean reversion signal will succeed.
    Filters out 'falling knives'.
    """
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, spread: np.ndarray) -> np.ndarray:
        """
        Create features from spread history:
        - Z-score
        - Momentum (ROC)
        - Volatility
        - Distance from MA
        """
        df = pd.DataFrame({'spread': spread})
        
        df['zscore'] = (df['spread'] - df['spread'].rolling(20).mean()) / df['spread'].rolling(20).std()
        df['mom_5'] = df['spread'].pct_change(5)
        df['vol_20'] = df['spread'].rolling(20).std()
        df['dist_ma'] = df['spread'] - df['spread'].rolling(50).mean()
        
        # Drop NaNs
        return df.dropna()
        
    def train(self, spreads: List[np.ndarray], outcomes: List[int]):
        """
        Train the model.
        spreads: List of spread arrays (history before trade)
        outcomes: 1 (profit) or 0 (loss)
        """
        X = []
        y = []
        
        for spread, outcome in zip(spreads, outcomes):
            feats = self.extract_features(spread)
            if not feats.empty:
                X.append(feats.iloc[-1].values) # Use last row features
                y.append(outcome)
                
        if len(X) > 50: # Min samples
            X = np.array(X)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
    def predict_success(self, spread: np.ndarray) -> float:
        """Returns probability of trade success"""
        if not self.is_trained:
            return 0.5 # Neutral if not trained
            
        feats = self.extract_features(spread)
        if feats.empty:
            return 0.5
            
        X = feats.iloc[-1].values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1] # Prob of class 1
        return prob

# =============================================================================
# 3. HIERARCHICAL RISK PARITY (HRP)
# =============================================================================

class HRPAllocation:
    """
    Allocates capital using Hierarchical Risk Parity.
    Robust to noise and correlation structure.
    """
    def get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]) # Use concat instead of append
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = self.get_cluster_var(cov, c_items0)
                c_var1 = self.get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    def get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]
        w_ = self.get_ivp(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    def get_ivp(self, cov):
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate HRP weights from historical returns.
        """
        # 1. Correlation and Covariance
        corr = returns.corr().fillna(0)
        cov = returns.cov().fillna(0)
        
        # 2. Clustering (Distance Matrix)
        dist = np.sqrt((1 - corr) / 2)
        # Handle floating point errors
        dist = dist.clip(0, 1)
        
        # Linkage
        # Squareform is needed for linkage
        # Fill diagonal with 0
        np.fill_diagonal(dist.values, 0)
        link = linkage(squareform(dist), 'single')
        
        # 3. Quasi-Diagonalization
        sort_ix = self.get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        # 4. Recursive Bisection
        weights = self.get_rec_bipart(cov, sort_ix)
        weights.index = sort_ix
        
        return weights
