"""
ADVANCED STATISTICAL ARBITRAGE ENGINE (2025)
============================================

Implements "Holy Grail" features:
1. Copula-Based Dependence Modeling (Non-linear correlation)
2. ML-Based Residual Prediction (XGBoost/GradientBoosting)
3. Hierarchical Risk Parity (HRP) Allocation

"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class CopulaPairsTrading:
    """
    Uses Copulas to model the joint distribution of returns.
    Detects mispricing in the tails where linear correlation fails.
    """
    
    def __init__(self):
        self.copula_params = {}
        
    def fit_copula(self, returns_x: np.array, returns_y: np.array):
        """
        Fits a Gaussian Copula to the returns of two assets.
        """
        # 1. Transform to Uniform [0, 1] using ECDF (Empirical CDF)
        u = stats.rankdata(returns_x) / (len(returns_x) + 1)
        v = stats.rankdata(returns_y) / (len(returns_y) + 1)
        
        # 2. Transform to Gaussian (Inverse Normal CDF)
        x_norm = stats.norm.ppf(u)
        y_norm = stats.norm.ppf(v)
        
        # 3. Calculate Correlation of transformed data
        rho = np.corrcoef(x_norm, y_norm)[0, 1]
        
        self.copula_params = {
            'rho': rho,
            'dist_x': stats.describe(returns_x),
            'dist_y': stats.describe(returns_y)
        }
        return rho

    def get_mispricing_index(self, ret_x, ret_y):
        """
        Calculates conditional probabilities P(U < u | V = v)
        """
        if not self.copula_params:
            return 0.0
            
        # Transform current returns to quantiles (using historical distribution approximation)
        # In production, use the fitted ECDF
        # Here we simplify for demonstration
        u = stats.norm.cdf(ret_x) # Simplified
        v = stats.norm.cdf(ret_y) # Simplified
        
        rho = self.copula_params['rho']
        
        # Conditional Probability for Gaussian Copula
        # P(U <= u | V = v)
        cond_prob = stats.norm.cdf(
            (stats.norm.ppf(u) - rho * stats.norm.ppf(v)) / np.sqrt(1 - rho**2)
        )
        
        # Mispricing Index: How far is the conditional prob from 0.5?
        # If 0.01 or 0.99, it's an extreme event -> Reversion likely
        return cond_prob

class MLResidualFilter:
    """
    Filters Stat Arb signals using Machine Learning.
    Predicts if the spread will ACTUALLY revert.
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        self.scaler = StandardScaler()
        
    def prepare_features(self, spread_series: pd.Series, volume_x, volume_y):
        """
        Feature Engineering for the ML model.
        """
        df = pd.DataFrame({'spread': spread_series})
        
        # Technical Features on the Spread
        df['spread_ma_5'] = df['spread'].rolling(5).mean()
        df['spread_std_5'] = df['spread'].rolling(5).std()
        df['z_score'] = (df['spread'] - df['spread'].rolling(20).mean()) / df['spread'].rolling(20).std()
        
        # Momentum
        df['spread_roc'] = df['spread'].pct_change()
        
        # Volume Imbalance (if available)
        if volume_x is not None and volume_y is not None:
            df['vol_imbalance'] = np.log(volume_x / volume_y)
            
        df.dropna(inplace=True)
        return df
        
    def train(self, spread_series, target_horizon=1):
        """
        Train model to predict spread change.
        """
        features = self.prepare_features(spread_series, None, None)
        
        # Target: Future return of the spread
        features['target'] = features['spread'].shift(-target_horizon) - features['spread']
        features.dropna(inplace=True)
        
        X = features.drop(columns=['target'])
        y = features['target']
        
        self.model.fit(X, y)
        return self.model.score(X, y)
        
    def predict_reversion(self, current_spread_features):
        """
        Returns predicted spread change.
        """
        return self.model.predict(current_spread_features)

class HRPAllocation:
    """
    Hierarchical Risk Parity (HRP) for Portfolio Allocation.
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
            sort_ix = pd.concat([sort_ix, df0]) #.append(df0)
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

    def allocate(self, returns_df):
        """
        Main HRP Allocation function.
        """
        cov = returns_df.cov()
        corr = returns_df.corr()
        
        # 1. Clustering
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist), 'single')
        
        # 2. Quasi-Diagonalization
        sort_ix = self.get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        # 3. Recursive Bisection
        weights = self.get_rec_bipart(cov, sort_ix)
        return weights

