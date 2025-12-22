"""
CIFT Markets - Hidden Markov Model for Market Regime Detection

Detects market regimes (states) from observable market data.

Regimes:
1. Low Volatility / Range-bound: Tight spreads, low volume, mean-reverting
2. Trending: Directional moves, increasing volume, momentum
3. High Volatility / Choppy: Wide spreads, high volume, unpredictable
4. Crisis / Dislocation: Extreme moves, liquidity gaps

The HMM models:
- Hidden states (regimes) that aren't directly observable
- Emission probabilities (how each regime produces observations)
- Transition probabilities (how regimes switch)

Applications:
- Strategy selection (momentum in trending, mean-reversion in ranging)
- Risk management (reduce size in high-vol regimes)
- Model weight adjustment (trust different models per regime)

Implementation:
- PyTorch for GPU acceleration
- Baum-Welch algorithm for training
- Viterbi algorithm for regime inference
- Online updates for real-time adaptation

References:
- Rabiner (1989): "A Tutorial on Hidden Markov Models"
- Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"
"""

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# ============================================================================
# DATA STRUCTURES
# ============================================================================


class MarketRegime(IntEnum):
    """Market regime enumeration."""

    LOW_VOLATILITY = 0  # Quiet, range-bound
    TRENDING_UP = 1  # Bullish momentum
    TRENDING_DOWN = 2  # Bearish momentum
    HIGH_VOLATILITY = 3  # Choppy, high uncertainty
    CRISIS = 4  # Extreme dislocation

    @property
    def description(self) -> str:
        descriptions = {
            0: "Low volatility, range-bound market",
            1: "Upward trending with momentum",
            2: "Downward trending with momentum",
            3: "High volatility, choppy conditions",
            4: "Crisis/dislocation, extreme conditions",
        }
        return descriptions[self.value]


@dataclass
class RegimePrediction:
    """Prediction output from HMM model."""

    timestamp: float

    # Current regime
    current_regime: MarketRegime
    regime_probability: float  # Confidence in current regime

    # All regime probabilities
    regime_probs: dict[MarketRegime, float]

    # Regime metrics
    persistence: float  # Expected duration of current regime
    transition_prob: float  # P(regime change in next window)

    # Derived signals
    volatility_regime: str  # "low", "normal", "high", "extreme"
    trend_regime: str  # "none", "up", "down"

    # Risk adjustment
    suggested_position_scale: float  # 0.0 to 1.5


@dataclass
class RegimeFeatures:
    """Features for regime detection."""

    # Volatility features
    realized_vol_1m: float
    realized_vol_5m: float
    realized_vol_30m: float
    vol_of_vol: float  # Volatility of volatility

    # Spread features
    spread_mean: float
    spread_std: float
    spread_percentile: float

    # Volume features
    volume_ratio: float  # Current / average
    volume_imbalance: float

    # Return features
    return_1m: float
    return_5m: float
    return_30m: float
    return_autocorr: float  # Return autocorrelation

    # Microstructure
    order_flow_imbalance: float
    vpin: float
    kyle_lambda: float

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.realized_vol_1m,
                self.realized_vol_5m,
                self.realized_vol_30m,
                self.vol_of_vol,
                self.spread_mean,
                self.spread_std,
                self.spread_percentile,
                self.volume_ratio,
                self.volume_imbalance,
                self.return_1m,
                self.return_5m,
                self.return_30m,
                self.return_autocorr,
                self.order_flow_imbalance,
                self.vpin,
                self.kyle_lambda,
            ],
            dtype=np.float32,
        )


# ============================================================================
# EMISSION DISTRIBUTIONS
# ============================================================================


class GaussianEmission(nn.Module):
    """
    Gaussian emission distribution for HMM.

    Each hidden state has a multivariate Gaussian emission distribution.
    P(observation | state) = N(observation; μ_state, Σ_state)
    """

    def __init__(self, num_states: int, observation_dim: int, diagonal_cov: bool = True):
        super().__init__()

        self.num_states = num_states
        self.observation_dim = observation_dim
        self.diagonal_cov = diagonal_cov

        # Mean for each state: [num_states, observation_dim]
        self.means = nn.Parameter(torch.randn(num_states, observation_dim) * 0.1)

        if diagonal_cov:
            # Log variance for numerical stability
            self.log_vars = nn.Parameter(torch.zeros(num_states, observation_dim))
        else:
            # Full covariance (Cholesky factor)
            self.cholesky = nn.Parameter(
                torch.eye(observation_dim).unsqueeze(0).expand(num_states, -1, -1)
            )

    def log_prob(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute log emission probability.

        Args:
            observations: [batch, seq_len, observation_dim]

        Returns:
            Log probabilities [batch, seq_len, num_states]
        """
        batch_size, seq_len, _ = observations.shape

        # Expand observations for broadcasting
        obs = observations.unsqueeze(-2)  # [batch, seq_len, 1, obs_dim]

        if self.diagonal_cov:
            # Diagonal covariance case
            means = self.means.unsqueeze(0).unsqueeze(0)  # [1, 1, num_states, obs_dim]
            vars = torch.exp(self.log_vars).unsqueeze(0).unsqueeze(0)  # [1, 1, num_states, obs_dim]

            # Log probability of multivariate Gaussian with diagonal cov
            diff = obs - means
            log_prob = -0.5 * (
                self.observation_dim * math.log(2 * math.pi)
                + self.log_vars.sum(dim=-1)  # log|Σ|
                + (diff**2 / vars).sum(dim=-1)  # Mahalanobis
            )
        else:
            # Full covariance case
            log_prob = torch.zeros(batch_size, seq_len, self.num_states, device=observations.device)

            for s in range(self.num_states):
                L = self.cholesky[s]
                cov = L @ L.T + 1e-6 * torch.eye(self.observation_dim, device=L.device)

                dist = torch.distributions.MultivariateNormal(self.means[s], covariance_matrix=cov)

                log_prob[:, :, s] = dist.log_prob(observations)

        return log_prob


class MixtureOfGaussiansEmission(nn.Module):
    """
    Mixture of Gaussians emission for more complex distributions.

    Useful when observations within a regime are multi-modal.
    """

    def __init__(
        self,
        num_states: int,
        observation_dim: int,
        num_components: int = 3,
    ):
        super().__init__()

        self.num_states = num_states
        self.num_components = num_components

        # Mixture weights per state (logits)
        self.mixture_logits = nn.Parameter(torch.zeros(num_states, num_components))

        # Component means: [num_states, num_components, observation_dim]
        self.means = nn.Parameter(torch.randn(num_states, num_components, observation_dim) * 0.1)

        # Component log variances
        self.log_vars = nn.Parameter(torch.zeros(num_states, num_components, observation_dim))

    def log_prob(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute log emission probability."""
        batch_size, seq_len, obs_dim = observations.shape

        # [batch, seq_len, 1, 1, obs_dim]
        obs = observations.unsqueeze(-2).unsqueeze(-2)

        # [1, 1, num_states, num_components, obs_dim]
        means = self.means.unsqueeze(0).unsqueeze(0)
        vars = torch.exp(self.log_vars).unsqueeze(0).unsqueeze(0)

        # Gaussian log prob for each component
        diff = obs - means
        component_log_prob = -0.5 * (
            obs_dim * math.log(2 * math.pi)
            + self.log_vars.sum(dim=-1)
            + (diff**2 / vars).sum(dim=-1)
        )  # [batch, seq_len, num_states, num_components]

        # Mixture weights
        log_weights = F.log_softmax(self.mixture_logits, dim=-1)  # [num_states, num_components]
        log_weights = log_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, num_states, num_components]

        # Log-sum-exp over components
        log_prob = torch.logsumexp(log_weights + component_log_prob, dim=-1)

        return log_prob


# ============================================================================
# HIDDEN MARKOV MODEL
# ============================================================================


class MarketRegimeHMM(nn.Module):
    """
    Hidden Markov Model for Market Regime Detection.

    Architecture:
    - 5 hidden states (market regimes)
    - Gaussian or GMM emissions
    - Feature-conditioned transition probabilities (IO-HMM)
    - Online Viterbi for real-time inference
    - Baum-Welch for batch training

    RESEARCH-VALIDATED: Upgraded to Input-Output HMM (IO-HMM)
    Based on: Nystrup et al. (2017) "Dynamic Asset Allocation with Hidden Markov Models"

    Key insight: Transition probabilities should depend on macro features (VIX, volume, etc.)
    rather than being static. This allows the model to react faster to regime changes
    during macro shocks.

    Features:
    - Regime persistence modeling
    - Transition probability estimation
    - Confidence calibration
    """

    def __init__(
        self,
        num_states: int = 5,
        observation_dim: int = 16,
        emission_type: str = "gaussian",  # "gaussian" or "gmm"
        num_gmm_components: int = 3,
        min_state_duration: int = 5,  # Minimum timesteps in a state
        use_io_hmm: bool = True,  # NEW: Enable Input-Output HMM
        transition_feature_dim: int = 4,  # Features for transition conditioning
    ):
        super().__init__()

        self.num_states = num_states
        self.observation_dim = observation_dim
        self.min_state_duration = min_state_duration
        self.use_io_hmm = use_io_hmm

        # Initial state probabilities (log)
        self.log_initial = nn.Parameter(torch.zeros(num_states))

        # Transition probabilities
        if use_io_hmm:
            # IO-HMM: Transition matrix is a function of features
            # P(z_t = j | z_{t-1} = i, x_t) = softmax(W_i @ x_t + b_i)
            self.transition_nets = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(transition_feature_dim, num_states * 2),
                        nn.Tanh(),
                        nn.Linear(num_states * 2, num_states),
                    )
                    for _ in range(num_states)  # One network per source state
                ]
            )

            # Bias towards persistence (diagonal dominance)
            self.transition_bias = nn.Parameter(
                torch.eye(num_states) * 2.0 + torch.ones(num_states, num_states) * 0.1
            )

            logger.info(
                f"IO-HMM enabled: transitions conditioned on {transition_feature_dim} features"
            )
        else:
            # Standard HMM: Fixed transition matrix
            init_trans = torch.eye(num_states) * 2.0  # Favor staying
            init_trans += torch.ones(num_states, num_states) * 0.1  # Small transition prob
            self.log_transition = nn.Parameter(
                torch.log(init_trans / init_trans.sum(dim=1, keepdim=True))
            )

        # Emission distribution
        if emission_type == "gmm":
            self.emission = MixtureOfGaussiansEmission(
                num_states, observation_dim, num_gmm_components
            )
        else:
            self.emission = GaussianEmission(num_states, observation_dim)

        # State duration modeling (semi-Markov extension)
        self.log_duration_params = nn.Parameter(torch.ones(num_states) * math.log(10.0))

        # Position scaling per regime
        self.position_scales = nn.Parameter(
            torch.tensor(
                [
                    1.0,  # Low vol: normal
                    1.2,  # Trend up: slightly higher
                    1.2,  # Trend down: slightly higher
                    0.5,  # High vol: reduce
                    0.1,  # Crisis: minimal
                ]
            )
        )

        # For online inference
        self._alpha = None  # Forward probabilities
        self._current_state = None
        self._state_duration = 0

        logger.info(
            f"MarketRegimeHMM initialized ({num_states} states, {emission_type} emissions, IO-HMM={use_io_hmm})"
        )

    def get_transition_matrix(
        self,
        transition_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get transition probability matrix.

        For IO-HMM, this is conditioned on features.
        For standard HMM, this is the fixed learned matrix.

        Args:
            transition_features: [batch, transition_feature_dim] or None

        Returns:
            Transition matrix [num_states, num_states] or [batch, num_states, num_states]
        """
        if not self.use_io_hmm or transition_features is None:
            # Standard HMM
            if hasattr(self, "log_transition"):
                return F.softmax(self.log_transition, dim=1)
            else:
                # Return biased uniform if IO-HMM but no features provided
                return F.softmax(self.transition_bias, dim=1)

        # IO-HMM: Compute feature-conditioned transitions
        transition_features.shape[0]

        # Compute logits for each source state
        logits = []
        for i, net in enumerate(self.transition_nets):
            logit = net(transition_features)  # [batch, num_states]
            logit = logit + self.transition_bias[i].unsqueeze(0)  # Add persistence bias
            logits.append(logit)

        # Stack: [batch, num_states (from), num_states (to)]
        logits = torch.stack(logits, dim=1)

        # Softmax over destination states
        return F.softmax(logits, dim=-1)

    @property
    def initial_probs(self) -> torch.Tensor:
        """Normalized initial state probabilities."""
        return F.softmax(self.log_initial, dim=0)

    @property
    def transition_matrix(self) -> torch.Tensor:
        """Normalized transition probability matrix (for backward compat)."""
        return self.get_transition_matrix(None)

    def forward_algorithm(
        self,
        observations: torch.Tensor,
        mask: torch.Tensor | None = None,
        transition_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward algorithm for computing log-likelihood and forward probabilities.

        UPGRADED: Supports IO-HMM with feature-conditioned transitions.

        Args:
            observations: [batch, seq_len, observation_dim]
            mask: Optional mask [batch, seq_len]
            transition_features: Optional features for IO-HMM [batch, seq_len, transition_feature_dim]

        Returns:
            Tuple of:
            - log_likelihood: [batch]
            - alpha: Forward probabilities [batch, seq_len, num_states]
        """
        batch_size, seq_len, _ = observations.shape

        # Log emission probabilities
        log_emission = self.emission.log_prob(observations)  # [batch, seq_len, num_states]

        # Initialize alpha with initial probabilities
        log_initial = F.log_softmax(self.log_initial, dim=0)
        alpha = log_initial + log_emission[:, 0, :]  # [batch, num_states]

        all_alpha = [alpha]

        # Forward pass
        for t in range(1, seq_len):
            # Get transition matrix (feature-conditioned for IO-HMM)
            if self.use_io_hmm and transition_features is not None:
                # Get features at time t for each batch
                trans_feat_t = transition_features[:, t, :]  # [batch, transition_feature_dim]
                trans_probs = self.get_transition_matrix(
                    trans_feat_t
                )  # [batch, num_states, num_states]
                log_trans = torch.log(trans_probs + 1e-10)

                # alpha[t] = emission[t] * sum_s(alpha[t-1, s] * trans[s, current])
                # [batch, num_states, 1] + [batch, num_states, num_states]
                alpha_trans = alpha.unsqueeze(-1) + log_trans
            else:
                # Standard HMM: fixed transition matrix
                if hasattr(self, "log_transition"):
                    log_trans = F.log_softmax(
                        self.log_transition, dim=1
                    )  # [num_states, num_states]
                else:
                    log_trans = F.log_softmax(self.transition_bias, dim=1)
                alpha_trans = alpha.unsqueeze(-1) + log_trans.unsqueeze(0)

            # Logsumexp over previous states
            alpha = torch.logsumexp(alpha_trans, dim=1) + log_emission[:, t, :]

            if mask is not None:
                # Keep previous alpha where masked
                alpha = torch.where(mask[:, t].unsqueeze(-1), alpha, all_alpha[-1])

            all_alpha.append(alpha)

        # Stack all alphas
        all_alpha = torch.stack(all_alpha, dim=1)  # [batch, seq_len, num_states]

        # Log-likelihood is logsumexp of final alpha
        log_likelihood = torch.logsumexp(alpha, dim=-1)  # [batch]

        return log_likelihood, all_alpha

    def viterbi_algorithm(
        self,
        observations: torch.Tensor,
        transition_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Viterbi algorithm for most likely state sequence.

        UPGRADED: Supports IO-HMM with feature-conditioned transitions.

        Args:
            observations: [batch, seq_len, observation_dim]
            transition_features: Optional features for IO-HMM [batch, seq_len, transition_feature_dim]

        Returns:
            Tuple of:
            - best_path: [batch, seq_len] state indices
            - path_prob: [batch] log probability of best path
        """
        batch_size, seq_len, _ = observations.shape

        # Log emission probabilities
        log_emission = self.emission.log_prob(observations)

        # Initialize
        log_initial = F.log_softmax(self.log_initial, dim=0)
        delta = log_initial + log_emission[:, 0, :]  # [batch, num_states]

        # Store backpointers
        psi = []

        # Viterbi forward pass
        for t in range(1, seq_len):
            # Get transition matrix (feature-conditioned for IO-HMM)
            if self.use_io_hmm and transition_features is not None:
                trans_feat_t = transition_features[:, t, :]
                trans_probs = self.get_transition_matrix(trans_feat_t)
                log_trans = torch.log(trans_probs + 1e-10)  # [batch, num_states, num_states]

                # delta[t, s] = max_s'(delta[t-1, s'] + log_trans[s', s]) + log_emission[t, s]
                delta_trans = delta.unsqueeze(-1) + log_trans  # [batch, num_states, num_states]
            else:
                # Standard HMM
                if hasattr(self, "log_transition"):
                    log_trans = F.log_softmax(self.log_transition, dim=1)
                else:
                    log_trans = F.log_softmax(self.transition_bias, dim=1)
                delta_trans = delta.unsqueeze(-1) + log_trans.unsqueeze(0)

            max_delta, argmax_delta = delta_trans.max(dim=1)  # [batch, num_states]
            delta = max_delta + log_emission[:, t, :]

            psi.append(argmax_delta)

        # Backtrack
        best_path = []
        _, best_last = delta.max(dim=1)  # [batch]
        best_path.append(best_last)

        for t in range(seq_len - 2, -1, -1):
            best_prev = psi[t].gather(1, best_path[-1].unsqueeze(-1)).squeeze(-1)
            best_path.append(best_prev)

        best_path = torch.stack(best_path[::-1], dim=1)  # [batch, seq_len]
        path_prob = delta.max(dim=1)[0]  # [batch]

        return best_path, path_prob

    def baum_welch_step(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single Baum-Welch (EM) training step.

        Args:
            observations: [batch, seq_len, observation_dim]

        Returns:
            Negative log-likelihood (loss)
        """
        log_likelihood, _ = self.forward_algorithm(observations)
        return -log_likelihood.mean()

    def predict(
        self,
        features: np.ndarray,
        timestamp: float = 0.0,
    ) -> RegimePrediction:
        """
        Predict current market regime.

        Args:
            features: Observation features [observation_dim] or [seq, observation_dim]
            timestamp: Current timestamp

        Returns:
            RegimePrediction with regime and probabilities
        """
        self.eval()

        with torch.no_grad():
            device = next(self.parameters()).device

            # Handle single observation or sequence
            if features.ndim == 1:
                features = features.reshape(1, 1, -1)
            elif features.ndim == 2:
                features = features.reshape(1, *features.shape)

            obs = torch.tensor(features, dtype=torch.float32, device=device)

            # Get Viterbi path
            best_path, path_prob = self.viterbi_algorithm(obs)
            current_state = best_path[0, -1].item()

            # Get forward probabilities for state distribution
            _, alpha = self.forward_algorithm(obs)
            state_probs = F.softmax(alpha[0, -1], dim=0).cpu().numpy()

            # Create regime probabilities dict
            regime_probs = {MarketRegime(i): float(state_probs[i]) for i in range(self.num_states)}

            # Get transition probabilities
            trans_matrix = self.transition_matrix.cpu().numpy()
            persistence = trans_matrix[current_state, current_state]
            transition_prob = 1.0 - persistence

            # Determine volatility and trend regime descriptions
            if current_state in [0]:
                vol_regime = "low"
                trend_regime = "none"
            elif current_state == 1:
                vol_regime = "normal"
                trend_regime = "up"
            elif current_state == 2:
                vol_regime = "normal"
                trend_regime = "down"
            elif current_state == 3:
                vol_regime = "high"
                trend_regime = "none"
            else:
                vol_regime = "extreme"
                trend_regime = "none"

            # Position scale
            position_scale = float(self.position_scales[current_state].item())
            position_scale = max(0.0, min(1.5, position_scale))

            return RegimePrediction(
                timestamp=timestamp,
                current_regime=MarketRegime(current_state),
                regime_probability=float(state_probs[current_state]),
                regime_probs=regime_probs,
                persistence=float(persistence),
                transition_prob=float(transition_prob),
                volatility_regime=vol_regime,
                trend_regime=trend_regime,
                suggested_position_scale=position_scale,
            )

    def online_update(
        self,
        observation: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Online (streaming) state inference.

        Updates internal state based on new observation.

        Args:
            observation: New observation [observation_dim]

        Returns:
            Tuple of (current_state, state_probabilities)
        """
        device = next(self.parameters()).device

        obs = torch.tensor(observation.reshape(1, 1, -1), dtype=torch.float32, device=device)

        # Log emission for new observation
        log_emission = self.emission.log_prob(obs).squeeze()  # [num_states]

        # Log transition
        log_trans = F.log_softmax(self.log_transition, dim=1)

        if self._alpha is None:
            # Initialize with initial probabilities
            log_initial = F.log_softmax(self.log_initial, dim=0)
            self._alpha = log_initial + log_emission
        else:
            # Update alpha
            alpha_trans = self._alpha.unsqueeze(-1) + log_trans
            self._alpha = torch.logsumexp(alpha_trans, dim=0) + log_emission

        # Normalize to get probabilities
        state_probs = F.softmax(self._alpha, dim=0).cpu().numpy()
        current_state = int(state_probs.argmax())

        # Track state duration
        if self._current_state == current_state:
            self._state_duration += 1
        else:
            self._current_state = current_state
            self._state_duration = 1

        return current_state, state_probs

    def reset_online_state(self):
        """Reset online inference state."""
        self._alpha = None
        self._current_state = None
        self._state_duration = 0

    def get_regime_statistics(
        self,
        observations: np.ndarray,
    ) -> dict[str, any]:
        """
        Compute regime statistics over a sequence.

        Args:
            observations: [seq_len, observation_dim]

        Returns:
            Dictionary with regime statistics
        """
        device = next(self.parameters()).device
        obs = torch.tensor(
            observations.reshape(1, *observations.shape), dtype=torch.float32, device=device
        )

        # Get state sequence
        best_path, _ = self.viterbi_algorithm(obs)
        states = best_path[0].cpu().numpy()

        # Compute statistics
        stats = {
            "state_sequence": states.tolist(),
            "regime_counts": {},
            "regime_durations": {},
            "transitions": 0,
        }

        for regime in MarketRegime:
            count = (states == regime.value).sum()
            stats["regime_counts"][regime.name] = int(count)

        # Count transitions and durations
        current_regime = states[0]
        current_duration = 1
        durations = {r.value: [] for r in MarketRegime}

        for s in states[1:]:
            if s == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                stats["transitions"] += 1
                current_regime = s
                current_duration = 1

        durations[current_regime].append(current_duration)

        for regime in MarketRegime:
            d_list = durations[regime.value]
            stats["regime_durations"][regime.name] = {
                "mean": float(np.mean(d_list)) if d_list else 0,
                "max": int(max(d_list)) if d_list else 0,
                "count": len(d_list),
            }

        return stats


# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class HMMTrainer:
    """Training loop for HMM model."""

    def __init__(
        self,
        model: MarketRegimeHMM,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(
        self,
        observations: np.ndarray,
    ) -> float:
        """
        Single training step using Baum-Welch.

        Args:
            observations: [batch, seq_len, observation_dim]

        Returns:
            Loss value
        """
        self.model.train()

        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)

        loss = self.model.baum_welch_step(obs)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        seq_len: int = 100,
    ) -> float:
        """
        Train for one epoch.

        Args:
            data: Training data [total_len, observation_dim]
            batch_size: Batch size
            seq_len: Sequence length per sample

        Returns:
            Average loss
        """
        total_len = data.shape[0]
        num_samples = (total_len - seq_len) // batch_size

        total_loss = 0.0

        for _i in range(num_samples):
            # Sample random starting points
            starts = np.random.randint(0, total_len - seq_len, size=batch_size)
            batch = np.stack([data[s : s + seq_len] for s in starts])

            loss = self.train_step(batch)
            total_loss += loss

        return total_loss / max(num_samples, 1)


__all__ = [
    "MarketRegimeHMM",
    "MarketRegime",
    "RegimePrediction",
    "RegimeFeatures",
    "HMMTrainer",
    "GaussianEmission",
    "MixtureOfGaussiansEmission",
]
