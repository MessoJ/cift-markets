"""
CIFT Markets - Hawkes Process Model for Order Flow Prediction

A self-exciting point process model for tick-level order flow dynamics.

Theory:
The Hawkes process models events that cluster in time, where each event
increases the probability of future events (self-excitation). This is
ideal for modeling order flow because:
- Orders beget orders (momentum, herding)
- Market makers respond to order flow
- Algorithms react to detected patterns

Model:
λ(t) = μ + Σ α * exp(-β * (t - t_i))

Where:
- λ(t): Intensity (expected orders per unit time)
- μ: Base intensity (background order rate)
- α: Jump size (excitation from each event)
- β: Decay rate (how fast excitation fades)
- t_i: Past event times

Implementation:
- PyTorch-based for GPU acceleration
- Multi-dimensional (separate buy/sell processes)
- Mutually-exciting (buys can trigger sells and vice versa)
- Online learning for real-time adaptation

References:
- Hawkes (1971): "Spectra of some self-exciting and mutually exciting point processes"
- Bacry et al. (2015): "Hawkes Processes in Finance"
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HawkesEvent:
    """Single event in the point process."""
    timestamp: float     # Time in seconds
    event_type: int      # 0: buy, 1: sell, 2: cancel
    size: float = 1.0    # Event magnitude (e.g., order size)
    price: float = 0.0   # Price at event


@dataclass
class HawkesPrediction:
    """Prediction output from Hawkes model."""
    timestamp: float

    # Intensities for next interval
    buy_intensity: float      # Expected buy orders per second
    sell_intensity: float     # Expected sell orders per second

    # Probabilities
    buy_prob: float           # P(next event is buy)
    sell_prob: float          # P(next event is sell)

    # Derived signals
    flow_imbalance: float     # (buy - sell) / (buy + sell)
    urgency: float            # Total intensity (higher = more activity)

    # Confidence
    confidence: float         # Model confidence


# ============================================================================
# HAWKES INTENSITY KERNEL
# ============================================================================

class ExponentialKernel(nn.Module):
    """
    Exponential decay kernel for Hawkes process.

    φ(t) = α * exp(-β * t)

    Learnable parameters:
    - α (alpha): Jump size / excitation magnitude
    - β (beta): Decay rate
    """

    def __init__(self, num_event_types: int = 3, init_alpha: float = 0.5, init_beta: float = 1.0):
        super().__init__()

        # Cross-excitation matrix: [from_type, to_type]
        # Initialized to encourage some cross-excitation
        self.log_alpha = nn.Parameter(
            torch.log(torch.ones(num_event_types, num_event_types) * init_alpha)
        )

        # Decay rates (must be positive)
        self.log_beta = nn.Parameter(
            torch.log(torch.ones(num_event_types, num_event_types) * init_beta)
        )

    @property
    def alpha(self) -> torch.Tensor:
        """Get excitation matrix (positive)."""
        return torch.exp(self.log_alpha)

    @property
    def beta(self) -> torch.Tensor:
        """Get decay matrix (positive)."""
        return torch.exp(self.log_beta)

    def forward(
        self,
        dt: torch.Tensor,        # Time differences: [batch, num_events]
        event_types: torch.Tensor # Event types: [batch, num_events]
    ) -> torch.Tensor:
        """
        Compute kernel values for time differences.

        Args:
            dt: Time since each past event [batch, num_events]
            event_types: Type of each past event [batch, num_events]

        Returns:
            Kernel values [batch, num_events, num_event_types]
        """
        batch_size, num_events = dt.shape
        self.alpha.shape[0]

        # Get alpha and beta for each event type
        # [num_events] -> [batch, num_events, num_types]
        alpha_vals = self.alpha[event_types]  # [batch, num_events, num_types]
        beta_vals = self.beta[event_types]    # [batch, num_events, num_types]

        # Compute kernel: α * exp(-β * dt)
        dt_expanded = dt.unsqueeze(-1)  # [batch, num_events, 1]
        kernel = alpha_vals * torch.exp(-beta_vals * dt_expanded)

        return kernel


class SumOfExponentialsKernel(nn.Module):
    """
    Sum of exponentials kernel for multi-scale dynamics.

    φ(t) = Σ_k α_k * exp(-β_k * t)

    Captures both fast (HFT) and slow (institutional) timescales.
    """

    def __init__(
        self,
        num_event_types: int = 3,
        num_components: int = 3,
        init_betas: list[float] = None,  # Fast, medium, slow
    ):
        if init_betas is None:
            init_betas = [10.0, 1.0, 0.1]
        super().__init__()

        self.num_components = num_components

        # Alphas for each component and type pair
        self.log_alphas = nn.Parameter(
            torch.log(torch.ones(num_components, num_event_types, num_event_types) * 0.3)
        )

        # Fixed decay rates at different timescales
        betas = torch.tensor(init_betas).view(-1, 1, 1)
        betas = betas.expand(num_components, num_event_types, num_event_types)
        self.register_buffer("betas", betas)

    @property
    def alphas(self) -> torch.Tensor:
        return torch.exp(self.log_alphas)

    def forward(
        self,
        dt: torch.Tensor,
        event_types: torch.Tensor
    ) -> torch.Tensor:
        """Compute sum of exponentials kernel."""
        batch_size, num_events = dt.shape
        num_types = self.alphas.shape[1]

        dt.unsqueeze(-1).unsqueeze(0)  # [1, batch, num_events, 1]

        # Sum over components
        total_kernel = torch.zeros(batch_size, num_events, num_types, device=dt.device)

        for k in range(self.num_components):
            alpha_k = self.alphas[k][event_types]  # [batch, num_events, num_types]
            beta_k = self.betas[k][event_types]

            kernel_k = alpha_k * torch.exp(-beta_k * dt.unsqueeze(-1))
            total_kernel = total_kernel + kernel_k

        return total_kernel


class PowerLawApproxKernel(nn.Module):
    """
    Power-Law kernel approximated via Sum of Exponentials.

    RESEARCH-VALIDATED: Based on arXiv:1302.1405 (Hardiman et al., 2013)
    "Critical reflexivity in financial markets: a Hawkes process analysis"

    Key finding: Markets exhibit power-law decay with exponent ~-1.15 for
    short timescales (<1000s) and ~-1.45 for longer timescales.

    φ(t) ≈ Σ_k α_k * exp(-β_k * t)  where β_k are geometrically spaced

    This approximation gives O(N) complexity vs O(N²) for true power-law,
    while preserving long-memory characteristics critical for financial markets.
    """

    def __init__(
        self,
        num_event_types: int = 3,
        num_components: int = 8,       # More components = better approximation
        min_beta: float = 0.001,       # Slowest decay (longest memory ~1000s)
        max_beta: float = 100.0,       # Fastest decay (~10ms)
        power_law_exponent: float = 1.15,  # From Hardiman et al.
        learnable_exponent: bool = True,
    ):
        super().__init__()

        self.num_components = num_components
        self.num_event_types = num_event_types

        # Geometrically spaced decay rates (log-uniform from min to max)
        log_betas = torch.linspace(
            math.log(min_beta), math.log(max_beta), num_components
        )
        betas = torch.exp(log_betas)  # [num_components]

        # Expand for cross-excitation: [components, from_type, to_type]
        betas = betas.view(-1, 1, 1).expand(
            num_components, num_event_types, num_event_types
        ).clone()
        self.register_buffer("betas", betas)

        # Power-law exponent (learnable to adapt to market conditions)
        if learnable_exponent:
            self.log_exponent = nn.Parameter(torch.tensor(math.log(power_law_exponent)))
        else:
            self.register_buffer("log_exponent", torch.tensor(math.log(power_law_exponent)))

        # Alpha weights derived from power-law: α_k ∝ β_k^(1-γ)
        # where γ is the power-law exponent
        # These are learnable scaling factors on top of the power-law structure
        self.log_alpha_scale = nn.Parameter(
            torch.zeros(num_event_types, num_event_types)
        )

        # Criticality parameter: should be close to 1 for financial markets
        # (branching ratio / spectral radius of kernel integral)
        self.log_criticality = nn.Parameter(torch.tensor(math.log(0.9)))

        logger.info(
            f"PowerLawApproxKernel: {num_components} components, "
            f"β ∈ [{min_beta:.4f}, {max_beta:.1f}], γ₀={power_law_exponent:.2f}"
        )

    @property
    def exponent(self) -> torch.Tensor:
        """Power-law exponent (γ), typically 1.0-1.5 for financial markets."""
        # Constrain to reasonable range [0.5, 2.5]
        return 0.5 + 2.0 * torch.sigmoid(self.log_exponent)

    @property
    def criticality(self) -> torch.Tensor:
        """Branching ratio, should be <1 for stability, close to 1 for criticality."""
        return torch.sigmoid(self.log_criticality)  # Constrain to (0, 1)

    @property
    def alphas(self) -> torch.Tensor:
        """
        Compute α_k values following power-law structure.

        For power-law kernel φ(t) = c / (1+t)^γ, the approximation uses:
        α_k ∝ β_k^(1-γ) * (1 - exp(-β_k * Δt))

        where Δt is the characteristic time between components.
        """
        gamma = self.exponent

        # Base alpha from power-law: α_k ∝ β_k^(1-γ)
        # Shape: [num_components, 1, 1]
        base_alpha = torch.pow(self.betas[:, 0, 0], 1 - gamma).view(-1, 1, 1)

        # Expand to [num_components, num_types, num_types]
        base_alpha = base_alpha.expand(
            self.num_components, self.num_event_types, self.num_event_types
        )

        # Apply learnable scaling per type pair
        scale = torch.exp(self.log_alpha_scale).unsqueeze(0)  # [1, types, types]

        # Normalize to achieve target criticality
        # Spectral radius ≈ sum of alphas / sum of betas
        alpha_unnorm = base_alpha * scale
        alpha_sum = alpha_unnorm.sum(dim=0).max()  # Approximate spectral radius
        beta_sum = self.betas.sum(dim=0).min()

        target_crit = self.criticality
        normalization = target_crit * beta_sum / (alpha_sum + 1e-8)

        return alpha_unnorm * normalization

    def forward(
        self,
        dt: torch.Tensor,           # [batch, num_events]
        event_types: torch.Tensor   # [batch, num_events]
    ) -> torch.Tensor:
        """
        Compute power-law approximated kernel values.

        Returns:
            Kernel values [batch, num_events, num_event_types]
        """
        batch_size, num_events = dt.shape
        device = dt.device

        # Get current alpha values
        alphas = self.alphas  # [num_components, num_types, num_types]

        # Initialize output
        total_kernel = torch.zeros(
            batch_size, num_events, self.num_event_types, device=device
        )

        # Sum over exponential components
        dt_expanded = dt.unsqueeze(-1)  # [batch, num_events, 1]

        for k in range(self.num_components):
            # Get alpha and beta for this component and event types
            alpha_k = alphas[k][event_types]  # [batch, num_events, num_types]
            beta_k = self.betas[k][event_types]

            # Exponential kernel: α * exp(-β * t)
            kernel_k = alpha_k * torch.exp(-beta_k * dt_expanded)

            total_kernel = total_kernel + kernel_k

        return total_kernel

    def compute_intensity_increment(
        self,
        dt: torch.Tensor,
        event_types: torch.Tensor,
        prev_intensity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Efficient O(1) intensity update using exponential recursion.

        For exponential kernels, we can update intensity recursively:
        λ(t) = μ + Σ_k R_k(t)
        R_k(t) = exp(-β_k * Δt) * R_k(t_prev) + α_k  (at event time)
        R_k(t) = exp(-β_k * Δt) * R_k(t_prev)        (between events)

        This gives O(1) per event instead of O(N).
        """
        # This would be used in a streaming/online setting
        # For batch processing, the standard forward() is used
        pass

    def get_diagnostics(self) -> dict[str, float]:
        """Get kernel diagnostics for monitoring."""
        return {
            "power_law_exponent": self.exponent.item(),
            "criticality": self.criticality.item(),
            "alpha_sum": self.alphas.sum().item(),
            "beta_range": f"[{self.betas.min().item():.4f}, {self.betas.max().item():.1f}]",
        }


# ============================================================================
# HAWKES PROCESS MODEL
# ============================================================================

class HawkesOrderFlowModel(nn.Module):
    """
    Neural Hawkes Process for Order Flow Prediction.

    Architecture:
    - Multi-dimensional Hawkes process (buy/sell/cancel)
    - Mutually-exciting (cross-excitation between types)
    - Sum-of-exponentials kernel (multi-timescale)
    - Feature-modulated base intensity
    - Online learning capability

    Predicts:
    - Short-term order flow intensity
    - Buy/sell imbalance
    - Event probabilities
    """

    def __init__(
        self,
        num_event_types: int = 3,      # buy, sell, cancel
        feature_dim: int = 20,          # Order book features
        hidden_dim: int = 64,
        kernel_type: str = "sum_exp",   # "exp" or "sum_exp"
        num_kernel_components: int = 3,
        max_history: int = 500,         # Max events to consider
        prediction_horizon_ms: float = 500.0,  # Prediction window
    ):
        super().__init__()

        self.num_event_types = num_event_types
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_history = max_history
        self.prediction_horizon = prediction_horizon_ms / 1000.0  # Convert to seconds

        # Base intensity (μ) - modulated by features
        self.base_intensity_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_event_types),
            nn.Softplus(),  # Ensure positive intensity
        )

        # Excitation kernel
        if kernel_type == "sum_exp":
            self.kernel = SumOfExponentialsKernel(
                num_event_types=num_event_types,
                num_components=num_kernel_components,
            )
        elif kernel_type == "power_law":
            # NEW: Use research-validated power-law approximation
            self.kernel = PowerLawApproxKernel(
                num_event_types=num_event_types,
                num_components=8,  # Higher precision for power law
            )
        else:
            self.kernel = ExponentialKernel(num_event_types=num_event_types)

        # Size impact (larger orders have more impact)
        self.size_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(num_event_types + feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Event history buffer
        self._event_times: list[float] = []
        self._event_types: list[int] = []
        self._event_sizes: list[float] = []

        logger.info(f"HawkesOrderFlowModel initialized (types={num_event_types}, features={feature_dim})")

    def forward(
        self,
        features: torch.Tensor,         # [batch, feature_dim]
        event_times: torch.Tensor,      # [batch, num_events] - relative times
        event_types: torch.Tensor,      # [batch, num_events] - event types
        event_sizes: torch.Tensor | None = None,  # [batch, num_events]
        current_time: torch.Tensor | None = None, # [batch]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute intensity and predict order flow.

        Args:
            features: Order book features
            event_times: Timestamps of past events (seconds, relative)
            event_types: Type indices of past events
            event_sizes: Sizes of past events
            current_time: Current timestamp

        Returns:
            Tuple of (intensities, confidence)
            - intensities: [batch, num_event_types]
            - confidence: [batch, 1]
        """
        batch_size = features.shape[0]
        device = features.device

        # Base intensity from features
        mu = self.base_intensity_net(features)  # [batch, num_types]

        # Compute self-excitation from history
        if event_times.shape[1] > 0:
            # Time differences from current time
            if current_time is None:
                current_time = event_times[:, -1] + 0.001

            dt = current_time.unsqueeze(-1) - event_times  # [batch, num_events]
            dt = torch.clamp(dt, min=1e-6)  # Ensure positive

            # Kernel values
            kernel_vals = self.kernel(dt, event_types)  # [batch, num_events, num_types]

            # Apply size weighting if provided
            if event_sizes is not None:
                size_weights = self.size_embedding(event_sizes.unsqueeze(-1))  # [batch, num_events, 1]
                kernel_vals = kernel_vals * size_weights

            # Sum excitation from all past events
            excitation = kernel_vals.sum(dim=1)  # [batch, num_types]
        else:
            excitation = torch.zeros(batch_size, self.num_event_types, device=device)

        # Total intensity
        intensity = mu + excitation

        # Confidence estimation
        conf_input = torch.cat([intensity, features], dim=-1)
        confidence = self.confidence_net(conf_input)

        return intensity, confidence

    def predict(
        self,
        features: np.ndarray,
        current_time: float,
    ) -> HawkesPrediction:
        """
        Generate prediction from current state.

        Args:
            features: Order book features [feature_dim]
            current_time: Current timestamp in seconds

        Returns:
            HawkesPrediction with intensities and probabilities
        """
        self.eval()

        with torch.no_grad():
            # Prepare inputs
            device = next(self.parameters()).device

            features_t = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

            # Get event history
            if self._event_times:
                event_times = torch.tensor(
                    self._event_times[-self.max_history:],
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                event_types = torch.tensor(
                    self._event_types[-self.max_history:],
                    dtype=torch.long, device=device
                ).unsqueeze(0)
                event_sizes = torch.tensor(
                    self._event_sizes[-self.max_history:],
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
            else:
                event_times = torch.zeros(1, 0, device=device)
                event_types = torch.zeros(1, 0, dtype=torch.long, device=device)
                event_sizes = None

            current_t = torch.tensor([current_time], dtype=torch.float32, device=device)

            # Forward pass
            intensity, confidence = self.forward(
                features_t, event_times, event_types, event_sizes, current_t
            )

            # Extract values
            intensity = intensity.squeeze(0).cpu().numpy()
            confidence = confidence.item()

            buy_intensity = float(intensity[0])
            sell_intensity = float(intensity[1])
            total_intensity = buy_intensity + sell_intensity + 1e-8

            # Compute probabilities using softmax-like normalization
            buy_prob = buy_intensity / total_intensity
            sell_prob = sell_intensity / total_intensity

            # Flow imbalance
            flow_imbalance = (buy_intensity - sell_intensity) / total_intensity

            return HawkesPrediction(
                timestamp=current_time,
                buy_intensity=buy_intensity,
                sell_intensity=sell_intensity,
                buy_prob=buy_prob,
                sell_prob=sell_prob,
                flow_imbalance=flow_imbalance,
                urgency=total_intensity,
                confidence=confidence,
            )

    def add_event(self, event: HawkesEvent):
        """
        Add observed event to history.

        Args:
            event: Observed order event
        """
        self._event_times.append(event.timestamp)
        self._event_types.append(event.event_type)
        self._event_sizes.append(event.size)

        # Trim history
        if len(self._event_times) > self.max_history * 2:
            self._event_times = self._event_times[-self.max_history:]
            self._event_types = self._event_types[-self.max_history:]
            self._event_sizes = self._event_sizes[-self.max_history:]

    def clear_history(self):
        """Clear event history."""
        self._event_times.clear()
        self._event_types.clear()
        self._event_sizes.clear()

    def compute_log_likelihood(
        self,
        features: torch.Tensor,
        event_times: torch.Tensor,
        event_types: torch.Tensor,
        event_sizes: torch.Tensor | None = None,
        T: float | None = None,
    ) -> torch.Tensor:
        """
        Compute log-likelihood for training.

        The Hawkes log-likelihood is:
        LL = Σ log(λ(t_i)) - ∫_0^T λ(t) dt

        For exponential kernel, the integral has closed form.
        """
        batch_size, num_events = event_times.shape
        device = event_times.device

        if num_events == 0:
            return torch.zeros(batch_size, device=device)

        if T is None:
            T = event_times[:, -1].max().item() + 1.0

        # Compute intensity at each event time
        log_intensities = []

        for i in range(num_events):
            # Features at event i (simplified: use same features)
            feat = features

            # Past events
            past_times = event_times[:, :i] if i > 0 else torch.zeros(batch_size, 0, device=device)
            past_types = event_types[:, :i] if i > 0 else torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            past_sizes = event_sizes[:, :i] if event_sizes is not None and i > 0 else None

            current_t = event_times[:, i]

            intensity, _ = self.forward(feat, past_times, past_types, past_sizes, current_t)

            # Get intensity for actual event type
            event_type = event_types[:, i]
            intensity_at_event = intensity.gather(1, event_type.unsqueeze(1)).squeeze(1)

            log_intensities.append(torch.log(intensity_at_event + 1e-8))

        # Sum of log intensities
        log_intensity_sum = torch.stack(log_intensities, dim=1).sum(dim=1)

        # Compensator (integral) - approximate with numerical integration
        num_steps = 100
        times = torch.linspace(0, T, num_steps, device=device).unsqueeze(0).expand(batch_size, -1)
        dt = T / num_steps

        integral = torch.zeros(batch_size, device=device)

        for step in range(num_steps):
            t = times[:, step]

            # Events before this time
            mask = event_times <= t.unsqueeze(-1)
            past_times = event_times * mask
            past_types = event_types * mask.long()
            past_sizes = event_sizes * mask if event_sizes is not None else None

            intensity, _ = self.forward(features, past_times, past_types, past_sizes, t)

            integral = integral + intensity.sum(dim=1) * dt

        # Log-likelihood = sum(log λ) - integral
        log_likelihood = log_intensity_sum - integral

        return log_likelihood


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class HawkesTrainer:
    """Training loop for Hawkes model."""

    def __init__(
        self,
        model: HawkesOrderFlowModel,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

    def train_step(
        self,
        features: np.ndarray,
        event_times: np.ndarray,
        event_types: np.ndarray,
        event_sizes: np.ndarray | None = None,
    ) -> float:
        """Single training step."""
        self.model.train()

        # Convert to tensors
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
        event_times_t = torch.tensor(event_times, dtype=torch.float32, device=self.device)
        event_types_t = torch.tensor(event_types, dtype=torch.long, device=self.device)
        event_sizes_t = torch.tensor(event_sizes, dtype=torch.float32, device=self.device) if event_sizes is not None else None

        # Compute negative log-likelihood
        log_likelihood = self.model.compute_log_likelihood(
            features_t, event_times_t, event_types_t, event_sizes_t
        )

        loss = -log_likelihood.mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def online_update(
        self,
        features: np.ndarray,
        event: HawkesEvent,
        learning_rate: float = 1e-4,
    ):
        """
        Online learning update after observing new event.

        Uses a smaller learning rate for stability.
        """
        self.model.train()

        # Add event to history
        self.model.add_event(event)

        # Quick update on recent history
        if len(self.model._event_times) < 10:
            return

        # Get recent window
        recent_times = np.array(self.model._event_times[-50:])
        recent_types = np.array(self.model._event_types[-50:])
        recent_sizes = np.array(self.model._event_sizes[-50:])

        # Relative times
        recent_times = recent_times - recent_times[0]

        # Single update step
        features_t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        times_t = torch.tensor(recent_times, dtype=torch.float32, device=self.device).unsqueeze(0)
        types_t = torch.tensor(recent_types, dtype=torch.long, device=self.device).unsqueeze(0)
        sizes_t = torch.tensor(recent_sizes, dtype=torch.float32, device=self.device).unsqueeze(0)

        log_likelihood = self.model.compute_log_likelihood(
            features_t, times_t, types_t, sizes_t
        )

        loss = -log_likelihood.mean()

        # Small update
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()


__all__ = [
    "HawkesOrderFlowModel",
    "HawkesEvent",
    "HawkesPrediction",
    "HawkesTrainer",
    "ExponentialKernel",
    "SumOfExponentialsKernel",
]
