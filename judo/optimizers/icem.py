# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field

import colorednoise
import numpy as np
from scipy.interpolate import interp1d

from judo.gui import slider
from judo.optimizers.base import Optimizer, OptimizerConfig


@slider("num_nodes", 3, 12, 1)
@slider("num_rollouts", 1, 128, 1)
@slider("init_std", 0.01, 2.0)
@slider("alpha", 0.0, 1.0)
@slider("fraction_elites_reused", 0.0, 1.0)
@slider("noise_beta", 0.0, 3.0)
@dataclass
class ICEMConfig(OptimizerConfig):
    """Configuration for iCEM."""
    num_rollouts: int = 32
    init_std: float = 0.3
    alpha: float = 0.2  # Learning rate for mean/std update
    elites_size: int = 6    
    use_mean_actions: bool = True
    keep_previous_elites: bool = True
    shift_elites_over_time: bool = True
    fraction_elites_reused: float = 0.3
    shift_std_over_time: bool = False
    noise_beta: float = 1.0  # 0 for white noise, 1 for pink noise, 2 for brown noise


class ICEM(Optimizer[ICEMConfig]):
    """Improved Cross-Entropy Method (iCEM) optimizer."""

    def __init__(self, config: ICEMConfig, nu: int) -> None:
        """Initialize iCEM optimizer."""
        super().__init__(config, nu)
        self.std = np.ones((self.num_nodes, self.nu)) * self.config.init_std
        self.elites: np.ndarray | None = None  # Shape: (num_elites, num_nodes, nu)
        self.iter = 0
        self._current_nominal_knots: np.ndarray | None = None

    @property
    def init_std(self) -> float:
        """Get the initial standard deviation."""
        return self.config.init_std

    @property
    def alpha(self) -> float:
        """Get the learning rate."""
        return self.config.alpha

    @property
    def elites_size(self) -> int:
        """Get the number of elites."""
        return self.config.elites_size

    @property
    def noise_beta(self) -> float:
        """Get the noise beta."""
        return self.config.noise_beta

    def pre_optimization(self, old_times: np.ndarray, new_times: np.ndarray) -> None:
        """Update std and elites if the number of nodes has changed or time shifted."""
        self.iter = 0
        
        # Shift std
        if len(self.std) != self.num_nodes:
             self.std = interp1d(
                old_times,
                self.std,
                axis=0,
                fill_value="extrapolate",
                kind="linear",
            )(new_times)
        
        # Reset std to init_std at the beginning of optimization cycle (control step)
        # This matches the reference implementation which resets std at beginning of rollout
        self.std = np.ones((self.num_nodes, self.nu)) * self.init_std
        print("pre_optimization std shape:", self.std.shape)

        # Shift elites
        if self.config.shift_elites_over_time and self.elites is not None:
             self.elites = interp1d(
                old_times,
                self.elites,
                axis=1,
                fill_value="extrapolate",
                kind="linear",
            )(new_times)

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Samples control knots."""
        # Store current nominal knots (mean) for soft update later
        self._current_nominal_knots = nominal_knots.copy()

        num_rollouts = self.num_rollouts
        num_nodes = self.num_nodes
        nu = self.nu
        
        # 1. Sample from distribution (colored noise)
        if self.noise_beta > 0:
            # colorednoise.powerlaw_psd_gaussian generates (samples, channels, length)
            # We want (num_rollouts, nu, num_nodes) then transpose to (num_rollouts, num_nodes, nu)
            samples = colorednoise.powerlaw_psd_gaussian(
                self.noise_beta, size=(num_rollouts, nu, num_nodes)
            ).transpose([0, 2, 1])
        else:
            samples = np.random.randn(num_rollouts, num_nodes, nu)
            
        samples = samples * self.std + nominal_knots
        
        # 2. Add elites
        num_elites_to_inject = 0
        elites_to_inject = []
        
        if self.iter == 0 and self.config.shift_elites_over_time and self.elites is not None:
            # Reuse a fraction of shifted elites
            num_reuse = int(len(self.elites) * self.config.fraction_elites_reused)
            if num_reuse > 0:
                elites_to_inject.append(self.elites[:num_reuse])
                
        if self.iter > 0 and self.config.keep_previous_elites and self.elites is not None:
            # Reuse a fraction of elites from previous iteration
            num_reuse = int(len(self.elites) * self.config.fraction_elites_reused)
            if num_reuse > 0:
                elites_to_inject.append(self.elites[:num_reuse])
                
        if elites_to_inject:
            all_elites = np.concatenate(elites_to_inject, axis=0)
            num_elites_to_inject = len(all_elites)
            # Overwrite the first N samples
            if num_elites_to_inject > num_rollouts:
                num_elites_to_inject = num_rollouts
                all_elites = all_elites[:num_rollouts]
            
            samples[:num_elites_to_inject] = all_elites
        print("sample_control_knots nominal_knots shape:", nominal_knots.shape, "iter:", self.iter)
        # 3. Use mean actions (nominal_knots)
        # Always include the mean in the first slot if possible, overwriting whatever was there
        samples[0] = nominal_knots
        
        return samples

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update nominal knots and internal state."""
        # 1. Select elites
        elite_inds = np.flip(np.argsort(rewards))[: self.elites_size]
        self.elites = sampled_knots[elite_inds]
        
        # 2. Update mean (nominal_knots) and std
        new_mean = self.elites.mean(axis=0)
        new_std = self.elites.std(axis=0)
        
        # Soft update using the stored nominal knots (old mean)
        if self._current_nominal_knots is not None:
            old_mean = self._current_nominal_knots
        else:
            # Fallback if sample_control_knots wasn't called (shouldn't happen)
            old_mean = sampled_knots[0]

        updated_mean = (1 - self.alpha) * new_mean + self.alpha * old_mean
        self.std = (1 - self.alpha) * new_std + self.alpha * self.std
        print("update_nominal_knots updated_mean shape:", updated_mean.shape, "iter:", self.iter)
        self.iter += 1
        return updated_mean
