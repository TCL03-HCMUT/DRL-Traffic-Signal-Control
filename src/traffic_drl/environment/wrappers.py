"""Gymnasium wrappers for per-episode scenario selection and metric collection.

Wrapper stack (applied in order by :func:`wrap_environment`)
-------------------------------------------------------------
1. :class:`MultiScenarioWrapper` — selects a new route from the manifest before
   each episode reset.
2. :class:`MetricsInfoWrapper` — accumulates traffic metrics into the
   ``info`` dict at each step and at episode end.

Typical usage
-------------
::

    from traffic_drl.environment.wrappers import wrap_environment
    from traffic_drl.train.scenario_sampler import ScenarioSampler

    sampler = ScenarioSampler(manifest, allowed_split="TR", seed=42)
    wrapped = wrap_environment(base_env, scenario_sampler=sampler)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    # Avoid a hard circular import at runtime; ScenarioSampler imports
    # ScenarioRecord from contracts, which has no env dependency.
    from traffic_drl.train.scenario_sampler import ScenarioSampler
    from traffic_drl.contracts import ScenarioRecord


class MultiScenarioWrapper(gym.Wrapper):
    """Select exactly one route per episode — never a comma-joined route list.

    On each ``reset`` call the wrapper asks ``scenario_sampler.sample()`` for
    the next :class:`~traffic_drl.contracts.ScenarioRecord` and reconfigures
    the underlying SUMO-RL environment for that route.

    Attributes:
        scenario_sampler: Restricted to one manifest split; provides
            :meth:`~traffic_drl.train.scenario_sampler.ScenarioSampler.sample`.
        active_scenario: The :class:`~traffic_drl.contracts.ScenarioRecord`
            selected for the current episode; ``None`` before the first reset.

    TODO (SV2): implement ``reset`` to reconfigure the inner SUMO-RL env for
    the newly selected route file without restarting the SUMO process if
    SUMO-RL's ``reset`` supports route switching.
    """

    def __init__(self, env: gym.Env, scenario_sampler: "ScenarioSampler") -> None:
        super().__init__(env)
        self.scenario_sampler: "ScenarioSampler" = scenario_sampler
        self.active_scenario: "ScenarioRecord | None" = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Sample a new scenario then delegate to the wrapped environment."""

        self.active_scenario = self.scenario_sampler.sample()
        if options is None:
            options = {}
        options["route_file"] = str(self.active_scenario.route_file)
        
        # SUMO-RL uses route_file internally, we also set it directly on the unwrapped env
        # just in case options are not properly propagated by all wrapper layers.
        if hasattr(self.unwrapped, "_route"):
            self.unwrapped._route = str(self.active_scenario.route_file)
            
        obs, info = super().reset(seed=seed, options=options)
        info["scenario"] = self.active_scenario
        return obs, info

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Delegate one Gymnasium step and attach active scenario metadata."""

        obs, reward, terminated, truncated, info = super().step(action)
        if self.active_scenario is not None:
            info["scenario"] = self.active_scenario
        return obs, reward, terminated, truncated, info


class MetricsInfoWrapper(gym.Wrapper):
    """Accumulate traffic metrics and expose them in the Gymnasium ``info`` dict.

    At each step the wrapper reads per-lane queue and waiting-time values from
    the SUMO-RL traffic-signal object and accumulates them.  At episode end it
    computes episode-level aggregates and attaches them to ``info`` so that
    :class:`~traffic_drl.train.callbacks.TrafficMetricsCallback` can log them.

    Expected ``info`` keys at episode end
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``average_waiting_time`` (float)
    - ``average_queue_length`` (float)
    - ``time_loss`` (float)
    - ``throughput`` (float)
    - ``phase_switch_rate`` (float)
    - ``min_green_violations`` (int)

    TODO (SV2): map these keys to the actual SUMO-RL ``info`` fields returned
    by ``SumoEnvironment.step``.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Reset the environment and clear accumulated metric state."""

        self._step_count = 0
        self._total_queue_length = 0.0
        self._phase_switches = 0
        self._min_green_violations = 0
        self._last_phase = {}
        self._time_in_phase = {}
        
        return super().reset(seed=seed, options=options)

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Collect metrics while preserving the Gymnasium step contract."""

        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1
        
        # Accumulate metrics from all traffic signals for phase switching
        signals = getattr(self.unwrapped, "traffic_signals", {})
        
        for ts_id, ts in signals.items():
            # Phase Tracking
            current_phase = getattr(ts, "green_phase", None)
            if current_phase is not None:
                if ts_id not in self._last_phase:
                    self._last_phase[ts_id] = current_phase
                    self._time_in_phase[ts_id] = 0
                    
                if current_phase != self._last_phase[ts_id]:
                    self._phase_switches += 1
                    if self._time_in_phase[ts_id] < getattr(ts, "min_green", 0):
                        self._min_green_violations += 1
                    self._last_phase[ts_id] = current_phase
                    self._time_in_phase[ts_id] = 0
                else:
                    self._time_in_phase[ts_id] += getattr(ts, "delta_time", 1)
                
        self._total_queue_length += info.get("system_total_stopped", 0.0)
        
        if terminated or truncated:
            # Averages over the episode length
            info["average_waiting_time"] = info.get("system_mean_waiting_time", 0.0)
            info["average_queue_length"] = self._total_queue_length / max(1, self._step_count)
            # Use sumo-rl's internal variables for accurate throughput that isn't dropped between delta_time steps
            info["throughput"] = info.get("system_total_arrived", 0.0)
            info["phase_switch_rate"] = self._phase_switches / max(1, self._step_count)
            info["min_green_violations"] = self._min_green_violations
            
        return obs, reward, terminated, truncated, info


def wrap_environment(
    env: gym.Env,
    scenario_sampler: "ScenarioSampler | None" = None,
) -> gym.Env:
    """Apply the project wrappers in their required order.

    Order matters: :class:`MultiScenarioWrapper` must be the outermost wrapper
    (applied last) so that ``reset`` route-switching happens before
    :class:`MetricsInfoWrapper` initialises its accumulators.

    Args:
        env: Base SUMO-RL Gymnasium environment from :func:`~traffic_drl.environment.make_env.create_sumo_env`.
        scenario_sampler: If provided, wraps with :class:`MultiScenarioWrapper`.
            Omit for single-scenario evaluation.

    TODO (SV2): apply :class:`MetricsInfoWrapper` first, then
    :class:`MultiScenarioWrapper` if a sampler is provided.
    """

    env = MetricsInfoWrapper(env)
    if scenario_sampler is not None:
        env = MultiScenarioWrapper(env, scenario_sampler)
    return env
