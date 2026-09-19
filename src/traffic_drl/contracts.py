"""Shared contracts exchanged across the traffic DRL pipeline.

This module is the single source of truth for every data class and protocol
used across the project.  Nothing in this module depends on SUMO, SB3, or any
heavy runtime library — it can be imported safely in tests and type stubs.

Data classes
------------
- :class:`ScenarioRecord` — one row of the scenario manifest CSV.
- :class:`EpisodeMetrics` — typed traffic metrics for one controller episode.
- :class:`MetricSummary` — aggregated metrics for one controller/scenario group.
- :class:`BenchmarkReport` — final benchmark comparison relative to a baseline.
- :class:`TripInfoMetrics` — metrics parsed from a SUMO ``tripinfo.xml``.
- :class:`EmissionMetrics` — metrics parsed from a SUMO emissions XML.
- :class:`SmokeTestResult` — outcome of a short smoke test.
- :class:`ResumeInfo` — metadata needed to resume a training run.

Protocols (structural interfaces)
----------------------------------
Each protocol is ``@runtime_checkable`` so that ``isinstance(obj, Controller)``
works in tests and assertions.

+---------------------+-------------------------------------------------+----------------------------------------------------+
| Protocol            | Concrete implementations                        | Consumed by                                        |
+=====================+=================================================+====================================================+
| Controller          | MaxPressureController (baselines/max_pressure)  | evaluate_controller, evaluate_manifest             |
|                     | FixedTimeController (baselines/fixed_time)      | (evaluation/evaluate_benchmark)                    |
|                     | ActuatedController (baselines/actuated)         |                                                    |
|                     | SB3 DQN / PPO (via .predict)                    |                                                    |
+---------------------+-------------------------------------------------+----------------------------------------------------+
| EnvironmentFactory  | lambda / nested function created by the         | evaluate_manifest                                  |
|                     | benchmark runner                                | (evaluation/evaluate_benchmark)                    |
+---------------------+-------------------------------------------------+----------------------------------------------------+
| ScenarioSource      | ScenarioManifest                                | ScenarioSampler (train/scenario_sampler)           |
|                     | (environment/scenario_factory)                  | evaluate_manifest (evaluation/evaluate_benchmark)  |
+---------------------+-------------------------------------------------+----------------------------------------------------+
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioRecord:
    """One immutable row from the scenario manifest.

    Created by :meth:`~traffic_drl.environment.scenario_factory.ScenarioManifest.from_csv`
    and consumed by :class:`~traffic_drl.train.scenario_sampler.ScenarioSampler`
    and benchmark evaluation.

    TODO (SV3): verify split ownership, seeds, and checksum metadata before locking TE.
    """

    scenario_id: str
    split: str
    route_file: Path
    demand_seed: int
    sumo_seed: int
    num_seconds: int
    #: Optional scalar/string metadata from extra CSV columns.
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeMetrics:
    """Traffic and control metrics collected from one controller episode.

    ``average_waiting_time``, ``average_queue_length``, and ``time_loss`` are
    the primary proposal metrics.  ``throughput``, ``phase_switch_rate``, and
    ``min_green_violations`` are control constraints; remaining fields are
    supplementary.

    Created by :func:`~traffic_drl.evaluation.metrics.collect_episode_metrics`
    and written by :func:`~traffic_drl.evaluation.evaluate_benchmark.save_evaluation_results`.
    """

    controller: str
    scenario_id: str
    seed: int
    average_waiting_time: float
    average_queue_length: float
    time_loss: float
    throughput: float
    phase_switch_rate: float
    min_green_violations: int
    travel_time: float | None = None
    spillback: float | None = None
    recovery_time: float | None = None
    fuel: float | None = None
    co2: float | None = None
    nox: float | None = None
    inference_latency: float | None = None


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated metric values for one controller/scenario group.

    Created by :func:`~traffic_drl.evaluation.metrics.aggregate_metrics`.
    """

    controller: str
    scenario_id: str
    sample_count: int
    means: dict[str, float]
    standard_deviations: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class BenchmarkReport:
    """Interpreted benchmark output relative to a named baseline.

    Created by :func:`~traffic_drl.evaluation.metrics.interpret_results`.
    """

    baseline_controller: str
    summaries: tuple[MetricSummary, ...]
    #: Per-metric percentage improvement over the baseline, keyed by controller name.
    improvements: dict[str, dict[str, float]]
    #: Number of constraint violations per controller.
    constraint_violations: dict[str, int]


@dataclass(frozen=True)
class TripInfoMetrics:
    """Traffic metrics parsed from one SUMO tripinfo file.

    Created by :func:`~traffic_drl.evaluation.parse_tripinfo.parse_tripinfo`.
    """

    vehicle_count: int
    average_waiting_time: float
    average_time_loss: float
    average_travel_time: float
    throughput: float


@dataclass(frozen=True)
class EmissionMetrics:
    """Environmental metrics parsed from one SUMO emissions file.

    Created by :func:`~traffic_drl.evaluation.parse_tripinfo.parse_emissions`.
    """

    fuel: float
    co2: float
    nox: float


@dataclass(frozen=True)
class SmokeTestResult:
    """Outcome of a short environment reset/step smoke test.

    Created by :func:`~traffic_drl.evaluation.test.run_smoke_test`.
    """

    steps: int
    terminated: bool
    truncated: bool
    rewards: tuple[float, ...]
    last_info: dict[str, str | int | float | bool]


@dataclass
class ResumeInfo:
    """Metadata required to resume a training run reproducibly.

    Created and written to JSON by
    :func:`~traffic_drl.train.checkpointing.save_checkpoint`.
    Loaded by :func:`~traffic_drl.train.checkpointing.read_resume_info`.
    """

    num_timesteps: int
    model_class: str
    seed: int | None = None
    scenario_split: str | None = None
    config_path: Path | None = None
    git_commit: str | None = None
    best_metric_value: float | None = None
    best_metric_name: str | None = None
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class Controller(Protocol):
    """Common interface for learned and heuristic controllers.

    This is a *structural* protocol — any object that exposes ``predict`` and
    ``reset`` with the correct signatures satisfies it.  Use
    ``isinstance(obj, Controller)`` to assert conformance at runtime.

    Concrete implementations
    ~~~~~~~~~~~~~~~~~~~~~~~~
    - :class:`~traffic_drl.baselines.max_pressure.MaxPressureController`
    - :class:`~traffic_drl.baselines.fixed_time.FixedTimeController`
    - :class:`~traffic_drl.baselines.actuated.ActuatedController`
    - SB3 ``DQN`` / ``PPO`` objects returned by ``build_dqn_model`` /
      ``build_ppo_model`` (their ``.predict`` already satisfies this contract).

    Consumed by
    ~~~~~~~~~~~
    - :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_controller`
    - :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_manifest`
    """

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int | np.ndarray:
        """Return one legal action for an observation.

        Args:
            observation: Flat numeric state vector from the wrapped environment.
            deterministic: Disable policy exploration for benchmark evaluation.

        Returns:
            A legal action accepted by the environment — an integer phase index
            for single-signal discrete control, or a numpy array for
            multi-signal or continuous control.
        """
        ...

    def reset(self) -> None:
        """Reset any internal episode state before a new episode begins.

        Returns:
            None.
        """
        ...


@runtime_checkable
class EnvironmentFactory(Protocol):
    """Callable factory that creates one fresh environment per scenario and seed.

    This protocol is intentionally a *callable*, not a class.  In practice it is
    a ``lambda`` or nested function that closes over the shared
    :class:`~traffic_drl.config.EnvConfig` and returns a configured
    :class:`gymnasium.Env`.

    Consumed by
    ~~~~~~~~~~~
    - :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_manifest`

    Example
    ~~~~~~~
    ::

        factory: EnvironmentFactory = lambda record, seed: create_sumo_env(
            config, record.route_file, seed=seed
        )
    """

    def __call__(self, scenario: ScenarioRecord, seed: int) -> gym.Env:
        """Create one Gymnasium environment for one scenario and seed.

        Args:
            scenario: A single route record from the manifest.
            seed: SUMO/environment seed for this rollout.

        Returns:
            A fresh, unseeded Gymnasium environment ready for ``reset``.
        """
        ...


@runtime_checkable
class ScenarioSource(Protocol):
    """Read-only manifest contract required by the scenario sampler.

    Concrete implementation
    ~~~~~~~~~~~~~~~~~~~~~~~
    - :class:`~traffic_drl.environment.scenario_factory.ScenarioManifest`

    Consumed by
    ~~~~~~~~~~~
    - :class:`~traffic_drl.train.scenario_sampler.ScenarioSampler`
    - :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_manifest`
    """

    def records_for_split(self, split: str) -> Sequence[ScenarioRecord]:
        """Return all records belonging to *split*.

        Args:
            split: Split label such as ``TR``, ``VA``, or ``TE``.

        Returns:
            An ordered sequence of matching records.
        """
        ...

    def get(self, scenario_id: str) -> ScenarioRecord:
        """Return one record by its stable manifest ID.

        Args:
            scenario_id: The ``scenario_id`` column value from the CSV.

        Returns:
            The matching :class:`ScenarioRecord`.

        Raises:
            KeyError: If no record with that ID exists.
        """
        ...
