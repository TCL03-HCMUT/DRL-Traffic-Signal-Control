"""Fixed-time signal controller — satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

The fixed-time baseline runs SUMO with ``fixed_ts=True`` so the signal keeps a
pre-programmed phase plan and never responds to traffic state.  It is evaluated
through the same :func:`~traffic_drl.evaluation.evaluate_benchmark.evaluate_controller`
path as every other controller.

Typical usage
-------------
::

    from traffic_drl.baselines.fixed_time import FixedTimeController, make_fixed_time_env
    from traffic_drl.evaluation.evaluate_benchmark import evaluate_controller

    controller = FixedTimeController()
    env = make_fixed_time_env(record, config)
    metrics = evaluate_controller(controller, env, controller_name="fixed_time",
                                  scenario_id=record.scenario_id, seed=record.sumo_seed)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from traffic_drl.contracts import ScenarioRecord
from traffic_drl.config import EnvConfig


class FixedTimeController:
    """Controller stub for the fixed-time (pre-timed) signal baseline.

    The fixed-time plan is encoded in the SUMO network/additional files and
    executed by SUMO when ``fixed_ts=True`` is passed to the SUMO-RL
    environment.  ``predict`` is called at each control step but must always
    return the **current** phase (no switching) so the environment's
    ``fixed_ts`` logic takes over.

    Satisfies the :class:`~traffic_drl.contracts.Controller` protocol.

    TODO (SV3): confirm ``fixed_ts=True`` passes through SUMO-RL correctly and
    that the returned action is the identity action for the environment.
    """

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        """Return the identity (no-switch) action for the current phase."""
        return 0

    def reset(self) -> None:
        """No internal state to reset for fixed-time control."""
        pass


def make_fixed_time_env(
    record: ScenarioRecord, 
    config: EnvConfig,
    run_id: str = "baseline_fixed_time",
    base_dir: str | Path = "outputs/runs",
) -> gym.Env:
    """Create a SUMO-RL environment configured for fixed-time control."""
    from traffic_drl.environment.make_env import create_sumo_env
    from pathlib import Path
    
    return create_sumo_env(
        config=config,
        route_file=record.route_file,
        run_id=run_id,
        base_dir=Path(base_dir),
        fixed_ts=True,
        seed=record.sumo_seed,
        wrap=True,
    )
