"""Subclasses adding all-red clearance phase support to sumo-rl.

Drop this file next to your training script and import from it instead of
instantiating `sumo_rl.SumoEnvironment` / `sumo_rl.TrafficSignal` directly.

Works for:
  - fixed_ts=True   (SUMO just replays whatever the .net.xml defines, so this
                      already supports all-red phases -- we only fix a
                      phase-counting bug here)
  - agent-controlled (green -> yellow -> all-red -> green state machine)
  - single-agent, multi-agent (env.py dispatches to
    self.traffic_signals[ts] polymorphically, so nothing else needs to change)
"""
from typing import Dict, Optional, Union

import numpy as np

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.observations import DefaultObservationFunction

from traffic_drl.environment.all_red_traffic_signal import AllRedTrafficSignal


class AllRedObservationFunction(DefaultObservationFunction):
    """Default observation function, but the min_green flag also accounts for red_time."""

    def __call__(self) -> np.ndarray:
        ts = self.ts
        phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]
        clearance = ts.yellow_time + getattr(ts, "red_time", 0)
        min_green = [0 if ts.time_since_last_phase_change < ts.min_green + clearance else 1]
        density = ts.get_lanes_density()
        queue = ts.get_lanes_queue()
        return np.array(phase_id + min_green + density + queue, dtype=np.float32)


class AllRedSumoEnvironment(SumoEnvironment):
    """SumoEnvironment that builds AllRedTrafficSignal instances instead of TrafficSignal.

    New kwargs:
        red_time (int): duration in seconds of the all-red clearance phase
            inserted between yellow and the next green, in agent-controlled
            mode. Default: 2. Ignored (but harmless) when fixed_ts=True,
            since that mode just replays the .net.xml program as-is.
        program_id (Optional[str/Dict[str,str]]): which <tlLogic programID=...>
            block to read phases from, per traffic light. Pass a single str
            to apply to every ts_id, a dict keyed by ts_id for per-signal
            control, or leave None to use whichever program SUMO loaded
            first (upstream default behavior). This is the knob for e.g.
            running a "static" program under fixed_ts and a different "rl"
            program (different green-phase grouping / action space) for
            agent-controlled runs, without touching the .net.xml between runs.

    All other kwargs are identical to sumo_rl.SumoEnvironment. If you don't
    pass observation_class explicitly, it defaults to
    AllRedObservationFunction so the min_green flag accounts for red_time.
    """

    def __init__(self, *args, red_time: int = 0, program_id: Optional[Union[str, Dict[str, str]]] = None, **kwargs):
        self.red_time = red_time
        self.program_id = program_id
        kwargs.setdefault("observation_class", AllRedObservationFunction)
        super().__init__(*args, **kwargs)
        assert self.delta_time > self.yellow_time + self.red_time, (
            "delta_time must be greater than yellow_time + red_time"
        )

    def _build_traffic_signals(self, conn):
        if not isinstance(self.reward_fn, dict):
            self.reward_fn = {ts: self.reward_fn for ts in self.ts_ids}
        if isinstance(self.program_id, dict):
            program_ids = self.program_id
        else:
            program_ids = {ts: self.program_id for ts in self.ts_ids}
        self.traffic_signals = {
            ts: AllRedTrafficSignal(
                self,
                ts,
                self.delta_time,
                self.yellow_time,
                self.red_time,
                self.min_green,
                self.max_green,
                self.enforce_max_green,
                self.begin_time,
                self.reward_fn[ts],
                self.reward_weights,
                conn,
                program_id=program_ids.get(ts),
            )
            for ts in self.ts_ids
        }
