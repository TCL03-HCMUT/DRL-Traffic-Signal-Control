"""Training, checkpointing, and scenario-sampling APIs.

Public API
----------
- :func:`~traffic_drl.train.train_dqn.build_dqn_model` — construct an SB3 DQN.
- :func:`~traffic_drl.train.train_dqn.train_dqn` — run the DQN training loop.
- :func:`~traffic_drl.train.train_dqn.save_dqn_checkpoint` — save DQN bundle.
- :func:`~traffic_drl.train.train_dqn.load_dqn_checkpoint` — load DQN bundle.
- :func:`~traffic_drl.train.train_dqn.run_dqn_pilot` — full DEV-00 pilot pipeline.
- :func:`~traffic_drl.train.train_ppo.build_ppo_model` — construct an SB3 PPO *(Phase 2)*.
- :func:`~traffic_drl.train.train_ppo.train_ppo` — run the PPO training loop *(Phase 2)*.
- :class:`~traffic_drl.train.scenario_sampler.ScenarioSampler` — leakage-safe episode sampler.
- :func:`~traffic_drl.train.lr_schedules.linear_schedule` — linear LR schedule.
- :func:`~traffic_drl.train.lr_schedules.cosine_schedule` — cosine-annealing LR schedule.
- :class:`~traffic_drl.train.callbacks.RobustCheckpointCallback` — checkpoint SB3 callback.
- :class:`~traffic_drl.train.callbacks.TrafficMetricsCallback` — metric-logging SB3 callback.
- :func:`~traffic_drl.train.checkpointing.save_checkpoint` — low-level checkpoint save.
- :func:`~traffic_drl.train.checkpointing.load_checkpoint` — low-level checkpoint load.
- :func:`~traffic_drl.train.checkpointing.read_resume_info` — read resume JSON.
"""
from traffic_drl.train.scenario_sampler import ScenarioSampler
from traffic_drl.train.lr_schedules import linear_schedule, cosine_schedule
from traffic_drl.train.checkpointing import (
    CheckpointBundle,
    save_checkpoint,
    load_checkpoint,
    read_resume_info,
)
from traffic_drl.train.callbacks import (
    RobustCheckpointCallback,
    TrafficMetricsCallback,
)
from traffic_drl.train.train_dqn import (
    build_dqn_model,
    train_dqn,
    save_dqn_checkpoint,
    load_dqn_checkpoint,
    run_dqn_pilot,
)
from traffic_drl.train.train_ppo import (
    build_ppo_model,
    train_ppo,
    save_ppo_checkpoint,
    load_ppo_checkpoint,
)

__all__ = [
    # Sampler
    "ScenarioSampler",
    # LR schedules
    "linear_schedule",
    "cosine_schedule",
    # Checkpointing
    "CheckpointBundle",
    "save_checkpoint",
    "load_checkpoint",
    "read_resume_info",
    # Callbacks
    "RobustCheckpointCallback",
    "TrafficMetricsCallback",
    # DQN
    "build_dqn_model",
    "train_dqn",
    "save_dqn_checkpoint",
    "load_dqn_checkpoint",
    "run_dqn_pilot",
    # PPO
    "build_ppo_model",
    "train_ppo",
    "save_ppo_checkpoint",
    "load_ppo_checkpoint",
]
