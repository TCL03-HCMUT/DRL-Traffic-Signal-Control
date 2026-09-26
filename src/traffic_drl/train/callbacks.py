"""Stable-Baselines3 callbacks for checkpoints and traffic metrics."""
from __future__ import annotations

import math
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from traffic_drl.contracts import ResumeInfo
from .checkpointing import save_checkpoint


class RobustCheckpointCallback(BaseCallback):
    """Maintain exactly two checkpoint directories: ``last/`` and ``best/``.

    ``last/`` is overwritten every *save_freq* steps to enable resumption.
    ``best/`` is overwritten only when the tracked metric improves.

    The "best" metric is determined by **episode reward mean** (higher is
    better).  On a tie, **average waiting time** (lower is better) is used
    as a tiebreaker.

    Args:
        save_freq: Save a ``last/`` checkpoint every this many steps.
        save_dir: Parent directory that will contain ``last/`` and ``best/``.
        save_replay_buffer: Persist the DQN replay buffer (ignored for PPO).
        save_vec_normalize: Persist ``VecNormalize`` running statistics.
        verbose: SB3 callback verbosity level.
    """

    def __init__(
        self,
        save_freq: int,
        save_dir: str | Path,
        *,
        save_replay_buffer: bool = True,
        save_vec_normalize: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.save_replay_buffer = save_replay_buffer
        self.save_vec_normalize = save_vec_normalize

        # Best-metric tracking state
        self._best_reward: float = -math.inf
        self._best_waiting_time: float = math.inf

    def _get_vec_normalize_env(self):
        """Return the VecNormalize wrapper if applicable, else None."""
        if self.save_vec_normalize and hasattr(self.training_env, "save"):
            return self.training_env
        return None

    def _build_resume_info(self) -> ResumeInfo:
        """Build a ResumeInfo from current training state."""
        return ResumeInfo(
            num_timesteps=self.num_timesteps,
            model_class=self.model.__class__.__name__,
            best_metric_value=self._best_reward,
            best_metric_name="ep_rew_mean",
        )

    def _save_bundle(self, name: str) -> None:
        """Save a checkpoint bundle into ``save_dir / name``."""
        save_checkpoint(
            self.model,
            self.save_dir / name,
            vec_normalize_env=self._get_vec_normalize_env(),
            save_replay_buffer=self.save_replay_buffer,
            save_rng_state=True,
            resume_info=self._build_resume_info(),
        )
        if self.verbose >= 1:
            print(f"[RobustCheckpointCallback] Saved '{name}' checkpoint "
                  f"at step {self.num_timesteps}")

    def _check_best(self) -> None:
        """Check whether the current episode metrics beat the best so far.

        Uses episode reward mean as the primary metric (higher is better).
        On tie, uses average_waiting_time as a tiebreaker (lower is better).
        """
        # Read current episode reward from SB3's internal logger
        # ep_rew_mean is populated after the first episode completes
        if not hasattr(self.model, "logger") or self.model.logger is None:
            return

        # SB3 stores name_to_value in the logger's internal dict
        log_dict = getattr(self.model.logger, "name_to_value", {})
        current_reward = log_dict.get("rollout/ep_rew_mean")
        if current_reward is None:
            return

        # Read waiting time from the most recent info (if available)
        current_waiting = math.inf
        for info in self.locals.get("infos", []):
            if "average_waiting_time" in info:
                current_waiting = info["average_waiting_time"]

        # Compare: higher reward is better; on tie, lower waiting time wins
        is_better = False
        if current_reward > self._best_reward:
            is_better = True
        elif current_reward == self._best_reward and current_waiting < self._best_waiting_time:
            is_better = True

        if is_better:
            self._best_reward = current_reward
            self._best_waiting_time = current_waiting
            self._save_bundle("best")

    def _on_step(self) -> bool:
        """Save ``last/`` periodically and ``best/`` when metrics improve.

        Returns:
            bool: ``True`` to continue training.
        """
        # Check for best before saving last (so best reflects the latest state)
        self._check_best()

        if self.n_calls % self.save_freq == 0:
            self._save_bundle("last")

        return True

    def _on_training_end(self) -> None:
        """Always save a final ``last/`` checkpoint when training ends."""
        self._save_bundle("last")


class TrafficMetricsCallback(BaseCallback):
    """Copy episode traffic metrics from ``info`` into SB3's logger."""

    def _on_step(self) -> bool:
        """Log waiting, queue, timeLoss, throughput and control metrics.

        Returns:
            bool: ``True`` to continue training; ``False`` to stop training.
        """
        for info in self.locals.get("infos", []):
            for metric in ("average_waiting_time", "average_queue_length", "time_loss", "throughput", "phase_switch_rate", "min_green_violations"):
                if metric in info:
                    self.logger.record(f"traffic/{metric}", info[metric])
        return True
