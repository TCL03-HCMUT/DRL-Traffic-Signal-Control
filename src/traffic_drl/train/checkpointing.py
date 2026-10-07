"""Shared Stable-Baselines3 checkpoint persistence utilities.

One checkpoint bundle = one dedicated directory containing:
- ``model.zip``          — SB3 model weights, optimizer, hyperparameters.
- ``replay_buffer.pkl``  — DQN replay memory (optional).
- ``vec_normalize.pkl``  — ``VecNormalize`` running statistics (optional).
- ``rng_state.pkl``      — numpy / torch / python RNG states for exact resumption.
- ``resume_info.json``   — human-readable metadata for reproducibility.

A training run maintains exactly two checkpoint directories:
- ``last/``  — overwritten every ``save_freq`` steps.
- ``best/``  — overwritten only when a tracked metric improves.

Typical usage
-------------
::

    from traffic_drl.train.checkpointing import save_checkpoint, load_checkpoint
    from stable_baselines3 import DQN

    bundle = save_checkpoint(model, "outputs/checkpoints/last",
                             vec_normalize_env=vec_env,
                             save_replay_buffer=True)

    loaded = load_checkpoint(DQN, "outputs/checkpoints/last", env=fresh_env)
"""
from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from traffic_drl.contracts import ResumeInfo

if TYPE_CHECKING:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import VecNormalize

# Generic SB3 model type — either DQN or PPO.
# Used to preserve the concrete type through load_checkpoint's return.
_SB3ModelT = TypeVar("_SB3ModelT")


class CheckpointBundle:
    """Paths for one SB3 model checkpoint and its companion artifacts.

    Constructed by :func:`save_checkpoint`; passed to :func:`load_checkpoint`.

    Attributes:
        bundle_dir: Root directory for this checkpoint.
        model_path: ``.zip`` file containing the serialised SB3 policy.
        replay_buffer_path: ``.pkl`` for DQN replay memory.
        vec_normalize_path: ``.pkl`` for ``VecNormalize`` statistics.
        rng_state_path: ``.pkl`` for numpy/torch/python RNG states.
        resume_info_path: ``.json`` for reproducibility metadata.
    """

    def __init__(self, bundle_dir: str | Path) -> None:
        """Derive all artifact paths inside the dedicated *bundle_dir*.

        Args:
            bundle_dir: Directory that will contain the checkpoint artifacts.
        """
        self.bundle_dir = Path(bundle_dir)
        self.model_path: Path = self.bundle_dir / "model.zip"
        self.replay_buffer_path: Path = self.bundle_dir / "replay_buffer.pkl"
        self.vec_normalize_path: Path = self.bundle_dir / "vec_normalize.pkl"
        self.rng_state_path: Path = self.bundle_dir / "rng_state.pkl"
        self.resume_info_path: Path = self.bundle_dir / "resume_info.json"

    def ensure_dir(self) -> None:
        """Create the checkpoint directory if it does not already exist."""
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def has_vec_normalize(self) -> bool:
        """Check whether a saved VecNormalize file exists in this bundle."""
        return self.vec_normalize_path.exists()

    def has_replay_buffer(self) -> bool:
        """Check whether a saved replay buffer exists in this bundle."""
        return self.replay_buffer_path.exists()

    def has_rng_state(self) -> bool:
        """Check whether a saved RNG state file exists in this bundle."""
        return self.rng_state_path.exists()

    def has_resume_info(self) -> bool:
        """Check whether resume info JSON exists in this bundle."""
        return self.resume_info_path.exists()


def _capture_rng_state() -> dict:
    """Capture RNG states from numpy, python random, and torch (if available)."""
    state = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    try:
        import torch
        state["torch_cpu"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    except ImportError:
        pass
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNG states for numpy, python random, and torch (if available)."""
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        try:
            import torch
            torch.random.set_rng_state(state["torch_cpu"])
            if "torch_cuda" in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except ImportError:
            pass


def save_checkpoint(
    model: "DQN | PPO",
    bundle_dir: str | Path,
    *,
    vec_normalize_env: "VecNormalize | None" = None,
    save_replay_buffer: bool = False,
    save_rng_state: bool = True,
    resume_info: ResumeInfo | None = None,
) -> CheckpointBundle:
    """Save an SB3 model and all companion artifacts into a dedicated directory.

    Args:
        model: An SB3 algorithm instance (``DQN`` or ``PPO``) exposing
            ``.save()`` and optionally ``.save_replay_buffer()``.
        bundle_dir: Destination directory for the checkpoint bundle.
        vec_normalize_env: ``VecNormalize`` wrapper whose running statistics
            are saved to ``vec_normalize.pkl``.
        save_replay_buffer: Save replay memory when the model supports it
            (DQN only).
        save_rng_state: Capture and persist numpy/torch/python RNG states
            for bit-exact resumption.
        resume_info: Typed reproducibility metadata written as UTF-8 JSON.

    Returns:
        CheckpointBundle: Paths for the model and all saved companion artifacts.
    """
    bundle = CheckpointBundle(bundle_dir)
    bundle.ensure_dir()
    model.save(str(bundle.model_path))

    if save_replay_buffer and hasattr(model, "save_replay_buffer"):
        model.save_replay_buffer(str(bundle.replay_buffer_path))

    if vec_normalize_env is not None:
        vec_normalize_env.save(str(bundle.vec_normalize_path))

    if save_rng_state:
        rng_state = _capture_rng_state()
        with open(bundle.rng_state_path, "wb") as f:
            pickle.dump(rng_state, f)

    if resume_info is not None:
        payload = asdict(resume_info) if isinstance(resume_info, ResumeInfo) else dict(resume_info)
        bundle.resume_info_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    return bundle


def load_checkpoint(
    model_class: "type[_SB3ModelT]",
    bundle_dir: str | Path,
    *,
    env: "object | None" = None,
    training: bool = True,
    norm_reward: bool = False,
    restore_replay_buffer: bool = True,
    restore_rng_state: bool = True,
) -> "_SB3ModelT":
    """Load an SB3 model and all available companion artifacts from a bundle directory.

    Auto-detects ``vec_normalize.pkl``, ``replay_buffer.pkl``, and
    ``rng_state.pkl`` inside the bundle directory — no explicit paths required.

    Args:
        model_class: The SB3 class to load with, e.g. ``DQN`` or ``PPO``.
        bundle_dir: Directory containing the checkpoint bundle.
        env: Fresh environment compatible with the saved model.  Required when
            the bundle contains ``vec_normalize.pkl``.
        training: Keep normalisation statistics updating when ``True`` (for
            resumed training); ``False`` for evaluation.
        norm_reward: Restore reward normalisation mode.
        restore_replay_buffer: If True and a replay buffer exists in the
            bundle, load it into the model.
        restore_rng_state: If True and an RNG state file exists in the
            bundle, restore numpy/torch/python RNG states.

    Returns:
        An instance of *model_class* loaded from the bundle.
    """
    bundle = CheckpointBundle(bundle_dir)
    load_env = env

    # Auto-detect and restore VecNormalize
    if bundle.has_vec_normalize():
        if env is None:
            raise ValueError(
                f"env is required when loading VecNormalize statistics "
                f"(found {bundle.vec_normalize_path})."
            )
        from stable_baselines3.common.vec_env import VecNormalize

        load_env = VecNormalize.load(str(bundle.vec_normalize_path), env)
        load_env.training = training
        load_env.norm_reward = norm_reward

    loaded_model = model_class.load(str(bundle.model_path), env=load_env)

    # Auto-detect and restore replay buffer
    if restore_replay_buffer and bundle.has_replay_buffer():
        if hasattr(loaded_model, "load_replay_buffer"):
            loaded_model.load_replay_buffer(str(bundle.replay_buffer_path))

    # Auto-detect and restore RNG state
    if restore_rng_state and bundle.has_rng_state():
        with open(bundle.rng_state_path, "rb") as f:
            rng_state = pickle.load(f)
        _restore_rng_state(rng_state)

    return loaded_model


def read_resume_info(bundle_dir: str | Path) -> ResumeInfo:
    """Read resume metadata associated with a checkpoint bundle.

    Args:
        bundle_dir: Checkpoint directory whose ``resume_info.json`` is read.

    Raises:
        FileNotFoundError: If the resume-info JSON does not exist.

    Returns:
        ResumeInfo: Typed reproducibility metadata from the checkpoint bundle.
    """
    bundle = CheckpointBundle(bundle_dir)
    if not bundle.has_resume_info():
        raise FileNotFoundError(
            f"Resume info not found: {bundle.resume_info_path}"
        )
    payload = json.loads(bundle.resume_info_path.read_text(encoding="utf-8"))
    return ResumeInfo(
        num_timesteps=int(payload["num_timesteps"]),
        model_class=str(payload["model_class"]),
        seed=payload.get("seed"),
        scenario_split=payload.get("scenario_split"),
        config_path=Path(payload["config_path"]) if payload.get("config_path") else None,
        git_commit=payload.get("git_commit"),
        best_metric_value=payload.get("best_metric_value"),
        best_metric_name=payload.get("best_metric_name"),
        extra={k: v for k, v in payload.get("extra", {}).items()},
    )
