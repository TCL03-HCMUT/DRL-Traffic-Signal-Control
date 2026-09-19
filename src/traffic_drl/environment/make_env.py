"""Factories for the SUMO-RL Gymnasium environment.

This module is the **canonical** owner of ``create_sumo_env``.  All other
modules that need a SUMO-RL environment should call this factory rather than
constructing one directly.

Dependency graph
----------------
::

    EnvConfig ──► create_sumo_env ──► gym.Env
                       │
                       └─► make_dev_environment   (smoke tests / DEV-00)
                       └─► make_vectorized_environment  (SB3 training)

Typical usage
-------------
::

    from traffic_drl.config import load_env_config
    from traffic_drl.environment.make_env import create_sumo_env

    config = load_env_config("configs/environment/dev_single_intersection.yaml")
    env = create_sumo_env(config, route_file="scenarios/TR-01.rou.xml", seed=42)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Callable, Union
import gymnasium as gym
from traffic_drl.config import EnvConfig, RewardConfig, load_env_config
from traffic_drl.environment.wrappers import wrap_environment
from traffic_drl.environment.custom_observations import MixedTrafficObservation
from traffic_drl.environment.custom_rewards import CombinedReward
from traffic_drl.train.scenario_sampler import ScenarioSampler
import traffic_drl.run_id as r_id
from traffic_drl.environment.all_red_env import AllRedSumoEnvironment

# Type aliases for SB3 vectorised environments.  We use strings here so the
# module can be imported even when stable-baselines3 is not installed.
# At runtime the real types are resolved inside the functions that use them.
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore[import]

    VecEnv = DummyVecEnv | VecNormalize
except ModuleNotFoundError:
    VecEnv = None  # type: ignore[assignment,misc]


# Sumo-rl built-in reward function names recognised without instantiation.
_BUILTIN_REWARD_NAMES = frozenset({"pressure", "queue", "diff-waiting-time", "wait"})


def build_reward_fn(reward_config: RewardConfig) -> Union[str, Callable]:
    """Resolve a :class:`~traffic_drl.config.RewardConfig` into a callable or string.

    Converts the ``reward.type`` / ``reward.params`` YAML block into the value
    expected by :func:`create_sumo_env`'s ``reward_fn`` argument.

    Supported types
    ---------------
    * ``"combined"`` -- instantiates :class:`~traffic_drl.environment.custom_rewards.CombinedReward`
      and forwards every key in ``reward.params`` as a keyword argument.  Unknown
      keys raise ``TypeError`` at construction time, making config errors loud.
    * sumo-rl built-ins (``"pressure"``, ``"queue"``, ``"diff-waiting-time"``) --
      returned as-is; sumo-rl resolves them internally.

    Args:
        reward_config: Parsed reward section from the training YAML.

    Returns:
        str | Callable: Value suitable for the ``reward_fn`` argument of
        :func:`create_sumo_env`.

    Raises:
        ValueError: If ``reward_config.type`` is not recognised.
        TypeError: If ``reward_config.params`` contains a key that
            ``CombinedReward.__init__`` does not accept.
    """
    rtype = reward_config.type
    params = dict(reward_config.params)  # copy so callers can't mutate config

    if rtype == "combined":
        return CombinedReward(**params)

    if rtype in _BUILTIN_REWARD_NAMES:
        if params:
            import warnings
            warnings.warn(
                f"reward.params {list(params)} are ignored for built-in reward '{rtype}'.",
                stacklevel=2,
            )
        return rtype

    raise ValueError(
        f"Unknown reward type '{rtype}'. "
        f"Expected 'combined' or one of {sorted(_BUILTIN_REWARD_NAMES)}."
    )


def create_sumo_env(
    config: EnvConfig | str | Path,
    route_file: str | Path,
    run_id: str,
    base_dir: Path,
    *,
    fixed_ts: bool = False,
    seed: int | None = None,
    use_gui: bool = False,
    reward_fn: str | Callable = "pressure",
    wrap: bool = False,
    scenario_sampler: ScenarioSampler = None,
    custom_observation: bool = True,
    approach_ids: Sequence[str] = (),
    ring_segment_ids: Sequence[str] = (),
    vehicle_classes: Sequence[str] = ("passenger", "bus", "truck"),
) -> gym.Env:
    """Create a single-agent SUMO-RL Gymnasium environment for one route.

    This is the canonical environment factory.  It accepts either a fully
    parsed :class:`~traffic_drl.config.EnvConfig` or a path to a YAML file
    (which is loaded automatically via :func:`~traffic_drl.config.load_env_config`).

    Args:
        config: Typed environment configuration or path to its YAML file.
        route_file: Exactly **one** ``.rou.xml`` route file.  Must not be a
            comma-joined multi-route string; SUMO-RL handles multi-route via
            the manifest + wrapper, not via concatenation.
        run_id: The id of the specific run
        base_dir: The base run directory
        fixed_ts: Run the pre-timed signal plan instead of agent control.
            Pass ``True`` when creating the fixed-time baseline environment.
        seed: Reproducibility seed for SUMO demand and behaviour.
        use_gui: Launch ``sumo-gui`` instead of headless ``sumo``.
        reward_fn: Reward function name accepted by SUMO-RL (e.g. ``"pressure"``
            or ``"queue"``).  Custom reward functions should be registered with
            SUMO-RL before calling this factory.
        wrap: Whether to wrap the environment with multi-scenario logic.
        scenario_sampler: Active scenario sampler to use if ``wrap`` is True.
        custom_observation: Whether to inject the custom MDP observation space.
        approach_ids: List of incoming edge IDs for observation space grouping.
        ring_segment_ids: List of roundabout ring edges for density calculations.
        vehicle_classes: List of vehicle classes to track for PCU metrics.

    Output contract: the runner must create ``outputs/tripinfo/<run_id>/`` and
    pass SUMO's ``--tripinfo-output`` and ``--emission-output`` paths via
    ``config.sumo_options.additional_sumo_cmd`` or the SUMO-RL constructor
    kwargs before this factory is called.

    TODO (SV1): map every ``EnvConfig`` field to the SUMO-RL ``SumoEnvironment``
    constructor.  Decide whether ``additional_sumo_cmd`` is passed as a string
    or split into a list.  Verify that ``route_file`` is never a comma-joined
    multi-route value.

    Returns:
        gym.Env: A Gymnasium-compatible SUMO-RL single-agent environment.
    """
    if isinstance(config, (str, Path)):
        config = load_env_config(config)

    # 1. Validation checks
    if "," in str(route_file):
        raise ValueError(
            f"route_file must be a single file, not a comma-joined multi-route value: {route_file}"
        )
    
    if not Path(config.network.net_file).exists():
        raise FileNotFoundError(f"SUMO network file not found: {config.network.net_file}")
        
    if not Path(route_file).exists():
        raise FileNotFoundError(f"SUMO route file not found: {route_file}")

    additional_cmd = [
        config.sumo_options.additional_sumo_cmd,
        "--lateral-resolution",
        str(config.sumo_options.lateral_resolution),
    ]

    # Create directories before assigning paths
    tripinfo_dir, results_dir, _, _ = r_id.ensure_run_directories(run_id, base_dir)

    if config.sumo_options.save_tripinfo:
        tripinfo_path = tripinfo_dir / "tripinfo.xml"
        emissions_path = tripinfo_dir / "emissions.xml"
        additional_cmd.append("--tripinfo-output")
        additional_cmd.append(str(tripinfo_path))
        additional_cmd.append("--emission-output")
        additional_cmd.append(str(emissions_path))

    cmds = " ".join(additional_cmd)
    
    env_kwargs = dict(
        net_file=config.network.net_file,
        route_file=str(route_file),
        out_csv_name=str(results_dir / "run_csv"),
        use_gui=use_gui,
        num_seconds=config.timing.num_seconds,
        min_green=config.timing.min_green,
        max_green=config.timing.max_green,
        delta_time=config.timing.delta_time,
        yellow_time=config.timing.yellow_time,
        red_time=config.timing.red_time,
        sumo_seed=seed if seed is not None else "random",
        fixed_ts=fixed_ts,
        reward_fn=reward_fn,
        single_agent=config.traffic_light.single_agent,
        additional_sumo_cmd=cmds,
        program_id=config.traffic_light.program_id,
    )
    
    if custom_observation:
        env_kwargs["observation_class"] = lambda ts: MixedTrafficObservation(
            ts, 
            approach_ids=approach_ids, 
            ring_segment_ids=ring_segment_ids, 
            vehicle_classes=vehicle_classes
        )

    env = AllRedSumoEnvironment(**env_kwargs)

    if wrap:
        env = wrap_environment(env, scenario_sampler)

    return env


def make_dev_environment(
    config: EnvConfig | str | Path = "configs/environment/dev_single_intersection.yaml",
    *,
    seed: int | None = None,
    use_gui: bool = False,
) -> gym.Env:
    """Create the short DEV-00 smoke-test environment.

    Wraps :func:`create_sumo_env` with the DEV-00 route from the manifest.
    Intended for quick import/step checks, not for training or evaluation.

    Args:
        config: Typed environment configuration or path to its YAML file.
        seed: Development seed for the pilot smoke test.
        use_gui: Whether to show SUMO-GUI during the smoke test.

    TODO (SV2): load the DEV-00 manifest record and pass its ``route_file``
    to :func:`create_sumo_env`.

    Returns:
        gym.Env: A short-lived Gymnasium environment for the DEV-00 smoke test.
    """
    if isinstance(config, (str, Path)):
        config = load_env_config(config)

    from traffic_drl.environment.scenario_factory import ScenarioManifest
    
    # Load the DEV-00 manifest record for the smoke test
    manifest = ScenarioManifest.from_csv("scenarios/pilot_scenario_manifest.csv")
    record = manifest.get("DEV-00")

    env = create_sumo_env(
        config=config,
        route_file=record.route_file,
        run_id="DEV-00-SMOKE",
        base_dir=Path("outputs/runs"),
        seed=seed,
        use_gui=use_gui,
    )

    # 2. Run SB3 compliance checks
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
    except ImportError:
        pass

    return env


def make_vectorized_environment(
    config: EnvConfig | str | Path,
    route_file: str | Path,
    run_id: str,
    base_dir: Path,
    *,
    fixed_ts: bool = False,
    seed: int | None = None,
    use_gui: bool = False,
    reward_fn: str | Callable = "pressure",
    wrap: bool = False,
    scenario_sampler: ScenarioSampler = None,
    custom_observation: bool = True,
    training: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    approach_ids: Sequence[str] = (),
    ring_segment_ids: Sequence[str] = (),
    vehicle_classes: Sequence[str] = ("passenger", "bus", "truck"),
    vec_normalize: str | Path | None = None,
    **kwargs,
) -> "DummyVecEnv | VecNormalize":
    """Create a ``DummyVecEnv`` and optional ``VecNormalize`` wrapper for SB3.

    The returned object is the ``env`` argument accepted by
    :func:`~traffic_drl.train.train_dqn.build_dqn_model`.

    Args:
        config: Typed environment configuration or path to its YAML file.
        route_file: One route XML file selected from the training split.
        run_id: The unique identifier for this run, used for output paths.
        base_dir: The root output directory for training artifacts.
        fixed_ts: If True, uses the pre-timed signal plan instead of agent control.
        seed: Seed forwarded to the underlying SUMO-RL environment.
        use_gui: Whether to launch sumo-gui instead of headless sumo.
        reward_fn: The reward function identifier or callable.
        wrap: Whether to wrap the base SUMO-RL env with ScenarioSampler logic.
        scenario_sampler: The scenario sampler to use if wrap is True.
        custom_observation: Whether to inject the custom MDP observation space.
        training: True if training, False if evaluating. Used by VecNormalize.
        norm_obs: Wrap with ``VecNormalize`` and normalise observations.
        norm_reward: Wrap with ``VecNormalize`` and normalise rewards.
        clip_obs: Maximum absolute value for normalised observations.
        clip_reward: Maximum absolute value for normalised rewards.
        approach_ids: Incoming edge IDs passed to the custom observation space.
        ring_segment_ids: Ring edges passed to the custom observation space.
        vehicle_classes: Vehicle classes to track in custom observations.
        vec_normalize: Optional path to a saved ``VecNormalize`` file to load.
        **kwargs: Extra parameters passed to :func:`create_sumo_env`.

    TODO (SV2): fit ``VecNormalize`` only on ``TR`` episodes and freeze it
    (``training=False``, ``norm_reward=False``) for ``VA``/``TE`` evaluation.

    Returns:
        DummyVecEnv | VecNormalize: A single-environment vectorised wrapper,
        optionally wrapped by ``VecNormalize``.
    """
    if isinstance(config, (str, Path)):
        config = load_env_config(config)

    def _make_env():
        return create_sumo_env(
            config,
            route_file,
            run_id=run_id,
            base_dir=base_dir,
            fixed_ts=fixed_ts,
            use_gui=use_gui,
            reward_fn=reward_fn,
            wrap=wrap,
            scenario_sampler=scenario_sampler,
            custom_observation=custom_observation,
            seed=seed,
            approach_ids=approach_ids,
            ring_segment_ids=ring_segment_ids,
            vehicle_classes=vehicle_classes,
            **kwargs,
        )

    env = DummyVecEnv([_make_env])


    if vec_normalize is not None:
        env = VecNormalize.load(str(vec_normalize), env)
        env.training = training
        env.norm_reward = norm_reward
    elif norm_obs or norm_reward:
        env = VecNormalize(
            env,
            training=training,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
        )

    return env


def close_environment(env: gym.Env) -> None:
    """Close the environment and terminate its SUMO subprocess.

    Calls ``env.close()`` which SUMO-RL forwards to the running SUMO process.
    Callers must invoke this after every episode loop to prevent orphaned SUMO
    processes.

    Args:
        env: Any Gymnasium, vectorised, or wrapped SUMO-RL environment.

    """
    env.close()
