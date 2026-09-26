"""Typed configuration dataclasses and YAML loaders for the traffic DRL project.

Each ``load_*`` function returns a typed, frozen dataclass instead of a raw
``dict``.  YAML is parsed once by the private :func:`_load_yaml` helper; the
public loaders validate required keys and construct the dataclass.

Typical usage
-------------
::

    from traffic_drl.config import load_train_config, TrainConfig

    cfg: TrainConfig = load_train_config("configs/train/dqn_phase1.yaml")
    print(cfg.experiment.seed)          # IDE knows this is an int
    print(cfg.model_hyperparameters)    # dict[str, ...] for algorithm-specific params
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return the raw dict.

    Args:
        path: Path to the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _require(section: dict, keys: list[str], context: str) -> None:
    """Raise ``ValueError`` for any key missing from *section*."""
    for key in keys:
        if key not in section:
            raise ValueError(f"Missing required field '{key}' in {context}")


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkConfig:
    """SUMO network and route file paths."""

    net_file: str
    route_files: list[str]
    additional_files: list[str]


@dataclass(frozen=True)
class TrafficLightConfig:
    """Traffic-light identity and agent-mode flag."""

    ts_id: str
    single_agent: bool
    program_id: str | None = None
    """SUMO tlLogic programID to activate.  ``None`` uses the first program
    SUMO loads (sumo-rl default).  Set to a specific string (e.g. ``"rl"``)
    to select a named program from the net file without modifying the XML.
    """


@dataclass(frozen=True)
class TimingConfig:
    """Phase-timing parameters shared by all controllers."""

    num_seconds: int
    delta_time: int
    min_green: int
    max_green: int
    yellow_time: int
    red_time: int


@dataclass(frozen=True)
class SumoOptions:
    """SUMO process options."""

    use_gui: bool
    lateral_resolution: float
    additional_sumo_cmd: str
    save_tripinfo: bool = False


@dataclass(frozen=True)
class ObservationBoundsConfig:
    """Clipping bounds for observation normalisation."""

    max_queue_per_lane: float
    max_time_in_phase: float


@dataclass(frozen=True)
class EnvConfig:
    """Full environment configuration parsed from a YAML file.

    Concrete implementors create one of these via :func:`load_env_config` and
    pass it to :func:`~traffic_drl.environment.make_env.create_sumo_env`.

    Example YAML structure::

        network:
          net_file: sumo/single_intersection.net.xml
          route_files: [scenarios/TR-01.rou.xml]
          additional_files: []
        traffic_light:
          ts_id: J0
          single_agent: true
        timing:
          num_seconds: 3600
          delta_time: 5
          min_green: 5
          max_green: 60
          yellow_time: 2
        sumo_options:
          use_gui: false
          lateral_resolution: 0.0
          additional_sumo_cmd: ""
          save_tripinfo: false
        observation_bounds:
          max_queue_per_lane: 10.0
          max_time_in_phase: 60.0
    """

    network: NetworkConfig
    traffic_light: TrafficLightConfig
    timing: TimingConfig
    sumo_options: SumoOptions
    observation_bounds: ObservationBoundsConfig


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    """High-level experiment identity."""

    name: str
    seed: int
    device: str
    split_allowed: str


@dataclass(frozen=True)
class TrainEnvConfig:
    """Environment pointers used during training (not the full EnvConfig)."""

    env_config_path: str
    manifest_path: str


@dataclass(frozen=True)
class RewardConfig:
    """Reward function selection and weight parameters."""

    type: str
    params: dict[str, float | int | str | bool] = field(default_factory=dict)
    wrappers: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalisationConfig:
    """Observation and reward normalisation settings.

    ``gamma`` is optional here; when the RL algorithm uses gamma it is
    typically found under ``model_hyperparameters`` in the YAML.
    """

    norm_obs: bool
    norm_reward: bool
    clip_obs: float
    clip_reward: float
    gamma: float = 0.99


@dataclass(frozen=True)
class TrainingControlConfig:
    """Training-loop stopping and logging settings."""

    total_timesteps: int
    save_freq: int
    eval_freq: int = 0
    log_csv: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Full training configuration parsed from a YAML file.

    Created by :func:`load_train_config` and consumed by
    :func:`~traffic_drl.train.train_dqn.build_dqn_model` and
    :func:`~traffic_drl.train.train_dqn.run_dqn_pilot`.

    ``model_hyperparameters`` is kept as a plain dict because the keys vary
    between SB3 algorithms (DQN vs PPO).
    """

    experiment: ExperimentConfig
    environment: TrainEnvConfig
    reward: RewardConfig
    normalisation: NormalisationConfig
    model_hyperparameters: dict[str, float | int | str | bool]
    training_control: TrainingControlConfig


# ---------------------------------------------------------------------------
# EvalConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark identity and evaluation-loop settings."""

    name: str
    split: str
    deterministic: bool
    episodes_per_scenario: int


@dataclass(frozen=True)
class ArtifactsConfig:
    """Paths to model and normalisation artifacts."""

    model_checkpoint: str
    vec_normalize_stats: str


@dataclass(frozen=True)
class EvalNormalisationConfig:
    """Normalisation settings during evaluation (freeze statistics)."""

    training: bool
    norm_reward: bool


@dataclass(frozen=True)
class ReportingConfig:
    """Output paths and metric selection for evaluation reporting."""

    output_dir: str
    save_tripinfo: bool
    metrics_to_collect: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    """Full evaluation configuration parsed from a YAML file.

    Created by :func:`load_eval_config` and consumed by
    :func:`~traffic_drl.evaluation.evaluate_benchmark.run_benchmark`.
    """

    benchmark: BenchmarkConfig
    env_config_path: str
    manifest_path: str
    checksum_file: str
    evaluation_seeds: list[int]
    scenarios: list[str]
    artifacts: ArtifactsConfig
    normalisation: EvalNormalisationConfig
    reporting: ReportingConfig


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_env_config(
    config_path: str | Path = "configs/environment/dev_single_intersection.yaml",
) -> EnvConfig:
    """Load and parse an environment configuration YAML file.

    Args:
        config_path: Path to the environment YAML config.

    Returns:
        EnvConfig: Typed, frozen environment configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required section or field is missing.
    """
    raw = _load_yaml(config_path)

    _require(raw, ["network", "traffic_light", "timing", "sumo_options", "observation_bounds"], "env config")

    net = raw["network"]
    _require(net, ["net_file", "route_files", "additional_files"], "network")

    tl = raw["traffic_light"]
    _require(tl, ["ts_id", "single_agent"], "traffic_light")

    timing = raw["timing"]
    _require(timing, ["num_seconds", "delta_time", "min_green", "max_green", "yellow_time"], "timing")

    sumo = raw["sumo_options"]
    _require(sumo, ["use_gui", "lateral_resolution", "additional_sumo_cmd"], "sumo_options")

    obs = raw["observation_bounds"]
    _require(obs, ["max_queue_per_lane", "max_time_in_phase"], "observation_bounds")

    return EnvConfig(
        network=NetworkConfig(
            net_file=net["net_file"],
            route_files=list(net["route_files"]),
            additional_files=list(net.get("additional_files", [])),
        ),
        traffic_light=TrafficLightConfig(
            ts_id=tl["ts_id"],
            single_agent=bool(tl["single_agent"]),
            program_id=tl.get("program_id") or None,
        ),
        timing=TimingConfig(
            num_seconds=int(timing["num_seconds"]),
            delta_time=int(timing["delta_time"]),
            min_green=int(timing["min_green"]),
            max_green=int(timing["max_green"]),
            yellow_time=int(timing["yellow_time"]),
            red_time=int(timing.get("red_time", 0)),
        ),
        sumo_options=SumoOptions(
            use_gui=bool(sumo["use_gui"]),
            lateral_resolution=float(sumo["lateral_resolution"]),
            additional_sumo_cmd=str(sumo.get("additional_sumo_cmd", "")),
            save_tripinfo=bool(sumo.get("save_tripinfo", False)),
        ),
        observation_bounds=ObservationBoundsConfig(
            max_queue_per_lane=float(obs["max_queue_per_lane"]),
            max_time_in_phase=float(obs["max_time_in_phase"]),
        ),
    )


def load_train_config(
    config_path: str | Path = "configs/train/dqn_phase1.yaml",
) -> TrainConfig:
    """Load and parse a training configuration YAML file.

    Args:
        config_path: Path to the training YAML config.

    Returns:
        TrainConfig: Typed, frozen training configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required section or field is missing.
    """
    raw = _load_yaml(config_path)

    _require(
        raw,
        ["experiment", "environment", "reward", "normalization", "model_hyperparameters", "training_control"],
        "train config",
    )

    exp = raw["experiment"]
    _require(exp, ["name", "seed", "device", "split_allowed"], "experiment")

    env_sec = raw["environment"]
    _require(env_sec, ["env_config_path", "manifest_path"], "environment")

    reward = raw["reward"]
    if "type" not in reward:
        raise ValueError("Missing required field 'type' in reward")

    norm = raw["normalization"]
    _require(norm, ["norm_obs", "norm_reward", "clip_obs", "clip_reward"], "normalization")

    ctrl = raw["training_control"]
    _require(ctrl, ["total_timesteps", "save_freq"], "training_control")

    return TrainConfig(
        experiment=ExperimentConfig(
            name=str(exp["name"]),
            seed=int(exp["seed"]),
            device=str(exp["device"]),
            split_allowed=str(exp["split_allowed"]),
        ),
        environment=TrainEnvConfig(
            env_config_path=str(env_sec["env_config_path"]),
            manifest_path=str(env_sec["manifest_path"]),
        ),
        reward=RewardConfig(
            type=str(reward["type"]),
            params=dict(reward.get("params") or {}),
            wrappers=dict(reward.get("wrappers") or {}),
        ),
        normalisation=NormalisationConfig(
            norm_obs=bool(norm["norm_obs"]),
            norm_reward=bool(norm["norm_reward"]),
            clip_obs=float(norm["clip_obs"]),
            clip_reward=float(norm["clip_reward"]),
            gamma=float(norm.get("gamma") or raw.get("model_hyperparameters", {}).get("gamma", 0.99)),
        ),
        model_hyperparameters=dict(raw["model_hyperparameters"]),
        training_control=TrainingControlConfig(
            total_timesteps=int(ctrl["total_timesteps"]),
            save_freq=int(ctrl["save_freq"]),
            eval_freq=int(ctrl.get("eval_freq", 0)),
            log_csv=bool(ctrl.get("log_csv", True)),
        ),
    )


def load_eval_config(
    config_path: str | Path = "configs/evaluation/phase1_validation.yaml",
) -> EvalConfig:
    """Load and parse an evaluation configuration YAML file.

    Args:
        config_path: Path to the evaluation YAML config.

    Returns:
        EvalConfig: Typed, frozen evaluation configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a required section or field is missing.
    """
    raw = _load_yaml(config_path)

    _require(
        raw,
        ["benchmark", "evaluation_seeds", "scenarios", "artifacts", "normalization", "reporting"],
        "eval config",
    )

    bench = raw["benchmark"]
    _require(bench, ["name", "split", "deterministic", "episodes_per_scenario"], "benchmark")

    arts = raw["artifacts"]
    _require(arts, ["model_checkpoint", "vec_normalize_stats"], "artifacts")

    norm = raw["normalization"]
    _require(norm, ["training", "norm_reward"], "normalization")

    rep = raw["reporting"]
    _require(rep, ["output_dir", "save_tripinfo"], "reporting")

    # manifest_path may sit at top level or inside benchmark
    env_config_path = str(
        raw.get("env_config_path") or bench.get("env_config_path", "configs/environment/dev_single_intersection.yaml")
    )
    manifest_path = str(
        raw.get("manifest_path") or bench.get("manifest_path", "scenarios/scenario_manifest.csv")
    )
    checksum_file = str(
        raw.get("checksum_file") or bench.get("checksum_file", "scenarios/checksums.sha256")
    )

    return EvalConfig(
        benchmark=BenchmarkConfig(
            name=str(bench["name"]),
            split=str(bench["split"]),
            deterministic=bool(bench["deterministic"]),
            episodes_per_scenario=int(bench["episodes_per_scenario"]),
        ),
        env_config_path=env_config_path,
        manifest_path=manifest_path,
        checksum_file=checksum_file,
        evaluation_seeds=list(int(s) for s in raw["evaluation_seeds"]),
        scenarios=list(str(s) for s in raw["scenarios"]),
        artifacts=ArtifactsConfig(
            model_checkpoint=str(arts["model_checkpoint"]),
            vec_normalize_stats=str(arts["vec_normalize_stats"]),
        ),
        normalisation=EvalNormalisationConfig(
            training=bool(norm["training"]),
            norm_reward=bool(norm["norm_reward"]),
        ),
        reporting=ReportingConfig(
            output_dir=str(rep["output_dir"]),
            save_tripinfo=bool(rep["save_tripinfo"]),
            metrics_to_collect=list(rep.get("metrics_to_collect") or []),
        ),
    )