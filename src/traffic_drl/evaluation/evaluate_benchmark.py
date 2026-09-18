"""Deterministic benchmark evaluation on VA or unlocked TE scenarios.

All three baselines — fixed-time, actuated, and DRL — are evaluated through
the same :func:`evaluate_controller` loop.  There is no special-case path for
the fixed-time baseline.

Typical usage
-------------
::

    from traffic_drl.evaluation.evaluate_benchmark import (
        evaluate_controller,
        evaluate_manifest,
        run_benchmark,
    )
    from traffic_drl.baselines import FixedTimeController, make_fixed_time_env
    from traffic_drl.config import load_eval_config

    cfg = load_eval_config("configs/evaluation/phase1_validation.yaml")
    manifest = ScenarioManifest.from_csv(cfg.manifest_path)
    controller = FixedTimeController()
    factory = lambda record, seed: make_fixed_time_env(record, env_config)
    results = evaluate_manifest(
        controller, factory, manifest,
        split=cfg.benchmark.split,
        seeds=cfg.evaluation_seeds,
        controller_name="fixed_time",
    )
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

import gymnasium as gym

from traffic_drl.contracts import (
    BenchmarkReport,
    Controller,
    EnvironmentFactory,
    EpisodeMetrics,
    ScenarioSource,
)
from traffic_drl.config import EvalConfig, load_eval_config


def evaluate_controller(
    controller: Controller,
    env: gym.Env,
    *,
    controller_name: str,
    scenario_id: str,
    seed: int,
    deterministic: bool = True,
    episodes: int = 1,
) -> list[EpisodeMetrics]:
    """Roll out one controller and collect standard traffic metrics."""
    results: list[EpisodeMetrics] = []

    for ep in range(episodes):
        if hasattr(controller, "reset") and callable(controller.reset):
            controller.reset()
            
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            if hasattr(controller, "predict"):
                action, _ = controller.predict(obs, deterministic=deterministic)
            elif hasattr(controller, "step"):
                action = controller.step(obs)
            else:
                action = env.action_space.sample()
                
            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
            else:
                obs, reward, terminated, info = step_ret
                truncated = False
                
            step_count += 1
            
        results.append(
            EpisodeMetrics(
                controller=controller_name,
                scenario_id=scenario_id,
                seed=seed,
                average_waiting_time=float(info.get("average_waiting_time", 0.0)),
                average_queue_length=float(info.get("average_queue_length", 0.0)),
                time_loss=float(info.get("time_loss", 0.0)),
                throughput=float(info.get("throughput", 0.0)),
                phase_switch_rate=float(info.get("phase_switch_rate", 0.0)),
                min_green_violations=int(info.get("min_green_violations", 0)),
            )
        )
        
    return results

def evaluate_manifest(
    controller: Controller,
    env_factory: EnvironmentFactory,
    manifest: ScenarioSource,
    *,
    split: str,
    seeds: Iterable[int],
    controller_name: str,
    deterministic: bool = True,
    episodes_per_scenario: int = 1,
    allow_te: bool = False,
) -> list[EpisodeMetrics]:
    """Evaluate every scenario in *split* with each seed in *seeds*."""
    normalized_split = split.strip().upper()
    if normalized_split == "TE" and not allow_te:
        raise PermissionError(
            "Held-out test split (TE) is locked until the model and manifest are formally frozen."
        )

    records = manifest.records_for_split(split)
    all_metrics: list[EpisodeMetrics] = []
    seeds_list = list(seeds)

    for record in records:
        for seed in seeds_list:
            env = env_factory(record, seed)
            try:
                metrics = evaluate_controller(
                    controller,
                    env,
                    controller_name=controller_name,
                    scenario_id=record.scenario_id,
                    seed=seed,
                    deterministic=deterministic,
                    episodes=episodes_per_scenario,
                )
                all_metrics.extend(metrics)
            finally:
                if hasattr(env, "close") and callable(env.close):
                    env.close()

    return all_metrics


def save_evaluation_results(
    records: Iterable[EpisodeMetrics],
    output_path: str | Path,
) -> None:
    """Persist raw per-episode records for later statistical analysis.

    Selects the serialiser from the file suffix:
    - ``.csv``: one row per ``EpisodeMetrics`` with a header.
    - ``.json``: a JSON array of objects with the same field names.

    Args:
        records: Typed episode metrics to serialise.
        output_path: Destination file.  The parent directory is created if it
            does not exist.

    TODO (SV3): decide whether to write CSV or JSON as the canonical output
    format and document it in the evaluation config.

    Returns:
        None.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()

    records_list = list(records)
    dict_records = [asdict(r) for r in records_list]

    if suffix == ".csv":
        fieldnames = [f.name for f in fields(EpisodeMetrics)]
        with dest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in dict_records:
                writer.writerow(r)
    elif suffix == ".json":
        with dest.open("w", encoding="utf-8") as f:
            json.dump(dict_records, f, indent=2)
    else:
        raise ValueError(
            f"Unsupported output file extension: '{suffix}'. Supported formats are '.csv' and '.json'."
        )


def run_benchmark(
    config: EvalConfig | str | Path = "configs/evaluation/phase1_validation.yaml",
) -> BenchmarkReport:
    """Run the full fixed-time, actuated, and DRL controller comparison.

    Loads the manifest, creates an environment factory for each controller,
    calls :func:`evaluate_manifest` for each, then calls
    :func:`~traffic_drl.evaluation.metrics.interpret_results` to produce the
    final report.

    Args:
        config: Typed evaluation configuration or path to its YAML file.

    TODO (SV3): use identical routes and seeds for every controller; produce
    95% confidence intervals; prevent TE access before the model is frozen.

    Returns:
        BenchmarkReport: Final typed comparison report.
    """
    if isinstance(config, (str, Path)):
        config = load_eval_config(config)

    raise NotImplementedError
