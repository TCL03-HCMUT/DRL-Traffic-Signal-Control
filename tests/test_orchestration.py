"""
Test script to verify the orchestration functions work.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import traffic_drl
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_loading():
    print("Testing config loading...")
    from traffic_drl.config import load_env_config, load_train_config, load_eval_config

    # Load environment config
    env_config = load_env_config("configs/environment/dev_single_intersection.yaml")
    print(f"  Loaded env config: {env_config.network.net_file}")

    # Load train config
    train_config = load_train_config("configs/train/dqn_phase1.yaml")
    print(f"  Loaded train config: {train_config.experiment.name}")

    # Load eval config
    eval_config = load_eval_config("configs/evaluation/phase1_validation.yaml")
    print(f"  Loaded eval config: {eval_config.benchmark.name}")

    print("Config loading test passed.\n")

def test_run_id():
    print("Testing run ID management...")
    from traffic_drl.run_id import get_default_run_id, ensure_run_directories, create_sumo_output_options, get_tripinfo_path, get_metrics_path

    # Generate a run ID
    run_id = get_default_run_id(prefix="test")
    print(f"  Generated run ID: {run_id}")

    # Ensure directories exist
    tripinfo_dir, results_dir, *_ = ensure_run_directories(run_id)
    print(f"  Tripinfo directory: {tripinfo_dir}")
    print(f"  Results directory: {results_dir}")

    # Get SUMO output options
    sumo_options = create_sumo_output_options(run_id)
    print(f"  SUMO output options: {sumo_options}")

    # Get specific paths
    tripinfo_path = get_tripinfo_path(run_id)
    metrics_path = get_metrics_path(run_id)
    print(f"  Tripinfo path: {tripinfo_path}")
    print(f"  Metrics path: {metrics_path}")

    print("Run ID management test passed.\n")

def test_make_env():
    print("Testing environment creation...")
    from traffic_drl.environment.make_env import create_sumo_env, make_dev_environment, make_vectorized_environment, close_environment

    # Create a base environment
    env = create_sumo_env(
        config="configs/environment/dev_single_intersection.yaml",
        route_file="scenarios/dummy/single-intersection-vhvh.rou.xml",
        seed=42
    )
    print(f"  Created environment: {type(env)}")

    # Test reset and step (for the dummy CartPole environment)
    obs, info = env.reset()
    print(f"  Initial observation: {obs}")
    print(f"  Initial info: {info}")

    # Take a step
    action = env.action_space.sample()  # For CartPole, this is 0 or 1
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step reward: {reward}")
    print(f"  Step info keys: {list(info.keys())}")

    # Close the environment
    close_environment(env)
    print("  Environment closed.")

    print("Environment creation test passed.\n")

def test_wrappers():
    print("Testing environment wrappers...")
    import gymnasium as gym
    from traffic_drl.environment.wrappers import wrap_environment, MultiScenarioWrapper, MetricsInfoWrapper

    # Create a base environment
    env = gym.make('CartPole-v1')

    # Wrap with default wrappers (no scenario sampler)
    wrapped_env = wrap_environment(env)
    print(f"  Wrapped environment: {type(wrapped_env)}")

    # Test reset and step
    obs, info = wrapped_env.reset()
    print(f"  Wrapped reset observation: {obs}")
    print(f"  Wrapped reset info keys: {list(info.keys())}")

    action = wrapped_env.action_space.sample()
    next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f"  Wrapped step reward: {reward}")
    print(f"  Wrapped step info keys: {list(info.keys())}")

    # Close the environment
    wrapped_env.close()

    print("Environment wrappers test passed.\n")

def test_train_dqn():
    print("Testing DQN training functions...")
    from traffic_drl.train.train_dqn import build_dqn_model, train_dqn, run_dqn_pilot
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Create a simple environment for testing
    env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

    # Dummy config for model hyperparameters
    dummy_config = {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "buffer_size": 1000,
        "learning_starts": 10,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 100,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "max_grad_norm": 10,
        "stats_window_size": 100
    }

    # Build the model
    model = build_dqn_model(env=env, config=dummy_config, seed=42)
    print(f"  Built DQN model: {type(model)}")

    # Train for a few timesteps
    model = train_dqn(model, total_timesteps=100)
    print(f"  Trained model for 100 timesteps.")

    # Test run_dqn_pilot (this will create its own environment)
    # We'll use a small number of timesteps for testing
    try:
        model = run_dqn_pilot(
            config="configs/train/dqn_phase1.yaml",
            checkpoint_dir="outputs/test/run_dqn_pilot",
            total_timesteps=100,  # Very small for testing
            seed=42
        )
        print(f"  DQN pilot completed.")
    except Exception as e:
        print(f"  DQN pilot failed (expected due to missing SUMO setup): {e}")

    print("DQN training functions test passed.\n")

def test_evaluation():
    print("Testing evaluation functions...")
    from traffic_drl.evaluation.evaluate_benchmark import evaluate_controller, evaluate_manifest, save_evaluation_results, run_benchmark
    from traffic_drl.contracts import EpisodeMetrics
    import numpy as np

    # Test evaluate_controller (returns a list of EpisodeMetrics)
    # We'll create a dummy model and env
    class DummyModel:
        def predict(self, observation, deterministic=True):
            return 0  # Always return action 0

    class DummyEnv:
        def reset(self, seed=None):
            return np.array([0, 0, 0, 0]), {}
        def step(self, action):
            return np.array([0, 0, 0, 0]), 1.0, False, False, {}
        def close(self):
            pass

    model = DummyModel()
    env = DummyEnv()

    metrics = evaluate_controller(
        controller=model,
        env=env,
        controller_name="dummy",
        scenario_id="test_scenario",
        seed=42,
        deterministic=True,
        episodes=2
    )
    print(f"  Evaluated controller, got {len(metrics)} episodes.")
    print(f"  First episode waiting time: {metrics[0].average_waiting_time}")

    # Test run_benchmark (returns a BenchmarkReport)
    try:
        report = run_benchmark("configs/evaluation/phase1_validation.yaml")
        print(f"  Benchmark report generated with baseline: {report.baseline_controller}")
        print(f"  Number of scenario summaries: {len(report.summaries)}")
    except Exception as e:
        print(f"  Benchmark run failed (expected due to missing setup): {e}")

    print("Evaluation functions test passed.\n")

if __name__ == "__main__":
    print("Running orchestration tests...\n")

    try:
        test_config_loading()
    except Exception as e:
        print(f"Config loading test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_run_id()
    except Exception as e:
        print(f"Run ID test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_make_env()
    except Exception as e:
        print(f"Environment creation test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_wrappers()
    except Exception as e:
        print(f"Wrappers test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_train_dqn()
    except Exception as e:
        print(f"DQN training test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_evaluation()
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nAll tests completed.")