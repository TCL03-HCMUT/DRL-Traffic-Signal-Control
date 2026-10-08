import os
import sys

# Optional: Set SUMO_HOME if it's not already in your environment variables
# os.environ["SUMO_HOME"] = "C:\\Program Files (x86)\\Eclipse\\Sumo" 

from traffic_drl.environment.make_env import make_dev_environment

def get_phase_strings(env):
    """Helper to extract the current phase string (e.g. 'ggrrr') for each traffic light."""
    phases = {}
    # Access the underlying TrafficSignal objects
    for ts_id, ts in env.unwrapped.traffic_signals.items():
        # Ask traci directly for the current string being executed
        phases[ts_id] = ts.sumo.trafficlight.getRedYellowGreenState(ts_id)
    return phases

def get_available_action_phases(env):
    """Helper to list all valid action indices and their corresponding green phase strings."""
    available = {}
    for ts_id, ts in env.unwrapped.traffic_signals.items():
        # ts.green_phases is the list of phases the agent can select
        available[ts_id] = {action_idx: p.state for action_idx, p in enumerate(ts.green_phases)}
    return available

def test_environment():
    print("Creating DEV-00 environment...")
    # make_dev_environment automatically loads the dev config and pilot manifest
    env = make_dev_environment(use_gui=False, seed=42)

    print("\n--- Initializing ---")
    obs, info = env.reset(seed=42)
    print(f"Available Actions: {get_available_action_phases(env)}")
    print(f"Initial Observation Shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    print(f"Initial Observation: {obs}")
    print(f"Initial Info: {info}")
    print(f"Initial Phases: {get_phase_strings(env)}")

    print("\n--- Stepping ---")
    num_steps = 15
    for step in range(num_steps):
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take the action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action Taken: {action}")
        print(f"  Phases: {get_phase_strings(env)}")
        print(f"  Observation Shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
        print(f"  Observation: {obs}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Info: {info}")

        if terminated or truncated:
            print("Episode ended early!")
            break

    print("\n--- Cleaning up ---")
    env.close()
    print("Done.")

if __name__ == "__main__":
    test_environment()
