# Training Orchestration & Checkpointing Guide

This guide explains how to leverage the checkpointing system in `traffic_drl`. The system ensures that models, replay buffers, normalization statistics, RNG states, and metadata are saved cleanly into dedicated directories, enabling fully reproducible training resumption.

## Checkpoint Structure

Every training run maintains exactly **two** checkpoint directories:

```
outputs/checkpoints/my_experiment/
├── last/                    # Overwritten every save_freq steps
│   ├── model.zip            # SB3 model weights + optimizer + hyperparams
│   ├── replay_buffer.pkl    # DQN replay memory (only for DQN)
│   ├── vec_normalize.pkl    # VecNormalize running statistics
│   ├── rng_state.pkl        # numpy / torch / python RNG states
│   └── resume_info.json     # Human-readable metadata
└── best/                    # Overwritten only when reward improves
    ├── model.zip
    ├── replay_buffer.pkl
    ├── vec_normalize.pkl
    ├── rng_state.pkl
    └── resume_info.json
```

The **best** checkpoint uses episode reward mean as the primary metric (higher is better). On a tie, average waiting time (lower is better) breaks the tie.

## 1. Starting a New Training Run

### DQN Example

```python
from pathlib import Path
from traffic_drl.train.callbacks import RobustCheckpointCallback, TrafficMetricsCallback
from traffic_drl.train.train_dqn import build_dqn_model, train_dqn
from stable_baselines3.common.callbacks import CallbackList

save_dir = Path("outputs/checkpoints/my_experiment")

# Checkpoint callback: saves last/ every 10k steps, best/ on improvement
checkpoint_cb = RobustCheckpointCallback(
    save_freq=10_000,
    save_dir=save_dir,
    save_replay_buffer=True,
    save_vec_normalize=True,
    verbose=1,
)

# Traffic metrics callback: logs metrics to TensorBoard
metrics_cb = TrafficMetricsCallback()

# Combine callbacks
callbacks = CallbackList([checkpoint_cb, metrics_cb])

# Build and train
model = build_dqn_model(env, config)
trained_model = train_dqn(
    model,
    total_timesteps=100_000,
    callback=callbacks,
)
```

### PPO Example

The same callback works identically for PPO — it auto-detects that PPO has no replay buffer:

```python
from traffic_drl.train.train_ppo import build_ppo_model, train_ppo

model = build_ppo_model(env, config)
trained_model = train_ppo(
    model,
    total_timesteps=100_000,
    callback=callbacks,
)
```

## 2. Resuming an Interrupted Training Run

If training crashes or is preempted, resume from the `last/` checkpoint:

### DQN Resumption

```python
from traffic_drl.train.train_dqn import load_dqn_checkpoint
from traffic_drl.train.checkpointing import read_resume_info

checkpoint_dir = Path("outputs/checkpoints/my_experiment/last")

# Read metadata (optional, for logging)
info = read_resume_info(checkpoint_dir)
print(f"Resuming {info.model_class} from step {info.num_timesteps}")

# Load everything: model + replay buffer + VecNormalize + RNG state
# All auto-detected from the bundle directory!
loaded_model = load_dqn_checkpoint(
    checkpoint_dir,
    env=fresh_vec_env,      # A fresh DummyVecEnv (VecNormalize is loaded on top)
    training=True,          # Keep VecNormalize stats updating
    restore_rng_state=True, # Restore numpy/torch/python RNG for exact resumption
)

# Resume training (reset_num_timesteps=False is critical!)
trained_model = train_dqn(
    loaded_model,
    total_timesteps=50_000,
    callback=callbacks,
    reset_num_timesteps=False,
)
```

### PPO Resumption

```python
from traffic_drl.train.train_ppo import load_ppo_checkpoint

loaded_model = load_ppo_checkpoint(
    checkpoint_dir,
    env=fresh_vec_env,
    training=True,
    restore_rng_state=True,
)
```

## 3. Evaluation from a Checkpoint

For evaluation, load the **best** checkpoint with frozen normalization:

```python
from stable_baselines3 import DQN

checkpoint_dir = Path("outputs/checkpoints/my_experiment/best")

# For evaluation: no RNG restore needed, freeze VecNormalize
from traffic_drl.train.checkpointing import load_checkpoint

loaded_model = load_checkpoint(
    DQN,
    checkpoint_dir,
    env=eval_vec_env,
    training=False,         # Freeze VecNormalize stats
    norm_reward=False,      # Don't normalize rewards during eval
    restore_replay_buffer=False,
    restore_rng_state=False,
)

# Or use the evaluation env factory pattern:
def eval_env_factory(record, seed):
    return make_vectorized_environment(
        config=config_path,
        route_file=record.route_file,
        run_id="eval_run",
        base_dir="outputs/eval",
        vec_normalize=str(checkpoint_dir / "vec_normalize.pkl"),
        training=False,
        norm_reward=False,
        seed=seed,
        wrap=True,
    )
```

## 4. What Gets Saved and Restored

| Artifact | File | DQN | PPO | Purpose |
|---|---|---|---|---|
| Model weights | `model.zip` | ✅ | ✅ | Policy, optimizer state, hyperparameters |
| Replay buffer | `replay_buffer.pkl` | ✅ | ❌ | Off-policy experience memory |
| VecNormalize | `vec_normalize.pkl` | ✅ | ✅ | Observation/reward running mean & variance |
| RNG state | `rng_state.pkl` | ✅ | ✅ | numpy, torch (CPU+CUDA), python random |
| Resume info | `resume_info.json` | ✅ | ✅ | Timesteps, model class, best metric value |
