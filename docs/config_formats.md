# Format cho các file config được sử dụng trong code

Các file cấu hình được tạo theo đúng thư mục sau:

- `configs/env/`: cấu hình mạng, đèn, thời gian và SUMO.
- `configs/train/`: cấu hình DQN/PPO và ngân sách huấn luyện.
- `configs/evaluation/`: cấu hình validation/benchmark.

Đọc YAML bằng `yaml.safe_load`, kiểm tra các trường bắt buộc trước khi truyền tham số con vào `make_env.py`, `train_dqn.py`, `train_ppo.py` hoặc `evaluate_benchmark.py`. Không truyền toàn bộ YAML mapping trực tiếp vào constructor của SB3. Đường dẫn trong YAML là đường dẫn tương đối từ thư mục gốc repository và phải dùng dấu `/`.

Bản đầy đủ về nơi tạo file, file nào đọc chúng, và format đầu vào/đầu ra nằm ở [`interfaces_and_formats.md`](interfaces_and_formats.md).

# 1. Training config

```yaml
# configs/train/train_config.yaml
experiment:
  name: "[experiment_name]"
  seed: [int]
  device: "[cpu_or_cuda]"
  split_allowed: "[TR_or_other_split]"

environment:
  env_config_path: "[path/to/env_config.yaml]"
  manifest_path: "[path/to/manifest.csv]"
  num_seconds: [int]
  delta_time: [int]
  min_green: [int]
  yellow_time: [int]
  additional_sumo_cmd: "[sumo_cli_flags]"

reward:
  type: "[reward_function_type]"
  params:
    param_1: [float]
    param_2: [float]
    param_n: [float]
  wrappers:
    use_flicker_penalty: [true_or_false]
    flicker_lambda: [float]
    flicker_weight: [float]

normalization:
  norm_obs: [true_or_false]
  norm_reward: [true_or_false]
  clip_obs: [float]
  clip_reward: [float]
  gamma: [float]

model_hyperparameters:
  policy: "[PolicyType]"
  learning_rate: [float]
  n_steps: [int]
  batch_size: [int]
  n_epochs: [int]
  gamma: [float]
  gae_lambda: [float]
  clip_range: [float]
  ent_coef: [float]
  vf_coef: [float]
  max_grad_norm: [float]

training_control:
  total_timesteps: [int]
  save_freq: [int]
  eval_freq: [int]
  log_csv: [true_or_false]
```

# 2. Evaluation/Testing config

```yaml
# configs/evaluation/eval_config.yaml
benchmark:
  name: "[benchmark_run_name]"
  split: "[TE_or_VA]"
  deterministic: [true_or_false]
  episodes_per_scenario: [int]
  manifest_path: "[path/to/manifest.csv]"
  checksum_file: "[path/to/checksums.sha256]"

evaluation_seeds:
  - [seed_1]
  - [seed_2]
  - [seed_n]

scenarios:
  - "[scenario_id_1]"
  - "[scenario_id_2]"
  - "[scenario_id_n]"

artifacts:
  model_checkpoint: "[path/to/checkpoint.zip]"
  vec_normalize_stats: "[path/to/vec_normalize.pkl]"

normalization:
  training: false
  norm_reward: false
  clip_obs: [float]

reporting:
  output_dir: "[path/to/output_directory/]"
  save_tripinfo: [true_or_false]
  metrics_to_collect:
    - "[metric_name_1]"
    - "[metric_name_2]"
    - "[metric_name_n]"
```

# 3. Enviroment config

```yaml
# configs/env/env_config.yaml
network:
  net_file: "[path/to/network.net.xml]"
  route_files:
    - "[path/to/route_file_1.rou.xml]"
    - "[path/to/route_file_2.rou.xml]"
    - "[path/to/route_file_n.rou.xml]"
  additional_files:
    - "[path/to/additional_file_1.add.xml]"
    - "[path/to/additional_file_2.add.xml]"

traffic_light:
  ts_id: "[junction_node_id]"
  single_agent: [true_or_false]

timing:
  num_seconds: [int]
  delta_time: [int]
  min_green: [int]
  yellow_time: [int]

sumo_options:
  use_gui: [true_or_false]
  lateral_resolution: [float]
  additional_sumo_cmd: "[sumo_cli_flags]"
  tripinfo_output: "outputs/tripinfo/[run_id]/tripinfo.xml"
  emission_output: "outputs/tripinfo/[run_id]/emissions.xml"

observation_bounds:
  max_queue_per_lane: [float]
  max_time_in_phase: [float]
```
