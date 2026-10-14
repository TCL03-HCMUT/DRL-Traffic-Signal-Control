from types import SimpleNamespace

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from traffic_drl.environment.wrappers import MultiScenarioWrapper
from traffic_drl.train.callbacks import TrafficMetricsCallback


class FixedScenarioSampler:
    def sample(self):
        return SimpleNamespace(scenario_id="TR-TEST", route_file="test.rou.xml")


def test_monitor_records_scenario_id_for_episode(tmp_path):
    env = gym.make("CartPole-v1", max_episode_steps=1)
    env = MultiScenarioWrapper(env, FixedScenarioSampler())
    env = Monitor(env, filename=str(tmp_path / "monitor"), info_keywords=("scenario_id",))

    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(0)

        assert terminated or truncated
        assert info["scenario_id"] == "TR-TEST"
        assert info["episode"]["scenario_id"] == "TR-TEST"
    finally:
        env.close()


def test_metrics_callback_logs_scenario_id_as_text():
    recorded = {}
    callback = TrafficMetricsCallback()
    callback.model = SimpleNamespace(logger=SimpleNamespace(record=lambda key, value: recorded.update({key: value})))
    callback.locals = {"infos": [{"episode": {}, "scenario_id": "TR-TEST"}]}

    assert callback._on_step()
    assert recorded["scenario/id"] == "TR-TEST"