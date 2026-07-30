from types import SimpleNamespace

from eqmarl.callbacks import ActiveSLAMPilotMonitor, AlgorithmModelCheckpoint


def test_model_checkpoint_formats_episode_placeholder(tmp_path):
    saved = []
    algorithm = SimpleNamespace(
        save_model=lambda name, path, weights_only: saved.append((name, path, weights_only))
    )
    callback = AlgorithmModelCheckpoint(
        "actor", tmp_path / "actor-{episode}.weights.h5", save_freq=1
    )
    callback.algorithm = algorithm
    callback.on_episode_end(0)
    assert saved == [("actor", str(tmp_path / "actor-1.weights.h5"), True)]


def test_pilot_monitor_detects_sustained_turn_collapse():
    record = {
        "left_action_fraction": 0.49,
        "right_action_fraction": 0.49,
        "forward_action_fraction": 0.02,
        "coverage": 0.5,
    }
    algorithm = SimpleNamespace(
        episode_metrics_history=[record] * 4,
        stop_training=False,
        stop_reason=None,
    )
    callback = ActiveSLAMPilotMonitor(window=4, verbose=False)
    callback.algorithm = algorithm
    callback.on_episode_end(3)
    assert algorithm.stop_training
    assert "turn collapse" in algorithm.stop_reason
