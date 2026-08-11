import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine import api_server


def test_rejected_challenger_backoff_escalates_and_deploy_resets(tmp_path, monkeypatch):
    state_path = tmp_path / "retraining_state.json"
    monkeypatch.setattr(api_server, "RETRAIN_STATE_FILE", str(state_path))
    now = datetime.now()
    snapshot = {
        "fingerprint": "abc123",
        "label_count": 123,
        "positive_count": 70,
        "negative_count": 53,
        "training_end": "2026-08-04 00:00:00",
    }

    api_server.save_retraining_state(
        now, outcome="trained_not_better", training_data=snapshot
    )
    first = json.loads(state_path.read_text(encoding="utf-8"))
    assert first["rejection_streak"] == 1
    assert first["training_data_fingerprint"] == "abc123"
    assert 11.9 <= (
        api_server.parse_datetime_safe(first["backoff_until"]) - datetime.now()
    ).total_seconds() / 3600.0 <= 12.0

    api_server.save_retraining_state(
        now, outcome="trained_not_better", training_data=snapshot
    )
    second = json.loads(state_path.read_text(encoding="utf-8"))
    assert second["rejection_streak"] == 2
    assert 23.9 <= (
        api_server.parse_datetime_safe(second["backoff_until"]) - datetime.now()
    ).total_seconds() / 3600.0 <= 24.0

    api_server.save_retraining_state(now, outcome="deployed", training_data=snapshot)
    deployed = json.loads(state_path.read_text(encoding="utf-8"))
    assert deployed["rejection_streak"] == 0
    assert deployed["backoff_until"] is None


def test_retraining_stdout_parsers_extract_outcome_and_snapshot():
    stdout = "\n".join([
        'TRAINING_DATA_SNAPSHOT={"fingerprint":"f00d","label_count":42}',
        "重训练结果: trained_not_better (exit code: 0)",
    ])

    assert api_server._parse_retraining_outcome(stdout) == "trained_not_better"
    assert api_server._parse_training_data_snapshot(stdout) == {
        "fingerprint": "f00d",
        "label_count": 42,
    }
