import json
import time
from datetime import datetime

import scripts.health_check as health_check


def test_check_service_falls_back_to_http_when_systemd_bus_unavailable(monkeypatch):
    calls = []

    def fake_run_cmd(cmd):
        calls.append(cmd)
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return "Failed to connect to bus: Operation not permitted", 1
        if cmd[:2] == ["curl", "-sS"]:
            return '{"api_online": true}', 0
        raise AssertionError(cmd)

    monkeypatch.setattr(health_check, "run_cmd", fake_run_cmd)

    assert health_check.check_service() is True
    assert any(cmd[:2] == ["curl", "-sS"] for cmd in calls)


def test_check_service_retries_http_during_restart_window(monkeypatch):
    calls = []
    sleeps = []

    def fake_run_cmd(cmd):
        calls.append(cmd)
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return "Failed to connect to bus: Operation not permitted", 1
        if cmd[:2] == ["curl", "-sS"] and len([c for c in calls if c[:2] == ["curl", "-sS"]]) == 1:
            return "curl: (7) Failed to connect", 7
        if cmd[:2] == ["curl", "-sS"]:
            return '{"api_online": true}', 0
        raise AssertionError(cmd)

    monkeypatch.setattr(health_check, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    assert health_check.check_service() is True
    assert len([cmd for cmd in calls if cmd[:2] == ["curl", "-sS"]]) == 2
    assert sleeps


def test_check_service_uses_loopback_ip_for_api_fallback(monkeypatch):
    calls = []

    def fake_run_cmd(cmd):
        calls.append(cmd)
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return "Failed to connect to bus: Operation not permitted", 1
        if cmd[:2] == ["curl", "-sS"]:
            return '{"api_online": true}', 0
        raise AssertionError(cmd)

    monkeypatch.setattr(health_check, "run_cmd", fake_run_cmd)

    assert health_check.check_service() is True
    assert ["curl", "-sS", "http://127.0.0.1:5000/status"] in calls


def test_check_prediction_output_flags_stale_result(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result_path = data_dir / "optimal_combination.json"
    result_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
                "stale_data": True,
                "stale_reason": "partial_stale_data_skipped",
                "stale_issues": [{"currency": "fUSD", "period": 8}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(health_check, "DATA_DIR", str(data_dir))

    assert health_check.check_prediction_output() is False


def test_check_prediction_output_allows_supported_partial_stale_with_recommendations(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result_path = data_dir / "optimal_combination.json"
    result_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
                "stale_data": True,
                "stale_reason": "partial_stale_data_skipped",
                "stale_issues": [{"currency": "fUST", "period": 15}],
                "recommendations": [{"currency": "fUSD", "period": 120, "rate": 12.3}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(health_check, "DATA_DIR", str(data_dir))

    assert health_check.check_prediction_output() is True


def test_check_recent_errors_flags_nan_prediction(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    app_log = log_dir / "ml_optimizer.log"
    app_log.write_text(
        "\n".join(
            [
                "2026-06-16 10:00:00.000 | INFO | Starting CLOSED-LOOP Optimization Pipeline",
                "2026-06-16 10:01:00.000 | ERROR | NaN detected in final_rate for fUST-90",
                "2026-06-16 10:01:00.001 | ERROR | Prediction failed: NaN in final_rate for fUST-90",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(health_check, "APP_LOG", str(app_log))

    assert health_check.check_recent_errors() is False


def test_check_recent_errors_allows_partial_stale_gate_without_hard_errors(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    app_log = log_dir / "ml_optimizer.log"
    app_log.write_text(
        "\n".join(
            [
                "2026-06-16 10:00:00.000 | INFO | Starting CLOSED-LOOP Optimization Pipeline",
                "2026-06-16 10:01:00.000 | WARNING | Prediction finished with stale_data gate: partial_stale_data_skipped",
                "2026-06-16 10:01:00.001 | INFO | <<< ✅ CLOSED-LOOP Pipeline completed successfully!",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(health_check, "APP_LOG", str(app_log))

    assert health_check.check_recent_errors() is True


def test_check_recent_errors_allows_download_timeout_when_db_fallback_and_pipeline_succeeds(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    app_log = log_dir / "ml_optimizer.log"
    app_log.write_text(
        "\n".join(
            [
                "2026-06-16 12:00:00.000 | INFO | Starting CLOSED-LOOP Optimization Pipeline",
                "2026-06-16 12:10:00.000 | ERROR | TIMEOUT: Data Download exceeded 600s, killing process group",
                "2026-06-16 12:20:00.000 | INFO | ✅ Existing DB data is fresh enough, continuing pipeline despite download failure",
                "2026-06-16 12:26:58.001 | WARNING | Prediction finished with stale_data gate: partial_stale_data_skipped",
                "2026-06-16 12:26:58.001 | INFO | <<< ✅ CLOSED-LOOP Pipeline completed successfully!",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(health_check, "APP_LOG", str(app_log))

    assert health_check.check_recent_errors() is True


def test_download_timeout_budget_allows_slow_bitfinex_refresh():
    from ml_engine import api_server

    assert api_server.TIMEOUT_DOWNLOAD >= 1200
