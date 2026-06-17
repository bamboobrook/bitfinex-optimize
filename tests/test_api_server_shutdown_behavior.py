import asyncio
import signal
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_api_server_import_stubs(monkeypatch):
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda func: func

        def post(self, *args, **kwargs):
            return lambda func: func

        def on_event(self, *args, **kwargs):
            return lambda func: func

    class BackgroundTasks:
        def add_task(self, *args, **kwargs):
            pass

    fastapi.FastAPI = FastAPI
    fastapi.BackgroundTasks = BackgroundTasks
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.FileResponse = object
    fastapi_responses.JSONResponse = object
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses)

    fastapi_staticfiles = types.ModuleType("fastapi.staticfiles")
    fastapi_staticfiles.StaticFiles = object
    monkeypatch.setitem(sys.modules, "fastapi.staticfiles", fastapi_staticfiles)

    loguru = types.ModuleType("loguru")

    class Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    loguru.logger = Logger()
    monkeypatch.setitem(sys.modules, "loguru", loguru)

    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)


def test_run_full_pipeline_treats_prediction_sigterm_as_shutdown(monkeypatch):
    _install_api_server_import_stubs(monkeypatch)
    sys.modules.pop("ml_engine.api_server", None)
    import ml_engine.api_server as api_server

    statuses = []
    logged_errors = []
    calls = []

    async def fake_download_with_retry(cwd):
        return True

    async def fake_run_subprocess(cmd, cwd, timeout, step_name):
        calls.append(step_name)
        if step_name in {"Order Validation", "Retraining Check"}:
            return "", "", 0
        if step_name == "Prediction":
            return "", "terminated by service shutdown", -signal.SIGTERM
        raise AssertionError(step_name)

    monkeypatch.setattr(api_server, "_download_with_retry", fake_download_with_retry)
    monkeypatch.setattr(api_server, "_run_subprocess_with_timeout", fake_run_subprocess)
    monkeypatch.setattr(api_server, "update_status", lambda *args: statuses.append(args))
    monkeypatch.setattr(api_server.logger, "error", lambda msg: logged_errors.append(msg))

    asyncio.run(api_server.run_full_pipeline())

    assert calls == ["Order Validation", "Retraining Check", "Prediction"]
    assert not any(status[0] == "error" for status in statuses)
    assert not logged_errors
