import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from graphora_server.main import app
from graphora_server.auth import get_current_user_id
from graphora_server.services.transform.status_models import TransformationStage


pytestmark = pytest.mark.integration


@pytest.fixture
def transform_upload_setup(monkeypatch, tmp_path):
    from graphora_server.api import transform as transform_api
    from graphora_server.schemas.transform import ValidationResult

    recorded: dict = {}

    app.dependency_overrides[get_current_user_id] = lambda: "integration-user"
    monkeypatch.setattr(transform_api.settings, "UPLOAD_DIR", str(tmp_path))

    async def fake_validate(self, _file):
        return ValidationResult(is_valid=True, errors=[])

    async def fake_initialize(transform_id: str):
        recorded["initialized"] = transform_id

    async def fake_complete(transform_id: str, stage):
        recorded.setdefault("completed_stages", []).append(stage)

    async def fake_log_start(*_, **__):
        return "audit-log"

    async def fake_log_success(*_, **__):
        recorded["logged_success"] = True
        return None

    async def fake_run_transform_flow(
        transform_id: str,
        ontology_id: str,
        file_paths,
        metadata,
        user_id: str,
        audit_id: str,
        chunking_config=None,
    ):
        recorded["run_called"] = {
            "transform_id": transform_id,
            "ontology_id": ontology_id,
            "file_paths": file_paths,
            "metadata": metadata,
            "user_id": user_id,
            "audit_id": audit_id,
            "chunking_config": chunking_config,
        }

    monkeypatch.setattr(
        transform_api.FileValidator,
        "validate",
        fake_validate,
        raising=False,
    )
    monkeypatch.setattr(
        transform_api.progress_tracker,
        "initialize_transform",
        fake_initialize,
    )
    monkeypatch.setattr(
        transform_api.progress_tracker,
        "complete_stage",
        fake_complete,
    )
    monkeypatch.setattr(
        transform_api.audit_service, "log_operation_start", fake_log_start
    )
    monkeypatch.setattr(
        transform_api.audit_service, "log_operation_success", fake_log_success
    )
    monkeypatch.setattr(
        transform_api.audit_service, "log_operation_failure", fake_log_success
    )
    monkeypatch.setattr(transform_api, "run_transform_flow", fake_run_transform_flow)

    # B5-obs slice 2 preflight (commit 535f56d) + fail-closed
    # reviewer fix (HEAD): preflight hits Postgres for the budget
    # + spend reads, and a configured-but-degraded read returns
    # 503. Patch the helper to be a no-op so this integration
    # test exercises the happy path it was written for.
    async def fake_enforce_budget_preflight(_user_id):
        return None

    monkeypatch.setattr(
        transform_api,
        "enforce_budget_preflight",
        fake_enforce_budget_preflight,
    )

    yield recorded

    app.dependency_overrides.pop(get_current_user_id, None)


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_transform_upload_starts_background_flow(transform_upload_setup):
    recorded = transform_upload_setup

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [("files", ("sample.txt", b"hello world", "text/plain"))]
        response = await client.post(
            "/api/v1/transform/ontology-123/upload",
            files=files,
        )
        await asyncio.sleep(0)

    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["document_info"]["filename"] == "sample.txt"

    assert recorded["initialized"] == payload["id"]
    assert TransformationStage.UPLOAD in recorded["completed_stages"]
    assert recorded.get("logged_success") is True

    run_call = recorded.get("run_called")
    assert run_call is not None
    assert Path(run_call["file_paths"][0]).name == "sample.txt"
    assert Path(run_call["file_paths"][0]).exists()
    assert run_call["audit_id"] == "audit-log"
    assert run_call["user_id"] == "integration-user"
