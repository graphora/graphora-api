import json
from pathlib import Path

import pytest

from scripts.openapi_snapshot import stub_dependencies


@pytest.fixture(scope="module")
def openapi_schema():
    stub_dependencies()
    from app.main import app

    return app.openapi()


def test_openapi_schema_matches_snapshot(openapi_schema):
    snapshot_path = Path("tests/snapshots/openapi.json")
    expected = json.loads(snapshot_path.read_text())
    assert openapi_schema == expected, (
        "OpenAPI schema has changed. Run `make openapi-snapshot` and commit the updated "
        "tests/snapshots/openapi.json file if this change is intentional."
    )
