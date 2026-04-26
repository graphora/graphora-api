from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from graphora_server.services.transform.validators import FileValidator


@pytest.mark.asyncio
async def test_file_validator_accepts_allowed_mime(monkeypatch):
    validator = FileValidator()
    upload = UploadFile(filename="doc.pdf", file=BytesIO(b"%PDF-test"))

    monkeypatch.setattr(
        "graphora_server.services.transform.validators.magic.from_buffer",
        lambda _content, mime=True: "application/pdf",
    )

    result = await validator.validate(upload)

    assert result.is_valid is True
    assert result.errors == []


@pytest.mark.asyncio
async def test_file_validator_rejects_disallowed_mime(monkeypatch):
    validator = FileValidator()
    upload = UploadFile(filename="script.sh", file=BytesIO(b"#!/bin/bash"))

    monkeypatch.setattr(
        "graphora_server.services.transform.validators.magic.from_buffer",
        lambda _content, mime=True: "application/x-shellscript",
    )

    result = await validator.validate(upload)

    assert result.is_valid is False
    assert "not allowed" in result.errors[0]


@pytest.mark.asyncio
async def test_file_validator_rejects_files_over_limit(monkeypatch):
    validator = FileValidator()
    upload = UploadFile(filename="large.txt", file=BytesIO(b"0123456789ABCDEF"))

    monkeypatch.setattr(FileValidator, "MAX_FILE_SIZE", 8)
    monkeypatch.setattr(
        "graphora_server.services.transform.validators.magic.from_buffer",
        lambda _content, mime=True: "text/plain",
    )

    result = await validator.validate(upload)

    assert result.is_valid is False
    assert "exceeds" in result.errors[0]
