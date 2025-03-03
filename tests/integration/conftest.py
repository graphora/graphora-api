"""Fixtures for integration tests"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
import os
from typing import Generator

from app.main import app

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application"""
    with TestClient(app) as client:
        yield client 