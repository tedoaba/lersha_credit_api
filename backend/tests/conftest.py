"""Pytest configuration and shared fixtures for backend tests."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_farmer_df() -> pd.DataFrame:
    """Return a synthetic 3-row farmer DataFrame matching the raw schema."""
    return pd.DataFrame({
        "farmer_uid": ["F-001", "F-002", "F-003"],
        "first_name": ["Abebe", "Bekele", "Chaltu"],
        "middle_name": ["Hailu", "Girma", "Tigist"],
        "last_name": ["Woldemariam", "Abebe", "Teshome"],
        "gender": ["Male", "Male", "Female"],
        "age": [32, 45, 28],
        "family_size": [5, 7, 4],
        "estimated_income": [12000.0, 18000.0, 9000.0],
        "estimated_income_another_farm": [2000.0, 3000.0, 1000.0],
        "estimated_expenses": [4000.0, 6000.0, 3000.0],
        "estimated_cost": [2000.0, 3000.0, 1500.0],
        "agricultureexperience": [8, 15, 5],
        "hasmemberofmicrofinance": [1, 1, 0],
        "hascooperativeassociation": [0, 1, 1],
        "agriculturalcertificate": [1, 0, 0],
        "hascommunityhealthinsurance": [1, 1, 0],
        "farmsizehectares": [2.5, 4.0, 1.5],
        "expectedyieldquintals": [30.0, 50.0, 18.0],
        "seedquintals": [3.0, 5.0, 2.0],
        "ureafertilizerquintals": [1.5, 2.5, 1.0],
        "dapnpsfertilizerquintals": [1.0, 2.0, 0.5],
        "value_chain": ["maize", "wheat", "teff"],
        "total_farmland_size": [3.0, 5.0, 2.0],
        "land_size": [2.5, 4.0, 1.5],
        "childrenunder12": [2, 3, 1],
        "elderlymembersover60": [1, 0, 0],
        "maincrops": ["maize", "wheat", "teff"],
        "lastyearaverageprice": [500.0, 600.0, 450.0],
        "decision": ["Eligible", "Eligible", "Review"],
    })


@pytest.fixture
def api_client(monkeypatch):
    """Return a FastAPI TestClient wrapping create_app().

    Monkeypatches config to avoid requiring real DB/ChromaDB for API tests.
    Import is deferred so config loading only happens when the fixture is requested.
    """
    import os

    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-pro")
    os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
    os.environ.setdefault("DB_URI", "sqlite:///:memory:")

    from backend.main import create_app

    app = create_app()
    return TestClient(app)
