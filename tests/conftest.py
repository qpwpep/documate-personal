from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture
def unavailable_official_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search-cache fixtures cannot make live documentation GET requests."""
    def get(url: str, **kwargs: object) -> requests.Response:
        raise requests.Timeout("Official page unavailable in this search-cache fixture")

    monkeypatch.setattr(requests, "get", get)
