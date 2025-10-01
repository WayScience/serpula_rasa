"""
Fixtures for image tests.
"""

import pathlib
import pytest

@pytest.fixture
def tmp_db(tmp_path: pathlib.Path) -> pathlib.Path:
    db_dir = tmp_path / "lance_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir