"""Placeholder test to verify test infrastructure works."""

import pytest


@pytest.mark.unit
def test_import_airelab() -> None:
    import airelab

    assert airelab.__version__ == "0.1.0"
