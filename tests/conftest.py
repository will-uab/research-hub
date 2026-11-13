import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg2():
    """The example v2 interests.yaml shipped with the repo (your own config/interests.yaml is never used by the tests)."""
    from litsearch.core.config import load_interests
    return load_interests(ROOT / "config" / "interests.example.yaml")


@pytest.fixture(scope="session")
def cfg(cfg2):
    """v1-shaped literature view of the real config (what queries/scoring/digest consume)."""
    from litsearch.core.config import to_literature_cfg
    return to_literature_cfg(cfg2)


@pytest.fixture
def db():
    from litsearch.core.db import DB
    d = DB(":memory:")
    yield d
    d.close()
