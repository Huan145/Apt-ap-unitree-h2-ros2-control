import sys
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tests' / 'perception'))     # ros_stubs, helpers (shared)
sys.path.insert(0, str(Path(__file__).parent))             # navsim
sys.path.insert(0, str(root / 'nodes' / 'navigation'))

import ros_stubs  # noqa: E402

MSG = ros_stubs.install()


@pytest.fixture(autouse=True)
def reset_stubs():
    ros_stubs.Node.overrides = {}
    ros_stubs.Node.client_factory = None
    ros_stubs.Buffer.T = None
    yield
