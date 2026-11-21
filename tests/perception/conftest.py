import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))                                # ros_stubs
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'nodes' / 'perception'))

import ros_stubs  # noqa: E402

MSG = ros_stubs.install()


@pytest.fixture(autouse=True)
def reset_stubs():
    ros_stubs.Node.overrides = {}
    ros_stubs.Node.client_factory = None
    ros_stubs.Buffer.T = None
    yield
