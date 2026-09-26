from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ingest import load_capture
from src.run import main
from tests.conftest import SINGLE_ROOM


def test_load_mini_capture(mini_capture: Path):
    cap = load_capture(mini_capture)
    assert len(cap.poses) == 2
    assert cap.n_depth == 2
    assert cap.n_confidence == 2
    assert cap.K.shape == (3, 3)
    assert cap.rgb_size == (64, 48)
    assert cap.poses[0].frame == "000000"
    assert cap.poses[1].t[0] == pytest.approx(1.0)


def test_cli_inspect_mini(mini_capture: Path, capsys):
    assert main([str(mini_capture), "--inspect"]) == 0
    out = capsys.readouterr().out
    assert "poses: 2" in out
    assert "depth pngs: 2" in out


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="founder dump not on disk")
def test_inspect_single_room_counts():
    cap = load_capture(SINGLE_ROOM)
    assert len(cap.poses) == cap.n_depth == cap.n_confidence
    assert cap.rgb_size == (1920, 1440)
    assert cap.n_depth > 1000
    xyz = [p.t for p in cap.poses]
    span = np.stack(xyz).max(0) - np.stack(xyz).min(0)
    assert span[1] < 1.0  # Y is up: walk barely changes height
