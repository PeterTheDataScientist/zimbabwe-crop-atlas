from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.observation import Grid, Observation, ObservationStack, Provenance

SHAPE = (40, 50)


@pytest.fixture
def grid() -> Grid:
    # UTM 36S, the zone Harare falls in. 20 m pixels.
    return Grid(
        transform=(20.0, 0.0, 300_000.0, 0.0, -20.0, 8_100_000.0),
        crs="EPSG:32736",
        shape=SHAPE,
    )


def make_obs(
    grid: Grid,
    sensor: str,
    when: dt.datetime,
    *,
    bands: dict[str, np.ndarray] | None = None,
    valid: np.ndarray | None = None,
    scene_id: str | None = None,
) -> Observation:
    if valid is None:
        valid = np.ones(grid.shape, dtype=bool)
    if bands is None:
        rng = np.random.default_rng(abs(hash((sensor, when))) % 2**32)
        bands = {
            "red": rng.uniform(0.04, 0.09, grid.shape).astype(np.float32),
            "nir": rng.uniform(0.25, 0.45, grid.shape).astype(np.float32),
        }
    bands = {
        k: np.where(valid, v, np.float32(np.nan)).astype(np.float32)
        for k, v in bands.items()
    }
    return Observation(
        grid=grid,
        bands=bands,
        valid=valid,
        provenance=Provenance(
            sensor=sensor,
            scene_id=scene_id or f"{sensor}-{when:%Y%m%d}",
            acquired=when,
        ),
    )


@pytest.fixture
def stack(grid: Grid) -> ObservationStack:
    """A small mixed-sensor stack with realistic partial cloud."""
    rng = np.random.default_rng(7)
    obs = []
    for i, (sensor, day) in enumerate(
        [
            ("s2", 4),
            ("l8", 6),
            ("s2", 9),
            ("l9", 14),
            ("s2", 19),
            ("l8", 22),
        ]
    ):
        # Cloud as contiguous blocks, not salt and pepper: real cloud is
        # spatially correlated and a test with random pixel dropout would let
        # a compositor that cannot handle a solid hole pass.
        valid = np.ones(SHAPE, dtype=bool)
        y0 = (i * 7) % (SHAPE[0] - 12)
        x0 = (i * 11) % (SHAPE[1] - 12)
        valid[y0 : y0 + 12, x0 : x0 + 12] = False
        obs.append(
            make_obs(grid, sensor, dt.datetime(2025, 1, day, 8, 30), valid=valid)
        )
    _ = rng
    return ObservationStack.build(grid, obs)
