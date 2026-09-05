from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.observation import (
    BandUnavailable,
    Grid,
    Observation,
    ObservationStack,
    Provenance,
    stack_bands,
)
from tests.conftest import make_obs


class TestGrid:
    def test_pixel_area_from_transform(self, grid: Grid) -> None:
        assert grid.pixel_area_m2 == pytest.approx(400.0)

    def test_geographic_crs_refuses_to_give_an_area(self) -> None:
        """Degrees squared is not an area and must not be returned as one.

        This is the bug that produces a district reported in hectares that is
        wrong by five orders of magnitude and still looks like a number.
        """
        g = Grid((0.0002, 0, 30.0, 0, -0.0002, -17.0), "EPSG:4326", (10, 10))
        with pytest.raises(ValueError, match="projected CRS"):
            _ = g.pixel_area_m2

    def test_same_as_tolerates_float_noise_but_not_a_shift(self, grid: Grid) -> None:
        nudged = Grid(
            tuple(x + 1e-9 for x in grid.transform), grid.crs, grid.shape  # type: ignore[arg-type]
        )
        assert grid.same_as(nudged)
        shifted = Grid(
            (20.0, 0.0, 300_020.0, 0.0, -20.0, 8_100_000.0), grid.crs, grid.shape
        )
        assert not grid.same_as(shifted)


class TestObservation:
    def test_rejects_non_canonical_band_name(self, grid: Grid) -> None:
        with pytest.raises(ValueError, match="not a canonical band name"):
            Observation(
                grid=grid,
                bands={"B04": np.zeros(grid.shape, np.float32)},
                valid=np.ones(grid.shape, bool),
                provenance=Provenance("s2", "x", dt.datetime(2025, 1, 1)),
            )

    def test_rejects_wrong_dtype(self, grid: Grid) -> None:
        """float64 bands double memory for no gain and hide a missing cast."""
        with pytest.raises(TypeError, match="float32"):
            Observation(
                grid=grid,
                bands={"red": np.zeros(grid.shape, np.float64)},
                valid=np.ones(grid.shape, bool),
                provenance=Provenance("s2", "x", dt.datetime(2025, 1, 1)),
            )

    def test_rejects_mask_shape_mismatch(self, grid: Grid) -> None:
        with pytest.raises(ValueError, match="does not match grid"):
            Observation(
                grid=grid,
                bands={},
                valid=np.ones((3, 3), bool),
                provenance=Provenance("s2", "x", dt.datetime(2025, 1, 1)),
            )

    def test_band_unavailable_names_what_is_there(self, grid: Grid) -> None:
        obs = make_obs(grid, "l8", dt.datetime(2025, 1, 1))
        with pytest.raises(BandUnavailable, match="rededge1"):
            obs.band("rededge1")
        assert not obs.has("rededge1")
        assert obs.has("red", "nir")

    def test_masked_to_narrows_and_does_not_mutate(self, grid: Grid) -> None:
        obs = make_obs(grid, "s2", dt.datetime(2025, 1, 1))
        before = obs.valid.sum()
        keep = np.zeros(grid.shape, bool)
        keep[:10, :10] = True
        narrowed = obs.masked_to(keep)
        assert narrowed.valid.sum() == 100
        assert obs.valid.sum() == before, "original observation was mutated"
        assert np.isnan(narrowed.band("red")[20, 20])


class TestObservationStack:
    def test_requires_sorted_order(self, grid: Grid) -> None:
        a = make_obs(grid, "s2", dt.datetime(2025, 1, 10))
        b = make_obs(grid, "s2", dt.datetime(2025, 1, 5))
        with pytest.raises(ValueError, match="acquisition order"):
            ObservationStack(grid, (a, b))
        assert len(ObservationStack.build(grid, [a, b])) == 2

    def test_rejects_mismatched_grid(self, grid: Grid) -> None:
        other = Grid(grid.transform, grid.crs, (10, 10))
        odd = make_obs(other, "s2", dt.datetime(2025, 1, 1))
        with pytest.raises(ValueError, match="different grid"):
            ObservationStack(grid, (odd,))

    def test_clear_count_is_a_count_not_a_boolean(self, stack) -> None:
        """The accumulator bug from the measurement script, caught in a test.

        Summing boolean arrays with ``+`` on a boolean accumulator is OR, so
        the answer comes back as 0 or 1 everywhere and looks like a mask. The
        count has to exceed 1 somewhere or the whole coverage argument is void.
        """
        counts = stack.clear_count()
        assert counts.dtype == np.int16
        assert counts.max() > 1
        assert counts.max() <= len(stack)

    def test_with_bands_filters_rather_than_raising(self, grid: Grid) -> None:
        rich = make_obs(
            grid,
            "s2",
            dt.datetime(2025, 1, 1),
            bands={
                "red": np.zeros(grid.shape, np.float32),
                "nir": np.zeros(grid.shape, np.float32),
                "rededge1": np.zeros(grid.shape, np.float32),
            },
        )
        poor = make_obs(grid, "l8", dt.datetime(2025, 1, 2))
        s = ObservationStack.build(grid, [rich, poor])
        assert len(s.with_bands("rededge1")) == 1
        assert len(s.with_bands("red", "nir")) == 2
        assert len(s.with_bands("swir2")) == 0, "empty is a legitimate answer"

    def test_radar_and_optical_separate(self, grid: Grid) -> None:
        opt = make_obs(grid, "s2", dt.datetime(2025, 1, 1))
        rad = make_obs(
            grid,
            "s1",
            dt.datetime(2025, 1, 2),
            bands={
                "vv": np.full(grid.shape, -8.0, np.float32),
                "vh": np.full(grid.shape, -14.0, np.float32),
            },
        )
        s = ObservationStack.build(grid, [opt, rad])
        assert len(s.optical()) == 1
        assert len(s.radar()) == 1
        assert rad.is_radar and not opt.is_radar

    def test_stack_bands_refuses_a_missing_band_loudly(self, grid: Grid) -> None:
        a = make_obs(grid, "s2", dt.datetime(2025, 1, 1))
        b = make_obs(
            grid,
            "l8",
            dt.datetime(2025, 1, 2),
            bands={"red": np.zeros(grid.shape, np.float32)},
        )
        with pytest.raises(BandUnavailable, match="Filter with"):
            stack_bands([a, b], "nir")

    def test_summary_names_the_sensors(self, stack) -> None:
        s = stack.summary()
        assert "s2=3" in s and "l8=2" in s and "l9=1" in s
