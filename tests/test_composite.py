from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.composite import composite, coverage_table, monthly_composites
from cropatlas.observation import Grid, ObservationStack
from tests.conftest import make_obs


class TestComposite:
    def test_blind_fraction_is_zero_when_every_pixel_was_seen(self, stack) -> None:
        c = composite(stack, ["red", "nir"])
        assert c.blind_fraction == 0.0
        assert c.mean_looks > 1.0

    def test_a_hole_common_to_every_scene_stays_blind(self, grid: Grid) -> None:
        """The February case: some ground is never seen by anybody.

        The compositor must report it rather than filling it, because a filled
        pixel is indistinguishable from an observed one on a map and that is
        exactly how these products mislead.
        """
        obs = []
        for i, day in enumerate([2, 8, 15]):
            valid = np.ones(grid.shape, bool)
            valid[:5, :5] = False  # the same corner, every pass
            valid[10 + i, :] = False
            obs.append(make_obs(grid, "s2", dt.datetime(2025, 2, day), valid=valid))
        c = composite(ObservationStack.build(grid, obs), ["red", "nir"])
        assert c.blind_fraction == pytest.approx(25 / (grid.shape[0] * grid.shape[1]))
        assert np.isnan(c.band("red")[2, 2]), "unseen pixels must be NaN, not filled"
        assert c.clear_count[2, 2] == 0
        assert np.isfinite(c.band("red")[30, 30])

    def test_empty_stack_returns_an_empty_composite_not_an_exception(
        self, grid: Grid
    ) -> None:
        """A month with no usable scenes is a real month, and the pipeline has
        to render it as blank rather than crash the season."""
        c = composite(ObservationStack.build(grid, []), ["red"])
        assert c.blind_fraction == 1.0
        assert c.provenance.sensors == ()
        assert np.isnan(c.band("red")).all()
        assert "no observations" in " ".join(c.provenance.notes)

    def test_radar_is_ignored_by_an_optical_composite(self, grid: Grid) -> None:
        """Decibels must never be medianed with reflectance."""
        opt = make_obs(grid, "s2", dt.datetime(2025, 1, 1))
        rad = make_obs(
            grid,
            "s1",
            dt.datetime(2025, 1, 2),
            bands={"vv": np.full(grid.shape, -8.0, np.float32)},
        )
        c = composite(ObservationStack.build(grid, [opt, rad]), ["red", "nir"])
        assert c.provenance.sensors == ("s2",)
        assert "s1" not in c.provenance.scenes_by_sensor

    def test_median_rejects_a_single_bright_contaminated_look(
        self, grid: Grid
    ) -> None:
        """Residual haze the classifier missed pushes reflectance up.

        With three looks the median rejects it and the mean does not. This is
        the entire argument for the default reducer, so it is a test rather
        than a comment.
        """
        obs = []
        for i, day in enumerate([1, 5, 9]):
            red = np.full(grid.shape, 0.05, np.float32)
            if i == 1:
                red[:] = 0.35  # haze
            obs.append(
                make_obs(
                    grid,
                    "s2",
                    dt.datetime(2025, 1, day),
                    bands={"red": red, "nir": np.full(grid.shape, 0.4, np.float32)},
                )
            )
        s = ObservationStack.build(grid, obs)
        med = composite(s, ["red"], reducer="median").band("red")
        mean = composite(s, ["red"], reducer="mean").band("red")
        assert med[0, 0] == pytest.approx(0.05, abs=1e-4)
        assert mean[0, 0] == pytest.approx(0.15, abs=1e-3)

    def test_provenance_records_which_sensors_contributed(self, stack) -> None:
        c = composite(stack, ["red", "nir"])
        assert set(c.provenance.sensors) == {"s2", "l8", "l9"}
        assert c.provenance.scenes_by_sensor["s2"] == 3
        assert c.provenance.acquisition_days == 6

    def test_uncalibrated_sensors_surface_as_a_caveat(self, stack) -> None:
        """Findings/02 in code: a number from an uncalibrated mix has to say so."""
        c = composite(stack, ["red", "nir"], uncalibrated=["l8", "l9"])
        caveats = " ".join(c.caveats() if hasattr(c, "caveats") else c.provenance.caveats())
        assert "l8" in caveats and "calibration" in caveats

    def test_single_sensor_is_flagged(self, grid: Grid) -> None:
        obs = [make_obs(grid, "s2", dt.datetime(2025, 1, d)) for d in (1, 5)]
        c = composite(ObservationStack.build(grid, obs), ["red"])
        assert any("single sensor" in x for x in c.provenance.caveats())

    def test_confidence_saturates_at_three_looks(self, grid: Grid) -> None:
        obs = [make_obs(grid, "s2", dt.datetime(2025, 1, d)) for d in (1, 5, 9, 13)]
        c = composite(ObservationStack.build(grid, obs), ["red"])
        assert c.confidence().max() == pytest.approx(1.0)
        assert c.coverage(3) == 1.0

    def test_first_reducer_takes_the_first_finite_not_the_first_slot(
        self, grid: Grid
    ) -> None:
        obs = []
        for i, day in enumerate([1, 5, 9]):
            valid = np.ones(grid.shape, bool)
            if i == 0:
                valid[0, 0] = False  # first scene is cloudy on that pixel
            red = np.full(grid.shape, np.float32(0.1 * (i + 1)), np.float32)
            obs.append(
                make_obs(
                    grid, "s2", dt.datetime(2025, 1, day),
                    bands={"red": red}, valid=valid,
                )
            )
        c = composite(ObservationStack.build(grid, obs), ["red"], reducer="first")
        assert c.band("red")[0, 0] == pytest.approx(0.2), "skipped the cloudy first look"
        assert c.band("red")[5, 5] == pytest.approx(0.1)


class TestMonthly:
    def test_splits_a_season_into_months_including_empty_ones(
        self, grid: Grid
    ) -> None:
        obs = [
            make_obs(grid, "s2", dt.datetime(2024, 11, 10)),
            make_obs(grid, "s2", dt.datetime(2025, 1, 12)),
        ]
        s = ObservationStack.build(grid, obs)
        out = monthly_composites(
            s, ["red"], dt.datetime(2024, 11, 1), dt.datetime(2025, 3, 1)
        )
        assert sorted(out) == ["2024-11", "2024-12", "2025-01", "2025-02"]
        assert out["2024-12"].blind_fraction == 1.0, "December saw nothing"
        assert out["2025-01"].blind_fraction == 0.0

    def test_year_boundary_does_not_lose_december(self, grid: Grid) -> None:
        obs = [make_obs(grid, "s2", dt.datetime(2024, 12, 20))]
        out = monthly_composites(
            ObservationStack.build(grid, obs),
            ["red"],
            dt.datetime(2024, 12, 1),
            dt.datetime(2025, 2, 1),
        )
        assert out["2024-12"].blind_fraction == 0.0

    def test_coverage_table_renders(self, stack) -> None:
        out = monthly_composites(
            stack, ["red"], dt.datetime(2025, 1, 1), dt.datetime(2025, 3, 1)
        )
        table = coverage_table(out)
        assert "2025-01" in table and "never seen" in table
