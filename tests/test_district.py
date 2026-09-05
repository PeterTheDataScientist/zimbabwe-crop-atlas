from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.composite import composite
from cropatlas.district import (
    Zone,
    coverage_bias_check,
    grid_zones,
    roll_up,
    statistics_table,
    summarise,
    zone_statistic,
)
from cropatlas.observation import Grid, ObservationStack
from tests.conftest import make_obs


def full_zone(grid: Grid, name: str = "Goromonzi") -> Zone:
    return Zone(name=name, mask=np.ones(grid.shape, bool))


class TestZone:
    def test_area_from_pixel_count(self, grid: Grid) -> None:
        mask = np.zeros(grid.shape, bool)
        mask[:10, :10] = True  # 100 pixels of 400 m2
        z = Zone("block", mask)
        assert z.pixel_count == 100
        assert z.area_ha(grid) == pytest.approx(4.0)

    def test_rejects_a_non_boolean_mask(self, grid: Grid) -> None:
        with pytest.raises(TypeError, match="boolean"):
            Zone("bad", np.ones(grid.shape, np.uint8))


class TestWithholding:
    def test_reports_when_coverage_is_good(self, grid: Grid) -> None:
        vals = np.full(grid.shape, 0.55, np.float32)
        looks = np.full(grid.shape, 4, np.int16)
        s = zone_statistic(vals, looks, full_zone(grid))
        assert s.reported
        assert s.value == pytest.approx(0.55)
        assert s.coverage == 1.0

    def test_withholds_rather_than_reporting_a_biased_mean(self, grid: Grid) -> None:
        """The central claim of this module. February cloud sits on the high,
        wet ground, so a mean over the visible half of a district is biased
        with a known sign, not merely noisy. A blank cell that says why beats a
        plausible number nobody can audit."""
        vals = np.full(grid.shape, 0.55, np.float32)
        vals[: grid.shape[0] // 2, :] = np.nan  # half the district unseen
        looks = np.full(grid.shape, 2, np.int16)
        s = zone_statistic(vals, looks, full_zone(grid))
        assert not s.reported
        assert s.value is None
        assert "not random" in s.withheld_reason
        assert s.coverage == pytest.approx(0.5)

    def test_the_floor_is_a_parameter_not_a_secret(self, grid: Grid) -> None:
        vals = np.full(grid.shape, 0.55, np.float32)
        vals[: grid.shape[0] // 2, :] = np.nan
        s = zone_statistic(vals, np.full(grid.shape, 2, np.int16),
                           full_zone(grid), min_coverage=0.4)
        assert s.reported, "a caller can lower the floor, but must do it in the open"

    def test_withholds_on_thin_observation_density(self, grid: Grid) -> None:
        vals = np.full(grid.shape, 0.55, np.float32)
        looks = np.zeros(grid.shape, np.int16)
        s = zone_statistic(vals, looks, full_zone(grid), min_mean_looks=1.0)
        assert not s.reported
        assert "clear looks" in s.withheld_reason

    def test_empty_zone_says_so(self, grid: Grid) -> None:
        s = zone_statistic(
            np.full(grid.shape, 0.5, np.float32),
            np.full(grid.shape, 3, np.int16),
            Zone("nowhere", np.zeros(grid.shape, bool)),
        )
        assert not s.reported
        assert "no pixels" in s.withheld_reason

    def test_spread_travels_with_the_value(self, grid: Grid) -> None:
        rng = np.random.default_rng(4)
        vals = rng.normal(0.5, 0.1, grid.shape).astype(np.float32)
        s = zone_statistic(vals, np.full(grid.shape, 3, np.int16), full_zone(grid))
        assert s.p10 is not None and s.p90 is not None
        assert s.p10 < s.value < s.p90

    def test_median_statistic(self, grid: Grid) -> None:
        vals = np.full(grid.shape, 0.4, np.float32)
        vals[0, 0] = 10.0  # an outlier the mean would feel
        s = zone_statistic(
            vals, np.full(grid.shape, 3, np.int16), full_zone(grid),
            statistic="median",
        )
        assert s.value == pytest.approx(0.4)


class TestRollUp:
    def test_over_a_lattice_traces_the_shape_of_the_cloud(self, grid: Grid) -> None:
        """A regular lattice makes structured missingness visible: the withheld
        cells sit where the cloud was, which is a far stronger demonstration
        that gaps are not random than any amount of prose."""
        obs = []
        for day in (2, 8, 15):
            valid = np.ones(grid.shape, bool)
            valid[:20, :25] = False  # one quadrant, every pass
            obs.append(make_obs(grid, "s2", dt.datetime(2025, 2, day), valid=valid))
        c = composite(ObservationStack.build(grid, obs), ["red", "nir"])

        zones = grid_zones(grid, 2, 2)
        stats = roll_up(c.band("red"), c, zones)
        by_name = {s.zone: s for s in stats}
        assert not by_name["r0c0"].reported, "the permanently clouded quadrant"
        assert by_name["r1c1"].reported
        assert summarise(stats)["withheld"] == 1

    def test_table_renders_both_states(self, grid: Grid) -> None:
        vals = np.full(grid.shape, 0.5, np.float32)
        vals[:20, :] = np.nan
        c = composite(
            ObservationStack.build(
                grid, [make_obs(grid, "s2", dt.datetime(2025, 1, 5))]
            ),
            ["red"],
        )
        table = statistics_table(roll_up(vals, c, grid_zones(grid, 2, 1)))
        assert "withheld" in table and "reported" in table


class TestCoverageBias:
    def test_detects_that_unseen_ground_is_systematically_higher(
        self, grid: Grid
    ) -> None:
        """The check that turns 'coverage was 83%' into a statement about
        whether the 83% is representative."""
        elevation = np.zeros(grid.shape, np.float32)
        elevation[:20, :] = 1600.0  # the highveld
        elevation[20:, :] = 1200.0

        vals = np.full(grid.shape, 0.5, np.float32)
        looks = np.full(grid.shape, 3, np.int16)
        vals[:20, :] = np.nan  # cloud sat on the high ground
        looks[:20, :] = 0

        rep = coverage_bias_check(vals, looks, elevation, "elevation")
        assert rep["difference"] == pytest.approx(400.0)
        assert rep["standardised_difference"] > 1.0
        assert rep["perfectly_separated"] is True, (
            "two uniform groups 400 m apart are the most separated case there "
            "is; reporting that as no bias is the failure this test exists for"
        )

    def test_partially_overlapping_groups_give_a_finite_effect_size(
        self, grid: Grid
    ) -> None:
        rng = np.random.default_rng(21)
        elevation = rng.normal(1400, 100, grid.shape).astype(np.float32)
        elevation[:20, :] += 150.0
        vals = np.full(grid.shape, 0.5, np.float32)
        looks = np.full(grid.shape, 3, np.int16)
        vals[:20, :] = np.nan
        looks[:20, :] = 0
        rep = coverage_bias_check(vals, looks, elevation)
        assert rep["perfectly_separated"] is False
        assert 0.8 < rep["standardised_difference"] < 2.5

    def test_reports_no_difference_when_gaps_are_random(self, grid: Grid) -> None:
        rng = np.random.default_rng(6)
        elevation = rng.normal(1400, 100, grid.shape).astype(np.float32)
        vals = np.full(grid.shape, 0.5, np.float32)
        looks = np.full(grid.shape, 3, np.int16)
        drop = rng.random(grid.shape) < 0.3
        vals[drop] = np.nan
        looks[drop] = 0
        rep = coverage_bias_check(vals, looks, elevation)
        assert abs(rep["standardised_difference"]) < 0.2

    def test_handles_a_fully_observed_grid(self, grid: Grid) -> None:
        rep = coverage_bias_check(
            np.full(grid.shape, 0.5, np.float32),
            np.full(grid.shape, 3, np.int16),
            np.full(grid.shape, 1400.0, np.float32),
        )
        assert rep["unseen_n"] == 0
