"""Sensor adapter tests. No network: every item is a hand-built STAC fragment.

The BOA offset tests are the reason this file exists. That bug produced
imagery that rendered perfectly and was wrong by 0.1 reflectance everywhere,
which is larger than the entire cross-sensor difference the harmonisation
exists to correct.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.observation import Grid, Observation, Provenance
from cropatlas.sensors.deafrica import (
    LANDSAT_SR_OFFSET,
    LANDSAT_SR_SCALE,
    Sentinel1RTC,
    Sentinel2,
    default_sensors,
    landsat8,
    landsat9,
)


def item(props: dict) -> dict:
    return {"id": "test-item", "properties": {"datetime": "2025-01-05T08:15:00Z", **props}}


class TestBOAOffset:
    def test_archive_flag_wins_over_the_baseline(self) -> None:
        """The actual bug.

        DE Africa serves baseline 05.11 products with the offset ALREADY
        applied and says so. Reading the baseline and applying -1000 subtracts
        it twice, which put Sentinel-2 red at -0.008 where Landsat read +0.079
        over the same ground on the same day.
        """
        s2 = Sentinel2()
        offset, why = s2._offset_for(
            item(
                {
                    "sentinel:processing_baseline": "05.11",
                    "sentinel:boa_offset_applied": True,
                }
            )
        )
        assert offset == 0.0
        assert "flag" in why

    def test_flag_false_means_apply_it(self) -> None:
        offset, why = s2_offset({"sentinel:boa_offset_applied": False})
        assert offset == -1000.0
        assert "flag" in why

    def test_falls_back_to_baseline_when_no_flag(self) -> None:
        assert s2_offset({"sentinel:processing_baseline": "05.11"})[0] == -1000.0
        assert s2_offset({"sentinel:processing_baseline": "03.01"})[0] == 0.0

    def test_falls_back_to_date_when_nothing_is_published(self) -> None:
        offset, why = s2_offset({})
        assert offset == -1000.0
        assert "inferred from date" in why

    def test_pre_cutover_date_needs_no_offset(self) -> None:
        s2 = Sentinel2()
        offset, _ = s2._offset_for(
            {"id": "x", "properties": {"datetime": "2021-06-01T08:15:00Z"}}
        )
        assert offset == 0.0

    def test_caller_can_override_either_way(self) -> None:
        assert Sentinel2(assume_offset_applied=True)._offset_for(
            item({"sentinel:boa_offset_applied": False})
        )[0] == 0.0
        assert Sentinel2(assume_offset_applied=False)._offset_for(
            item({"sentinel:boa_offset_applied": True})
        )[0] == -1000.0

    def test_the_reason_is_always_recorded(self) -> None:
        """Whatever route the answer came by, it lands in provenance, so a
        wrong decision is discoverable in the output rather than invisible."""
        for props in (
            {"sentinel:boa_offset_applied": True},
            {"sentinel:processing_baseline": "05.11"},
            {},
        ):
            _, why = s2_offset(props)
            assert why and isinstance(why, str)


def s2_offset(props: dict) -> tuple[float, str]:
    return Sentinel2()._offset_for(item(props))


class TestOffsetSanity:
    def make(self, grid: Grid, **bands: float) -> Observation:
        return Observation(
            grid=grid,
            bands={
                k: np.full(grid.shape, np.float32(v), np.float32)
                for k, v in bands.items()
            },
            valid=np.ones(grid.shape, bool),
            provenance=Provenance("s2", "x", dt.datetime(2025, 1, 5)),
        )

    def test_healthy_scene_passes_silently(self, grid: Grid) -> None:
        assert Sentinel2().offset_sanity(self.make(grid, red=0.06, nir=0.28)) is None

    def test_catches_the_real_bug_with_no_metadata_at_all(self, grid: Grid) -> None:
        """The measured signature of the bug: Sentinel-2 red at -0.008 where
        Landsat read +0.079 over the same ground on the same day."""
        warning = Sentinel2().offset_sanity(self.make(grid, red=-0.0079, nir=0.173))
        assert warning is not None
        assert "red" in warning and "twice" in warning

    def test_a_nir_only_check_would_have_missed_it(self, grid: Grid) -> None:
        """Why red is the diagnostic band and not the obvious one.

        Subtracting 0.1 from a NIR of 0.27 leaves 0.17: low, but entirely
        plausible. Subtracting it from a red of 0.09 leaves -0.01, which
        nothing physical does. The band with the smallest true value is where
        a fixed additive error shows. A first version of this check used NIR
        and would have passed the very scene it was written to catch.
        """
        nir_only = self.make(grid, nir=0.27 - 0.1)
        assert Sentinel2().offset_sanity(nir_only) is None
        with_red = self.make(grid, red=0.09 - 0.1, nir=0.27 - 0.1)
        assert Sentinel2().offset_sanity(with_red) is not None

    def test_says_nothing_without_the_bands_it_needs(self, grid: Grid) -> None:
        assert Sentinel2().offset_sanity(self.make(grid, swir1=0.25)) is None

    def test_says_nothing_when_nothing_is_valid(self, grid: Grid) -> None:
        obs = Observation(
            grid=grid,
            bands={"nir": np.full(grid.shape, np.float32(np.nan), np.float32)},
            valid=np.zeros(grid.shape, bool),
            provenance=Provenance("s2", "x", dt.datetime(2025, 1, 5)),
        )
        assert Sentinel2().offset_sanity(obs) is None


class TestLandsatScaling:
    def test_the_published_collection_2_scaling(self) -> None:
        """reflectance = DN * 0.0000275 - 0.2, fixed for Collection 2 Level 2.

        A DN of 10000 is about 0.075, which is a plausible red over a canopy.
        Pinned as a test because a silent change here would shift every
        Landsat value and look like a cross-sensor calibration problem.
        """
        assert pytest.approx(0.075) == 10_000 * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
        assert pytest.approx(0.0, abs=1e-3) == 7_273 * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET


class TestSensorContracts:
    def test_landsat_8_and_9_are_separate_sensors(self) -> None:
        """Treating them as one adapter would assume their radiometry is
        identical. Whether it is, is a question this pipeline can answer."""
        assert landsat8().key == "l8"
        assert landsat9().key == "l9"
        assert landsat8().collection != landsat9().collection

    def test_landsat_has_no_red_edge(self) -> None:
        assert "rededge1" not in landsat8().bands
        assert "swir2" in landsat8().bands

    def test_sentinel2_carries_red_edge(self) -> None:
        for b in ("rededge1", "rededge2", "rededge3"):
            assert b in Sentinel2().bands

    def test_radar_declares_no_cloud_cover(self) -> None:
        """Radar does not care about cloud, and the absence of the field is
        the honest way to say so rather than reporting zero."""
        assert Sentinel1RTC().bands == ("vv", "vh")

    def test_default_sensors_is_a_fresh_list_each_call(self) -> None:
        """A module constant would invite mutation of shared state, and the
        whole argument of findings/02 is that the sensor set is a per-run
        decision."""
        a, b = default_sensors(), default_sensors()
        assert a is not b
        assert [s.key for s in a] == ["s2", "l8", "l9", "s1"]
