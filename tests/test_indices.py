from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas import indices
from cropatlas.indices import CIRE, EVI, NDMI, NDRE, NDVI, SAVI, available_indices
from cropatlas.observation import Grid
from tests.conftest import make_obs


def arr(v: float, shape=(2, 2)) -> np.ndarray:
    return np.full(shape, v, dtype=np.float32)


class TestNDVI:
    def test_healthy_canopy(self) -> None:
        out = NDVI({"nir": arr(0.40), "red": arr(0.05)})
        assert out[0, 0] == pytest.approx((0.40 - 0.05) / 0.45, abs=1e-5)

    def test_bare_soil_is_low_not_negative(self) -> None:
        out = NDVI({"nir": arr(0.25), "red": arr(0.20)})
        assert 0.0 < out[0, 0] < 0.2

    def test_water_is_negative(self) -> None:
        out = NDVI({"nir": arr(0.02), "red": arr(0.06)})
        assert out[0, 0] < 0

    def test_negative_red_does_not_produce_a_supergreen_pixel(self) -> None:
        """The regression test for the second-order bug.

        Allowing slightly negative reflectance is correct for compositing and
        wrong for a ratio: red = -0.05 with nir = 0.30 gives a raw NDVI of
        1.40, which the range check then turns into NaN, and the pixel vanishes
        from the map. Fixing the reflectance floor collapsed a whole February
        of zone statistics this way. Two individually correct decisions
        produced a wrong answer where they met.
        """
        out = NDVI({"nir": arr(0.30), "red": arr(-0.05)})
        assert np.isfinite(out).all(), "the pixel must survive"
        assert out[0, 0] == pytest.approx(1.0), "clamped at the physical bound"

    def test_clamping_can_be_turned_off_deliberately(self) -> None:
        raw = indices.Index(
            name="ndvi_raw",
            bands=("nir", "red"),
            compute=indices._normalised,
            description="unclamped, for demonstrating the artefact",
            clamp_negative=False,
        )
        assert np.isnan(raw({"nir": arr(0.30), "red": arr(-0.05)})).all()

    def test_zero_denominator_is_nan_not_an_exception(self) -> None:
        out = NDVI({"nir": arr(0.0), "red": arr(0.0)})
        assert np.isnan(out).all()

    def test_nan_input_propagates(self) -> None:
        out = NDVI({"nir": arr(np.nan), "red": arr(0.05)})
        assert np.isnan(out).all()

    def test_missing_band_names_what_is_needed(self) -> None:
        with pytest.raises(KeyError, match="ndvi needs"):
            NDVI({"nir": arr(0.4)})


class TestOtherIndices:
    def test_savi_is_lower_than_ndvi_over_partial_canopy(self) -> None:
        """SAVI's whole purpose: NDVI over sparse canopy is inflated by the
        soil line, which is what a smallholder plot looks like for six weeks
        after planting."""
        bands = {"nir": arr(0.25), "red": arr(0.12)}
        assert SAVI(bands)[0, 0] < NDVI(bands)[0, 0]

    def test_ndmi_falls_with_canopy_water(self) -> None:
        wet = NDMI({"nir": arr(0.35), "swir1": arr(0.15)})[0, 0]
        dry = NDMI({"nir": arr(0.35), "swir1": arr(0.30)})[0, 0]
        assert wet > dry

    def test_evi_stays_finite_over_normal_reflectance(self) -> None:
        out = EVI({"nir": arr(0.40), "red": arr(0.05), "blue": arr(0.03)})
        assert np.isfinite(out).all()
        assert 0.0 < out[0, 0] < 2.5

    def test_ndre_and_cire_need_red_edge(self) -> None:
        with pytest.raises(KeyError):
            NDRE({"nir": arr(0.4), "red": arr(0.05)})
        out = CIRE({"nir": arr(0.40), "rededge1": arr(0.20)})
        assert out[0, 0] == pytest.approx(1.0)


class TestAvailability:
    def test_landsat_offers_no_red_edge(self, grid: Grid) -> None:
        """The honest way to drive a UI: offer what the season supports, rather
        than offering everything and rendering an empty map."""
        l8 = make_obs(grid, "l8", dt.datetime(2025, 1, 1))
        got = available_indices(l8)
        assert "ndvi" in got and "savi" in got
        assert "ndre" not in got and "cire" not in got

    def test_sentinel2_offers_red_edge(self, grid: Grid) -> None:
        s2 = make_obs(
            grid, "s2", dt.datetime(2025, 1, 1),
            bands={
                "red": arr(0.05, grid.shape),
                "nir": arr(0.4, grid.shape),
                "rededge1": arr(0.2, grid.shape),
                "swir1": arr(0.2, grid.shape),
            },
        )
        got = available_indices(s2)
        assert "ndre" in got and "cire" in got and "ndmi" in got

    def test_compute_all_skips_what_it_cannot_make(self, grid: Grid) -> None:
        l8 = make_obs(grid, "l8", dt.datetime(2025, 1, 1))
        out = indices.compute_all(l8, ["ndvi", "ndre"])
        assert "ndvi" in out and "ndre" not in out
