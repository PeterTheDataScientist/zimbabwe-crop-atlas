from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.harmonise import (
    BandAdjustment,
    Harmoniser,
    find_coincident,
    fit_band,
    fit_harmoniser,
    harmonise_stack,
    merge_calibration,
    sensor_offset_report,
)
from cropatlas.observation import Grid, ObservationStack
from tests.conftest import make_obs

# The truth the fitter has to recover. Chosen to look like a real cross-sensor
# offset: a few percent of gain and a small additive term, which is the size of
# thing that is invisible by eye and produces a sawtooth in a time series.
TRUE_SLOPE = 0.964
TRUE_INTERCEPT = 0.0083


def paired_stack(
    grid: Grid,
    *,
    hours_apart: float = 6.0,
    n_pairs: int = 4,
    noise: float = 0.0,
    seed: int = 11,
) -> ObservationStack:
    """Coincident s2/l8 acquisitions where s2 = a * l8 + b, exactly.

    Building the relationship in this direction means the fitter, which
    regresses reference on source, must return TRUE_SLOPE and TRUE_INTERCEPT
    themselves rather than their inverse. If the code ever flips the direction
    of the regression this test fails loudly instead of applying a correction
    that doubles the error it was meant to remove.
    """
    rng = np.random.default_rng(seed)
    obs = []
    for i in range(n_pairs):
        base = dt.datetime(2025, 1, 4, 8, 30) + dt.timedelta(days=7 * i)
        l8_red = rng.uniform(0.02, 0.30, grid.shape).astype(np.float32)
        l8_nir = rng.uniform(0.15, 0.55, grid.shape).astype(np.float32)
        s2_red = (TRUE_SLOPE * l8_red + TRUE_INTERCEPT).astype(np.float32)
        s2_nir = (TRUE_SLOPE * l8_nir + TRUE_INTERCEPT).astype(np.float32)
        if noise:
            s2_red = (s2_red + rng.normal(0, noise, grid.shape)).astype(np.float32)
            s2_nir = (s2_nir + rng.normal(0, noise, grid.shape)).astype(np.float32)
        obs.append(
            make_obs(grid, "l8", base, bands={"red": l8_red, "nir": l8_nir})
        )
        obs.append(
            make_obs(
                grid,
                "s2",
                base + dt.timedelta(hours=hours_apart),
                bands={"red": s2_red, "nir": s2_nir},
            )
        )
    return ObservationStack.build(grid, obs)


class TestCoincidence:
    def test_pairs_within_the_window(self, grid: Grid) -> None:
        s = paired_stack(grid, hours_apart=6.0, n_pairs=3)
        assert len(find_coincident(s, "s2", max_hours=30.0)) == 3

    def test_window_excludes_scenes_a_week_apart(self, grid: Grid) -> None:
        s = paired_stack(grid, hours_apart=6.0, n_pairs=3)
        assert find_coincident(s, "s2", max_hours=1.0) == []

    def test_a_pair_straddling_midnight_utc_is_kept(self, grid: Grid) -> None:
        """Sentinel-2 crosses near 10:30 local and Landsat near 10:00, so a
        genuine same-cycle pair can land on different UTC dates. A calendar day
        test would silently discard the best calibration evidence there is."""
        obs = [
            make_obs(grid, "l8", dt.datetime(2025, 1, 8, 22, 0)),
            make_obs(grid, "s2", dt.datetime(2025, 1, 9, 8, 30)),
        ]
        s = ObservationStack.build(grid, obs)
        pairs = find_coincident(s, "s2", max_hours=30.0)
        assert len(pairs) == 1
        assert pairs[0].hours_apart == pytest.approx(10.5)

    def test_radar_never_pairs_with_optical(self, grid: Grid) -> None:
        obs = [
            make_obs(
                grid, "s1", dt.datetime(2025, 1, 8),
                bands={"vv": np.full(grid.shape, -8.0, np.float32)},
            ),
            make_obs(grid, "s2", dt.datetime(2025, 1, 8, 8, 30)),
        ]
        s = ObservationStack.build(grid, obs)
        assert find_coincident(s, "s2", max_hours=30.0) == []


class TestFitting:
    def test_recovers_a_known_linear_offset(self, grid: Grid) -> None:
        s = paired_stack(grid, n_pairs=4)
        adj = fit_band(find_coincident(s, "s2"), "red")
        assert adj is not None
        assert adj.slope == pytest.approx(TRUE_SLOPE, abs=1e-4)
        assert adj.intercept == pytest.approx(TRUE_INTERCEPT, abs=1e-4)
        assert adj.r2 == pytest.approx(1.0, abs=1e-6)
        assert adj.n_pairs > 1000

    def test_recovers_the_offset_through_noise(self, grid: Grid) -> None:
        s = paired_stack(grid, n_pairs=6, noise=0.01)
        adj = fit_band(find_coincident(s, "s2"), "red")
        assert adj is not None
        assert adj.slope == pytest.approx(TRUE_SLOPE, abs=0.02)
        assert adj.rmse is not None and adj.rmse < 0.02

    def test_returns_none_rather_than_a_fit_from_nothing(self, grid: Grid) -> None:
        """A fit from nine pixels applied to a whole district is worse than no
        fit at all, because it looks like calibration."""
        s = paired_stack(grid, n_pairs=1)
        assert fit_band(find_coincident(s, "s2"), "red", min_pairs=10_000) is None

    def test_returns_none_for_a_band_neither_sensor_has(self, grid: Grid) -> None:
        s = paired_stack(grid, n_pairs=3)
        assert fit_band(find_coincident(s, "s2"), "swir2") is None

    def test_only_pixels_both_sensors_called_clear_contribute(
        self, grid: Grid
    ) -> None:
        """A cloudy pixel in either scene would drag the fit. The masks already
        encode which pixels each instrument stands behind, so the intersection
        is the calibration set."""
        rng = np.random.default_rng(3)
        l8_red = rng.uniform(0.02, 0.3, grid.shape).astype(np.float32)
        s2_red = (TRUE_SLOPE * l8_red + TRUE_INTERCEPT).astype(np.float32)
        # Corrupt a corner of the s2 scene and mark it invalid there.
        s2_valid = np.ones(grid.shape, bool)
        s2_valid[:15, :15] = False
        s2_red = s2_red.copy()
        s2_red[:15, :15] = 0.9  # nonsense that would wreck an unmasked fit

        obs = [
            make_obs(grid, "l8", dt.datetime(2025, 1, 8), bands={"red": l8_red}),
            make_obs(
                grid, "s2", dt.datetime(2025, 1, 8, 8),
                bands={"red": s2_red}, valid=s2_valid,
            ),
        ]
        s = ObservationStack.build(grid, obs)
        adj = fit_band(find_coincident(s, "s2"), "red", trim=0.0)
        assert adj is not None
        assert adj.slope == pytest.approx(TRUE_SLOPE, abs=1e-3)

    def test_degenerate_input_does_not_produce_a_garbage_slope(
        self, grid: Grid
    ) -> None:
        """Every source pixel identical means there is no slope to fit.
        numpy.polyfit warns and returns a number; this must not."""
        flat = np.full(grid.shape, 0.2, np.float32)
        obs = [
            make_obs(grid, "l8", dt.datetime(2025, 1, 8), bands={"red": flat}),
            make_obs(
                grid, "s2", dt.datetime(2025, 1, 8, 8),
                bands={"red": np.full(grid.shape, 0.25, np.float32)},
            ),
        ]
        s = ObservationStack.build(grid, obs)
        adj = fit_band(find_coincident(s, "s2"), "red")
        assert adj is not None
        assert adj.slope == pytest.approx(1.0)


class TestErrorsInVariables:
    """Why the estimator is Deming rather than ordinary least squares.

    OLS assumes the predictor is known exactly. Both sides of a cross-sensor
    pair are measurements: resampled from different native resolutions,
    geolocated to within a fraction of a pixel, acquired up to thirty hours
    apart, each with its own atmospheric correction error. Noise in the
    predictor biases the OLS slope toward zero, and the bias grows with the
    scatter.

    That is not a theoretical worry. The first live run of this pipeline fitted
    Landsat 9 NIR at slope 0.64 with r-squared 0.62, while Landsat 8 NIR on the
    same season fitted at 0.86 with r-squared 0.79. Two nominally identical
    instruments cannot differ from Sentinel-2 by that much. The slopes were
    tracking the noise.
    """

    @staticmethod
    def noisy_pair(
        n: int = 60_000, slope: float = 1.0, noise_x: float = 0.03, seed: int = 17
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        truth = rng.uniform(0.02, 0.45, n)
        x = truth + rng.normal(0, noise_x, n)
        y = slope * truth + rng.normal(0, noise_x, n)
        return x, y

    def test_ols_is_attenuated_by_exactly_the_predicted_factor(self) -> None:
        """Not merely 'OLS comes out lower', but lower by the amount theory says.

        The attenuation factor is the reliability of the predictor,
        var(truth) / (var(truth) + var(noise)). Here truth is uniform on
        [0.02, 0.45], so var(truth) = 0.43^2 / 12 = 0.01541, and the added
        noise has var 0.05^2 = 0.0025. The factor is therefore 0.861, and a
        true slope of 1.0 should come back at about 0.861.

        Asserting the predicted value rather than an arbitrary bound is the
        difference between observing that something is off and knowing why.
        """
        from cropatlas.harmonise import _ols

        noise = 0.05
        var_truth = (0.45 - 0.02) ** 2 / 12.0
        predicted = var_truth / (var_truth + noise**2)

        x, y = self.noisy_pair(slope=1.0, noise_x=noise)
        slope, _ = _ols(x, y)
        assert slope == pytest.approx(predicted, abs=0.01)
        assert predicted < 0.87, "sanity: the bias should be visible at this noise"

    def test_deming_recovers_the_true_slope_through_that_same_noise(self) -> None:
        from cropatlas.harmonise import _deming

        x, y = self.noisy_pair(slope=1.0, noise_x=0.05)
        slope, intercept = _deming(x, y)
        assert slope == pytest.approx(1.0, abs=0.03)
        assert intercept == pytest.approx(0.0, abs=0.02)

    def test_both_agree_when_the_predictor_is_clean(self) -> None:
        """Deming is not a different answer, it is the same answer without the
        bias. With no noise in x the two estimators must coincide, or one of
        them is simply wrong."""
        from cropatlas.harmonise import _deming, _ols

        rng = np.random.default_rng(4)
        x = rng.uniform(0.02, 0.45, 20_000)
        y = 0.96 * x + 0.008 + rng.normal(0, 0.004, x.size)
        a, _ = _ols(x, y)
        b, _ = _deming(x, y)
        assert a == pytest.approx(b, abs=0.02)

    def test_deming_recovers_a_non_unit_slope(self) -> None:
        from cropatlas.harmonise import _deming

        x, y = self.noisy_pair(slope=0.90, noise_x=0.04)
        slope, _ = _deming(x, y)
        assert slope == pytest.approx(0.90, abs=0.04)

    def test_degenerate_input_is_refused_by_both_estimators(self) -> None:
        from cropatlas.harmonise import _deming, _ols

        flat = np.full(5000, 0.2)
        other = np.full(5000, 0.25)
        assert _ols(flat, other)[0] == pytest.approx(1.0)
        assert _deming(flat, other)[0] == pytest.approx(1.0)

    def test_the_fit_reports_both_slopes_so_the_bias_is_visible(
        self, grid: Grid
    ) -> None:
        """The gap between the two is the measured size of the attenuation, and
        a large gap means the calibration is less certain whatever the
        estimator says. Reporting only the chosen one hides that."""
        s = paired_stack(grid, n_pairs=5, noise=0.03)
        adj = fit_band(find_coincident(s, "s2"), "red")
        assert adj is not None
        assert adj.estimator == "deming"
        assert adj.ols_slope is not None
        assert adj.ols_slope != adj.slope

    def test_deming_runs_away_as_the_pairs_get_noisier(self, grid: Grid) -> None:
        """The failure mode the r-squared guard exists to catch.

        Deming assumes noise in BOTH variables in a known ratio, taken as equal
        here. That assumption does real work, and when it is violated or the
        scatter is simply large, the correction stops removing bias and starts
        amplifying disagreement. Measured on this synthetic ladder, true
        slope 0.964:

            noise   r2      deming   ols
            0.004   0.998   0.965    0.964
            0.020   0.950   0.996    0.971
            0.050   0.739   1.177    1.006
            0.080   0.444   1.555    1.058
            0.250  -3.854   7.953    1.289

        This is the same shape as the live run over Harare, where blue fitted
        at r-squared 0.60 and Deming returned 1.51 for Landsat 8 and 1.82 for
        Landsat 9. Two nominally identical instruments cannot differ from
        Sentinel-2 by that much: the estimator was tracking the scatter.
        """
        slopes = {}
        for noise in (0.004, 0.05, 0.08):
            s = paired_stack(grid, n_pairs=4, noise=noise)
            adj = fit_band(find_coincident(s, "s2"), "red", min_r2=-1e9)
            assert adj is not None
            slopes[noise] = (adj.r2, adj.slope)

        assert slopes[0.004][0] > slopes[0.05][0] > slopes[0.08][0]
        assert slopes[0.004][1] < slopes[0.05][1] < slopes[0.08][1]
        assert slopes[0.08][1] > 1.4, "the runaway, reproduced"

    def test_a_band_too_noisy_to_calibrate_is_refused(self, grid: Grid) -> None:
        """The honest answer for such a band is that it cannot be calibrated
        from this evidence. An adjustment that injects more error than it
        removes is worse than none, and worse precisely because it looks like
        diligence."""
        pairs = find_coincident(paired_stack(grid, n_pairs=4, noise=0.08), "s2")
        assert fit_band(pairs, "red", min_r2=0.70) is None
        loose = fit_band(pairs, "red", min_r2=-1e9)
        assert loose is not None and loose.r2 < 0.70

    def test_a_clean_band_still_passes_the_guard(self, grid: Grid) -> None:
        s = paired_stack(grid, n_pairs=4, noise=0.004)
        adj = fit_band(find_coincident(s, "s2"), "red", min_r2=0.70)
        assert adj is not None and adj.r2 > 0.70
        assert adj.slope == pytest.approx(TRUE_SLOPE, abs=0.01), (
            "and where the guard passes, the answer is right"
        )

    def test_a_refused_band_leaves_the_sensor_uncalibrated_in_the_report(
        self, grid: Grid
    ) -> None:
        s = paired_stack(grid, n_pairs=4, noise=0.08)
        h, diag = fit_harmoniser(s, "s2", bands=("red", "nir"), min_r2=0.70)
        assert not h.is_calibrated("l8")
        assert set(diag["skipped"]["l8"]) == {"red", "nir"}

    def test_estimator_is_selectable(self, grid: Grid) -> None:
        s = paired_stack(grid, n_pairs=4, noise=0.02)
        pairs = find_coincident(s, "s2")
        d = fit_band(pairs, "red", estimator="deming")
        o = fit_band(pairs, "red", estimator="ols")
        assert d is not None and o is not None
        assert d.estimator == "deming" and o.estimator == "ols"


class TestHarmoniser:
    def test_identity_changes_nothing_and_says_so(self, grid: Grid) -> None:
        h = Harmoniser.identity()
        obs = make_obs(grid, "l8", dt.datetime(2025, 1, 1))
        assert h.apply(obs) is obs
        assert not h.is_calibrated("l8")
        assert "nothing fitted" in h.report()

    def test_reference_sensor_is_never_adjusted(self, grid: Grid) -> None:
        h = Harmoniser(
            reference="s2",
            adjustments={"s2": {"red": BandAdjustment("red", 2.0, 0.0)}},
        )
        obs = make_obs(
            grid, "s2", dt.datetime(2025, 1, 1),
            bands={"red": np.full(grid.shape, 0.1, np.float32)},
        )
        assert h.apply(obs).band("red")[0, 0] == pytest.approx(0.1)

    def test_applying_the_fit_makes_the_two_sensors_agree(self, grid: Grid) -> None:
        """The end to end claim: after harmonisation, coincident observations
        of the same ground from different instruments return the same number.
        Without that the phenology curve has a sawtooth in it."""
        s = paired_stack(grid, n_pairs=4)
        h, diag = fit_harmoniser(s, "s2", bands=("red", "nir"))
        assert h.is_calibrated("l8")

        pairs = find_coincident(s, "s2")
        before = sensor_offset_report(pairs, "red")
        assert before is not None
        assert abs(before["mean_difference"]) > 0.001, "there was an offset to remove"

        adjusted = harmonise_stack(s, h)
        after = sensor_offset_report(find_coincident(adjusted, "s2"), "red")
        assert after is not None
        assert abs(after["mean_difference"]) < 1e-4, "offset survived harmonisation"
        assert diag["pair_count"]["l8"] == 4

    def test_uncalibrated_sensor_passes_through_rather_than_raising(
        self, grid: Grid
    ) -> None:
        """Refusing to composite because calibration is missing would make the
        package useless on day one. The absence is recorded instead."""
        h = Harmoniser(
            reference="s2",
            adjustments={"l8": {"red": BandAdjustment("red", 0.9, 0.01)}},
        )
        l9 = make_obs(grid, "l9", dt.datetime(2025, 1, 1))
        assert h.apply(l9) is l9

    def test_adjustment_is_recorded_in_provenance(self, grid: Grid) -> None:
        h = Harmoniser(
            reference="s2",
            adjustments={"l8": {"red": BandAdjustment("red", 0.9, 0.01)}},
        )
        out = h.apply(make_obs(grid, "l8", dt.datetime(2025, 1, 1)))
        assert out.provenance.detail["harmonised_to"] == "s2"
        assert out.provenance.detail["harmonised_bands"] == "red"

    def test_round_trips_through_json(self, grid: Grid, tmp_path) -> None:
        s = paired_stack(grid, n_pairs=4)
        h, _ = fit_harmoniser(s, "s2", bands=("red", "nir"))
        path = str(tmp_path / "cal.json")
        h.save(path)
        back = Harmoniser.load(path)
        assert back.reference == h.reference
        assert back.adjustments["l8"]["red"].slope == pytest.approx(
            h.adjustments["l8"]["red"].slope
        )
        assert back.adjustments["l8"]["red"].r2 == pytest.approx(
            h.adjustments["l8"]["red"].r2
        )

    def test_merge_fills_only_the_bands_this_season_could_not_fit(self) -> None:
        """Band by band, because the two cases are different.

        A band that fitted cleanly this season keeps its own coefficients: they
        are local in space and time and there is no reason to replace them. A
        band that failed takes the carried ones rather than staying
        uncorrected, since the instrument difference is real and known even
        when this season's pairs were too few to measure it.
        """
        own = Harmoniser(
            reference="s2",
            adjustments={"l8": {"red": BandAdjustment("red", 1.11, -0.004)}},
        )
        prior = Harmoniser(
            reference="s2",
            adjustments={
                "l8": {
                    "red": BandAdjustment("red", 9.99, 9.99),
                    "blue": BandAdjustment("blue", 1.05, 0.002),
                },
                "l9": {"red": BandAdjustment("red", 1.20, -0.012)},
            },
        )
        merged = merge_calibration(own, prior, "2024/25")
        assert merged.adjustments["l8"]["red"].slope == pytest.approx(1.11), (
            "this season's own fit must win"
        )
        assert merged.adjustments["l8"]["blue"].slope == pytest.approx(1.05)
        assert merged.adjustments["l9"]["red"].slope == pytest.approx(1.20)
        assert merged.fitted_on == "2024/25"

    def test_merge_records_nothing_when_it_borrowed_nothing(self) -> None:
        own = Harmoniser(
            reference="s2",
            adjustments={"l8": {"red": BandAdjustment("red", 1.11, -0.004)}},
        )
        merged = merge_calibration(own, own, "2024/25")
        assert merged.fitted_on is None

    def test_borrowed_coefficients_are_named_in_provenance(self, grid: Grid) -> None:
        """A composite corrected with another season's coefficients has to say
        so. The borrowing is defensible; hiding it is not."""
        h = Harmoniser(
            reference="s2",
            adjustments={"l8": {"red": BandAdjustment("red", 1.11, -0.004)}},
            fitted_on="2024/25",
        )
        out = h.apply(make_obs(grid, "l8", dt.datetime(2026, 1, 5)))
        assert out.provenance.detail["calibration_fitted_on"] == "2024/25"
        assert "coefficients from 2024/25" in h.report()

    def test_offset_report_says_when_correction_is_unnecessary(
        self, grid: Grid
    ) -> None:
        """If two sensors already agree, the honest output is 'no correction
        needed', not a cosmetic adjustment applied for the look of rigour."""
        same = np.random.default_rng(5).uniform(0.05, 0.3, grid.shape).astype(np.float32)
        obs = [
            make_obs(grid, "l8", dt.datetime(2025, 1, 8), bands={"red": same}),
            make_obs(grid, "s2", dt.datetime(2025, 1, 8, 8), bands={"red": same}),
        ]
        s = ObservationStack.build(grid, obs)
        rep = sensor_offset_report(find_coincident(s, "s2"), "red")
        assert rep is not None
        assert abs(rep["mean_difference"]) < 1e-7
