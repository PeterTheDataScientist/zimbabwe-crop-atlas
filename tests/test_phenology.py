from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from cropatlas.phenology import (
    _banded_whittaker,
    extract_phenology,
    to_daily_grid,
    whittaker,
)

SEASON_START = dt.date(2024, 11, 1)
SEASON_END = dt.date(2025, 5, 1)
N_DAYS = (SEASON_END - SEASON_START).days

# A maize season shaped like the real thing: planted with the November rains,
# tasselling in February, senescing through April.
TRUE_SOS_DAY = 25  # 26 November
TRUE_PEAK_DAY = 100  # 9 February
TRUE_EOS_DAY = 160  # 10 April


def true_curve(days: np.ndarray) -> np.ndarray:
    """Double logistic: the standard functional form for a crop season.

    Green-up as a rising logistic, senescence as a falling one. Baseline 0.18
    is bare Zimbabwean soil in NDVI, peak 0.78 is a healthy closed maize
    canopy.
    """
    green_up = 1.0 / (1.0 + np.exp(-0.16 * (days - TRUE_SOS_DAY - 18)))
    senesce = 1.0 / (1.0 + np.exp(0.13 * (days - TRUE_EOS_DAY + 12)))
    return 0.18 + 0.60 * np.minimum(green_up, senesce)


def sample(
    observed_days: list[int], noise: float = 0.0, seed: int = 42
) -> tuple[list[dt.datetime], list[float]]:
    rng = np.random.default_rng(seed)
    days = np.array(observed_days, dtype=float)
    vals = true_curve(days)
    if noise:
        vals = vals + rng.normal(0, noise, vals.size)
    times = [
        dt.datetime.combine(SEASON_START, dt.time(9, 0)) + dt.timedelta(days=int(d))
        for d in observed_days
    ]
    return times, [float(v) for v in vals]


def dense_days() -> list[int]:
    return list(range(2, N_DAYS - 2, 5))


def realistic_days() -> list[int]:
    """Observation dates with the measured February hole in them.

    Findings/01: 8.77 mean clear looks in December against 1.76 in February.
    Reproducing that imbalance is the whole point of the test, because a
    smoother that only works on evenly spaced data passes on a regular grid and
    fails on Zimbabwe.
    """
    nov = [3, 8, 13, 19, 26]
    dec = [31, 33, 36, 39, 42, 45, 48, 51, 55, 58]
    jan = [64, 71, 79, 86]
    feb = [98]  # one look in the month that decides the yield
    mar = [125, 133, 140]
    apr = [152, 157, 162, 167, 172, 177]
    return nov + dec + jan + feb + mar + apr


class TestWhittaker:
    def test_recovers_a_smooth_curve_through_noise(self) -> None:
        days = np.arange(N_DAYS, dtype=float)
        truth = true_curve(days)
        rng = np.random.default_rng(1)
        noisy = truth + rng.normal(0, 0.05, N_DAYS)
        smoothed = whittaker(noisy, np.ones(N_DAYS), lam=100.0)
        assert np.abs(smoothed - truth).mean() < 0.02

    def test_zero_weight_samples_are_never_read(self) -> None:
        """The gap-handling contract: a day with no observation contributes
        nothing to the fit and is determined by curvature alone. If the value
        at a zero-weight day leaked into the answer, poisoning it would move
        the result."""
        days = np.arange(N_DAYS, dtype=float)
        truth = true_curve(days)
        w = np.ones(N_DAYS)
        w[80:110] = 0.0  # the February hole

        clean = truth.copy()
        clean[80:110] = np.nan
        poisoned = truth.copy()
        poisoned[80:110] = -999.0

        a = whittaker(clean, w, lam=100.0)
        b = whittaker(poisoned, w, lam=100.0)
        assert np.allclose(a, b, equal_nan=True)

    def test_interpolates_across_a_gap_by_curvature(self) -> None:
        days = np.arange(N_DAYS, dtype=float)
        truth = true_curve(days)
        w = np.ones(N_DAYS)
        w[85:105] = 0.0
        y = np.where(w > 0, truth, np.nan)
        z = whittaker(y, w, lam=50.0)
        # Across a 20 day hole on a smooth curve the smoother should stay close.
        assert np.abs(z[85:105] - truth[85:105]).max() < 0.05

    def test_all_zero_weights_returns_nan_not_zeros(self) -> None:
        """A pixel nobody ever saw must come back as unknown, not as a flat
        line at zero that a map would render as bare ground."""
        z = whittaker(np.full(50, np.nan), np.zeros(50), lam=10.0)
        assert np.isnan(z).all()

    def test_banded_and_dense_solvers_agree(self) -> None:
        """The banded path is what makes per-pixel work feasible. If it drifts
        from the reference implementation every district number is wrong in a
        way no other test would catch."""
        days = np.arange(N_DAYS, dtype=float)
        rng = np.random.default_rng(2)
        y = true_curve(days) + rng.normal(0, 0.04, N_DAYS)
        w = np.ones(N_DAYS)
        w[70:95] = 0.0
        y = np.where(w > 0, y, np.nan)
        assert np.allclose(
            whittaker(y, w, lam=100.0), _banded_whittaker(y, w, 100.0), atol=1e-8
        )

    def test_higher_lambda_is_smoother(self) -> None:
        rng = np.random.default_rng(3)
        y = true_curve(np.arange(N_DAYS, dtype=float)) + rng.normal(0, 0.06, N_DAYS)
        w = np.ones(N_DAYS)
        rough = np.abs(np.diff(whittaker(y, w, lam=5.0), 2)).sum()
        smooth = np.abs(np.diff(whittaker(y, w, lam=500.0), 2)).sum()
        assert smooth < rough


class TestDailyGrid:
    def test_places_observations_on_the_right_days(self) -> None:
        times, vals = sample([0, 10, 20])
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        assert g.n_observations == 3
        assert g.observed_days == (0, 10, 20)
        assert np.isnan(g.values[5])
        assert g.weights[10] == 1.0

    def test_two_sensors_on_one_day_are_averaged_not_dropped(self) -> None:
        """Landsat 8 and 9 both passing, or two overlapping S2 tiles. That is
        real extra evidence and discarding one to keep the grid tidy throws it
        away."""
        t = dt.datetime.combine(SEASON_START, dt.time(9)) + dt.timedelta(days=30)
        g = to_daily_grid(
            [t, t + dt.timedelta(hours=3)], [0.4, 0.6], SEASON_START, SEASON_END
        )
        assert g.values[30] == pytest.approx(0.5)
        assert g.weights[30] == pytest.approx(2.0)
        assert g.n_observations == 2

    def test_weights_are_honoured_in_the_average(self) -> None:
        t = dt.datetime.combine(SEASON_START, dt.time(9)) + dt.timedelta(days=30)
        g = to_daily_grid(
            [t, t], [0.4, 0.6], SEASON_START, SEASON_END, weights=[3.0, 1.0]
        )
        assert g.values[30] == pytest.approx(0.45)

    def test_observations_outside_the_season_are_ignored(self) -> None:
        early = dt.datetime(2024, 9, 1)
        late = dt.datetime(2025, 8, 1)
        inside = dt.datetime(2024, 12, 15)
        g = to_daily_grid([early, inside, late], [0.3, 0.5, 0.4], SEASON_START, SEASON_END)
        assert g.n_observations == 1

    def test_nan_values_do_not_count_as_observations(self) -> None:
        times, _ = sample([5, 10, 15])
        g = to_daily_grid(times, [0.3, float("nan"), 0.5], SEASON_START, SEASON_END)
        assert g.n_observations == 2
        assert g.weights[10] == 0.0

    def test_largest_gap_finds_the_february_hole(self) -> None:
        times, vals = sample(realistic_days())
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        assert g.largest_gap() == 27, "86 Jan to 98 Feb to 125 Mar, the 27 day hole"


class TestPhenologyExtraction:
    def test_recovers_the_season_from_dense_observations(self) -> None:
        times, vals = sample(dense_days())
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        assert ph.peak_of_season is not None
        assert abs(ph.peak_of_season - TRUE_PEAK_DAY) <= 12
        assert abs(ph.start_of_season - TRUE_SOS_DAY) <= 12
        assert ph.amplitude == pytest.approx(0.60, abs=0.06)
        assert ph.confidence > 0.8

    def test_survives_the_measured_february_gap(self) -> None:
        """The test that matters. One February look, a 27 day hole either side,
        and the peak still lands in the right fortnight."""
        times, vals = sample(realistic_days(), noise=0.02)
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        assert ph.peak_of_season is not None
        assert abs(ph.peak_of_season - TRUE_PEAK_DAY) <= 20
        assert ph.largest_gap_days == 27

    def test_reports_low_confidence_when_the_gap_is_large(self) -> None:
        sparse = [5, 20, 40, 150, 165, 175]
        times, vals = sample(sparse)
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        assert ph.confidence < 0.35
        assert any("gap" in n for n in ph.notes)

    def test_refuses_to_find_a_season_in_flat_ground(self) -> None:
        """The most common way these products embarrass themselves: a
        confident start-of-season date for a pixel that never grew anything."""
        times = [
            dt.datetime.combine(SEASON_START, dt.time(9)) + dt.timedelta(days=d)
            for d in dense_days()
        ]
        rng = np.random.default_rng(9)
        flat = [0.19 + float(rng.normal(0, 0.01)) for _ in times]
        g = to_daily_grid(times, flat, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        assert ph.start_of_season is None
        assert ph.peak_of_season is None
        assert ph.confidence == 0.0
        assert any("no crop season" in n for n in ph.notes)

    def test_too_few_observations_is_not_a_season(self) -> None:
        times, vals = sample([10, 60, 120])
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        assert ph.peak_of_season is None
        assert any("fewer than four" in n for n in ph.notes)

    def test_confidence_multiplies_rather_than_averages(self) -> None:
        """Forty observations that all fall after the peak do not tell you when
        the season started. An averaged confidence would let the density term
        hide that; a multiplied one cannot."""
        late_only = list(range(120, 178, 2))
        times, vals = sample(late_only)
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        assert g.n_observations > 25
        assert ph.confidence < 0.5

    def test_integral_tracks_a_better_season(self) -> None:
        """The productivity proxy has to order two seasons correctly or it is
        worth nothing."""
        days = np.array(dense_days(), dtype=float)
        times = [
            dt.datetime.combine(SEASON_START, dt.time(9)) + dt.timedelta(days=int(d))
            for d in days
        ]
        good = to_daily_grid(times, list(true_curve(days)), SEASON_START, SEASON_END)
        poor_vals = 0.18 + 0.45 * (true_curve(days) - 0.18) / 0.60
        poor = to_daily_grid(times, list(poor_vals), SEASON_START, SEASON_END)
        assert extract_phenology(good).integral > extract_phenology(poor).integral

    def test_dates_convert_back_to_calendar_days(self) -> None:
        times, vals = sample(dense_days())
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        peak = ph.date_of(ph.peak_of_season)
        assert peak is not None
        assert peak.month in (1, 2, 3), f"peak landed in month {peak.month}"
        assert "peak" in ph.report()

    def test_length_of_season_is_none_when_an_end_is_not_reached(self) -> None:
        """A season still green at the last observation has no end date, and
        inventing one from the edge of the window would be fabrication."""
        rising = list(range(2, 100, 5))
        times, vals = sample(rising)
        g = to_daily_grid(times, vals, SEASON_START, SEASON_END)
        ph = extract_phenology(g)
        if ph.end_of_season is None:
            assert ph.length_of_season is None
