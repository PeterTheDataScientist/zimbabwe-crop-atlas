"""Turning an irregular, gappy index series into season metrics.

Why not Savitzky-Golay, which is what the plan originally said.

SG is a moving polynomial fit over a fixed number of neighbouring samples. It
assumes the samples are evenly spaced, because the polynomial is fitted in
sample index rather than in time. Satellite observations of Zimbabwe in the wet
season are the opposite of evenly spaced: findings/01 measured 8.77 clear looks
in December against 1.76 in February. Feeding that to SG treats one February
observation as the same temporal distance from its neighbour as two December
observations a day apart, which smears the December signal across the February
gap and produces a curve that looks smooth precisely where there is no data.
Sawtooth in, smooth curve out, and the smoothness is fabricated.

The Whittaker smoother is used instead. It minimises

    sum_i w_i (y_i - z_i)^2  +  lambda * sum_j (second difference of z)^2

over a regular daily grid, with w_i = 0 wherever there is no observation. The
weights do the work: a day with no observation contributes nothing to the fit
term and is determined entirely by the roughness penalty, so the smoother
interpolates across a gap by curvature rather than by pretending a
neighbouring sample was closer than it was. Gaps stay gaps, and their effect on
the answer is visible in the confidence output rather than hidden in the curve.

It also takes per-observation weights, so an observation built from six clear
looks can be trusted more than one built from a single look. That is the
findings/01 clear-count array feeding directly into the science rather than
sitting in a diagnostic panel.

Reference for the method: Eilers, "A perfect smoother", Analytical Chemistry
2003; applied to vegetation series by Atzberger and Eilers 2011.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# The smoother
# ---------------------------------------------------------------------------


def whittaker(
    y: np.ndarray,
    w: np.ndarray,
    lam: float = 100.0,
    order: int = 2,
) -> np.ndarray:
    """Smooth one series on a regular grid with per-sample weights.

    ``y`` may contain NaN wherever ``w`` is zero; those entries are never read.
    ``lam`` sets the stiffness. Larger is smoother. 100 on a daily grid over a
    six month season is roughly a two week effective window, which is the right
    order for a maize canopy: fast enough to follow emergence, slow enough to
    ignore a single haze-contaminated look.

    Solved with a banded Cholesky. The system is (W + lam * D'D) z = W y, which
    is symmetric positive definite and pentadiagonal for order 2, so it costs
    O(n) rather than the O(n^3) a dense solve would.
    """
    n = y.size
    if w.size != n:
        raise ValueError(f"weights {w.size} do not match series {n}")
    if n < order + 2:
        raise ValueError(f"series of {n} too short for order {order} smoothing")
    if not np.any(w > 0):
        return np.full(n, np.nan, dtype=np.float64)

    yy = np.where(w > 0, np.nan_to_num(y, nan=0.0), 0.0).astype(np.float64)
    ww = np.asarray(w, dtype=np.float64)

    d = np.diff(np.eye(n), n=order, axis=0)
    a = np.diag(ww) + lam * (d.T @ d)
    try:
        z = np.linalg.solve(a, ww * yy)
    except np.linalg.LinAlgError:
        return np.full(n, np.nan, dtype=np.float64)
    return z


def _banded_whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Order-2 Whittaker via a banded solver. Same answer, linear time.

    Used for per-pixel work where the dense version's O(n^2) memory per pixel
    would be the difference between a district running in a minute and not
    running at all.
    """
    from scipy.linalg import solveh_banded

    n = y.size
    yy = np.where(w > 0, np.nan_to_num(y, nan=0.0), 0.0).astype(np.float64)
    ww = np.asarray(w, dtype=np.float64)

    # D'D for the second difference operator is pentadiagonal with a known
    # stencil; the first two and last two rows differ from the interior.
    main = np.full(n, 6.0)
    main[0] = main[-1] = 1.0
    main[1] = main[-2] = 5.0
    off1 = np.full(n - 1, -4.0)
    off1[0] = off1[-1] = -2.0
    off2 = np.full(n - 2, 1.0)

    ab = np.zeros((3, n))
    ab[0, 2:] = lam * off2
    ab[1, 1:] = lam * off1
    ab[2, :] = lam * main + ww
    try:
        return solveh_banded(ab, ww * yy, lower=False)
    except Exception:
        return whittaker(y, w, lam=lam)


# ---------------------------------------------------------------------------
# Irregular observations onto a daily grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesOnGrid:
    """An irregular observation series laid onto a regular daily grid."""

    days: np.ndarray  # day offsets from season start
    values: np.ndarray  # NaN where no observation
    weights: np.ndarray  # 0 where no observation
    season_start: dt.date
    n_observations: int
    observed_days: tuple[int, ...]

    @property
    def gap_days(self) -> int:
        return int((self.weights == 0).sum())

    def largest_gap(self) -> int:
        """Longest run of consecutive days with no observation.

        The number that decides whether a phenology metric is meaningful. A
        peak date derived across a 40 day gap is an interpolation artefact
        wearing a date, and this is how the code knows to say so.
        """
        if not self.observed_days:
            return len(self.days)
        obs = np.array(sorted(self.observed_days))
        edges = np.concatenate(([0], obs, [len(self.days) - 1]))
        return int(np.max(np.diff(edges)))


def to_daily_grid(
    times: Sequence[dt.datetime],
    values: Sequence[float],
    season_start: dt.date,
    season_end: dt.date,
    weights: Sequence[float] | None = None,
) -> SeriesOnGrid:
    """Place irregular observations on a daily grid, without interpolating.

    Two observations on the same day are averaged by weight rather than one
    being dropped. That case is common where Landsat 8 and 9 both pass, or
    where two Sentinel-2 tiles overlap, and it is real extra evidence.
    """
    n_days = (season_end - season_start).days
    if n_days < 2:
        raise ValueError("season must span at least two days")

    acc = np.zeros(n_days, dtype=np.float64)
    wsum = np.zeros(n_days, dtype=np.float64)
    w_in = (
        np.ones(len(times), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    observed: set[int] = set()
    used = 0
    for t, v, wt in zip(times, values, w_in, strict=True):
        if not np.isfinite(v) or wt <= 0:
            continue
        day = (t.date() - season_start).days
        if not 0 <= day < n_days:
            continue
        acc[day] += v * wt
        wsum[day] += wt
        observed.add(day)
        used += 1

    out = np.full(n_days, np.nan, dtype=np.float64)
    hit = wsum > 0
    out[hit] = acc[hit] / wsum[hit]

    return SeriesOnGrid(
        days=np.arange(n_days),
        values=out,
        weights=wsum,
        season_start=season_start,
        n_observations=used,
        observed_days=tuple(sorted(observed)),
    )


# ---------------------------------------------------------------------------
# Season metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phenology:
    """What the season did, with an explicit statement of how well it is known."""

    season_start: dt.date
    start_of_season: int | None
    peak_of_season: int | None
    end_of_season: int | None
    baseline: float
    peak_value: float
    amplitude: float
    integral: float
    # Evidence
    n_observations: int
    largest_gap_days: int
    observations_before_peak: int
    confidence: float
    notes: tuple[str, ...] = ()

    def date_of(self, day: int | None) -> dt.date | None:
        return None if day is None else self.season_start + dt.timedelta(days=int(day))

    @property
    def length_of_season(self) -> int | None:
        if self.start_of_season is None or self.end_of_season is None:
            return None
        return self.end_of_season - self.start_of_season

    def report(self) -> str:
        def d(x: int | None) -> str:
            dd = self.date_of(x)
            return "not detected" if dd is None else f"{dd:%d %b %Y}"

        lines = [
            f"  start of season  {d(self.start_of_season)}",
            f"  peak             {d(self.peak_of_season)}  value {self.peak_value:.3f}",
            f"  end of season    {d(self.end_of_season)}",
            f"  length           {self.length_of_season or 0} days",
            f"  amplitude        {self.amplitude:.3f} over baseline {self.baseline:.3f}",
            f"  integral         {self.integral:.1f}",
            f"  evidence         {self.n_observations} observations, "
            f"largest gap {self.largest_gap_days} days",
            f"  confidence       {self.confidence:.2f}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def extract_phenology(
    series: SeriesOnGrid,
    *,
    lam: float = 100.0,
    threshold: float = 0.2,
    min_amplitude: float = 0.10,
    max_trusted_gap: int = 30,
) -> Phenology:
    """Smooth, then read the season off the smoothed curve.

    ``threshold`` is the fraction of amplitude above baseline at which the
    season is declared started and ended. Twenty percent is the common choice
    and is a convention, not a measurement; it is a parameter so the
    sensitivity can be shown rather than argued.

    ``min_amplitude`` is the guard against reading a season into noise. A pixel
    that never greened by more than 0.10 NDVI over its own baseline did not
    have a crop on it, and returning a confident start date for such a pixel is
    the most common way these products embarrass themselves.
    """
    notes: list[str] = []
    n = series.days.size
    if series.n_observations < 4:
        return _no_season(
            series, "fewer than four observations in the season", notes
        )

    z = _banded_whittaker(series.values, series.weights, lam=lam)
    if not np.all(np.isfinite(z)):
        return _no_season(series, "smoothing did not converge", notes)

    # Baseline from the low tail rather than the minimum: a single bad look at
    # the season edge would otherwise set the baseline and inflate amplitude.
    baseline = float(np.quantile(z, 0.10))
    peak_idx = int(np.argmax(z))
    peak_value = float(z[peak_idx])
    amplitude = peak_value - baseline

    if amplitude < min_amplitude:
        notes.append(
            f"amplitude {amplitude:.3f} below the {min_amplitude:.2f} threshold; "
            "treated as no crop season rather than a weak one"
        )
        return _no_season(series, None, notes, baseline=baseline,
                          peak_value=peak_value, amplitude=amplitude)

    level = baseline + threshold * amplitude
    sos = _rising_crossing(z, peak_idx, level)
    eos = _falling_crossing(z, peak_idx, level)

    integral = float(np.sum(np.clip(z - baseline, 0.0, None)))

    obs_before_peak = sum(1 for d in series.observed_days if d <= peak_idx)
    largest_gap = series.largest_gap()

    confidence = _confidence(
        n_obs=series.n_observations,
        largest_gap=largest_gap,
        obs_before_peak=obs_before_peak,
        n_days=n,
        max_trusted_gap=max_trusted_gap,
    )

    if largest_gap > max_trusted_gap:
        notes.append(
            f"largest observation gap is {largest_gap} days; dates near that "
            "gap are interpolated rather than observed"
        )
    if obs_before_peak < 3:
        notes.append(
            f"only {obs_before_peak} observation(s) before the peak; the start "
            "of season date is weakly constrained"
        )
    if sos is not None and _in_gap(series, sos, radius=7):
        notes.append("start of season falls inside an observation gap")
    if _in_gap(series, peak_idx, radius=7):
        notes.append("peak falls inside an observation gap")

    return Phenology(
        season_start=series.season_start,
        start_of_season=sos,
        peak_of_season=peak_idx,
        end_of_season=eos,
        baseline=baseline,
        peak_value=peak_value,
        amplitude=amplitude,
        integral=integral,
        n_observations=series.n_observations,
        largest_gap_days=largest_gap,
        observations_before_peak=obs_before_peak,
        confidence=confidence,
        notes=tuple(notes),
    )


def _no_season(
    series: SeriesOnGrid,
    reason: str | None,
    notes: list[str],
    *,
    baseline: float = float("nan"),
    peak_value: float = float("nan"),
    amplitude: float = float("nan"),
) -> Phenology:
    if reason:
        notes.append(reason)
    return Phenology(
        season_start=series.season_start,
        start_of_season=None,
        peak_of_season=None,
        end_of_season=None,
        baseline=baseline,
        peak_value=peak_value,
        amplitude=amplitude,
        integral=0.0,
        n_observations=series.n_observations,
        largest_gap_days=series.largest_gap(),
        observations_before_peak=0,
        confidence=0.0,
        notes=tuple(notes),
    )


def _rising_crossing(z: np.ndarray, peak: int, level: float) -> int | None:
    """Last day before the peak at which the curve was still below level."""
    before = z[: peak + 1]
    below = np.nonzero(before < level)[0]
    if below.size == 0:
        return None
    return int(below[-1])


def _falling_crossing(z: np.ndarray, peak: int, level: float) -> int | None:
    after = z[peak:]
    below = np.nonzero(after < level)[0]
    if below.size == 0:
        return None
    return int(peak + below[0])


def _in_gap(series: SeriesOnGrid, day: int, radius: int) -> bool:
    if not series.observed_days:
        return True
    obs = np.array(series.observed_days)
    return bool(np.min(np.abs(obs - day)) > radius)


def _confidence(
    *,
    n_obs: int,
    largest_gap: int,
    obs_before_peak: int,
    n_days: int,
    max_trusted_gap: int,
) -> float:
    """One number in [0, 1] combining the three ways this can be wrong.

    Density: how many looks across the season at all.
    Continuity: whether the biggest hole is small enough to interpolate across.
    Timing: whether the rising limb, which sets the start date, was observed.

    Multiplied rather than averaged, because these are not compensating. Forty
    observations that all fall after the peak do not tell you when the season
    started, and an average would let the density term hide that.
    """
    density = min(1.0, n_obs / (n_days / 10.0))
    continuity = min(1.0, max_trusted_gap / max(largest_gap, 1))
    timing = min(1.0, obs_before_peak / 5.0)
    return round(float(density * continuity * timing), 3)


def phenology_from_composites(
    dates: Sequence[dt.date],
    values: Sequence[float],
    clear_counts: Sequence[int],
    season_start: dt.date,
    season_end: dt.date,
    **kwargs: object,
) -> Phenology:
    """Convenience path from monthly composites straight to season metrics.

    Weights are the clear counts, capped at 6. Capping matters: an unweighted
    fit lets a December pixel with 12 looks dominate a February pixel with 2 so
    completely that February stops influencing the curve at all, which is the
    opposite of what the extra Landsat data was acquired for. Six is where the
    marginal look stops improving a median.
    """
    times = [dt.datetime(d.year, d.month, d.day) for d in dates]
    w = [min(float(c), 6.0) for c in clear_counts]
    grid = to_daily_grid(times, values, season_start, season_end, weights=w)
    return extract_phenology(grid, **kwargs)  # type: ignore[arg-type]
