"""Making Landsat and Sentinel-2 reflectance mean the same thing.

This is the technical centre of the project, and the reason is worth stating
plainly, because it is the step most often skipped.

Landsat 8/9 OLI and Sentinel-2 MSI both measure a band called red. They do not
measure the same red. OLI's red runs roughly 640 to 670 nm, MSI's roughly 650
to 680, with different response shapes inside those windows. Over a green
canopy, where reflectance changes steeply with wavelength, two instruments
looking at the identical field on the identical day return different numbers.
The difference is small, a few percent in most bands, and that is exactly what
makes it dangerous: it is too small to notice by eye and large enough to
produce a fake step in a time series every time the sensor alternates.

Findings/01 established that Landsat is what makes February readable, taking
clear looks from 1.76 to 4.02 and the blind fraction from 16.7% to zero. But a
time series that alternates between two instruments without correction has a
sawtooth in it, and a phenology algorithm reads a sawtooth as real. The
correction is what turns three sensors into one record.

The approach here has three parts, in order of how much they should be
trusted:

1. The mechanism: a per-band linear adjustment, target = slope * source +
   intercept. This is the form used by NASA's Harmonized Landsat Sentinel
   product and by the published cross-calibration literature.
2. Identity coefficients as the default. No adjustment is applied unless
   somebody has supplied coefficients. Silently applying numbers copied from a
   paper written about a different continent is worse than applying none,
   because it looks like rigour.
3. An empirical fitter that derives the coefficients from near-coincident
   acquisitions over the actual study area. Sentinel-2 and Landsat overlap
   within a day or so regularly; where both saw the same pixel within a short
   window and both called it clear, the pair is a calibration point. Thousands
   of such pairs over one season give a local, defensible, reproducible
   adjustment, with a measured scatter that says how well it worked.

Point 3 is the part worth having. It replaces a citation with a measurement.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cropatlas.observation import (
    OPTICAL_BANDS,
    Observation,
    ObservationStack,
)


@dataclass(frozen=True)
class BandAdjustment:
    """target = slope * source + intercept, for one band.

    ``n_pairs`` and ``rmse`` are not decoration. An adjustment fitted on
    forty pixels is not the same object as one fitted on forty thousand, and
    the code that applies it should be able to refuse the first.
    """

    band: str
    slope: float
    intercept: float
    n_pairs: int = 0
    rmse: float | None = None
    r2: float | None = None
    estimator: str = "deming"
    # The OLS slope on the same data. Kept because the gap between the two is
    # the measured size of the attenuation bias, which is worth reporting: a
    # large gap means the pair scatter is large and the calibration is
    # correspondingly less certain, whatever the estimator says.
    ols_slope: float | None = None

    @property
    def is_identity(self) -> bool:
        return abs(self.slope - 1.0) < 1e-9 and abs(self.intercept) < 1e-9

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values * np.float32(self.slope) + np.float32(self.intercept)).astype(
            np.float32
        )

    def __str__(self) -> str:
        base = f"{self.band}: x{self.slope:.4f} {self.intercept:+.5f}"
        if self.n_pairs:
            base += f" (n={self.n_pairs:,}"
            if self.rmse is not None:
                base += f", rmse={self.rmse:.4f}"
            if self.r2 is not None:
                base += f", r2={self.r2:.3f}"
            if self.ols_slope is not None:
                base += f", ols={self.ols_slope:.4f}"
            base += ")"
        return base


@dataclass(frozen=True)
class Harmoniser:
    """Adjustments from one sensor's basis onto a reference sensor's basis.

    Sentinel-2 is the reference by default, for two reasons: it has the finest
    resolution of the three, and it is the only one carrying red edge, so
    adjusting it onto Landsat would mean the reference basis had bands the
    reference could not measure.
    """

    reference: str
    adjustments: Mapping[str, Mapping[str, BandAdjustment]]
    # Which season's coincident pairs produced these coefficients. None means
    # this season's own. Anything else has to reach the output as a caveat.
    fitted_on: str | None = None

    @classmethod
    def identity(cls, reference: str = "s2") -> Harmoniser:
        """No adjustment. The honest default before anything is measured."""
        return cls(reference=reference, adjustments={})

    def for_sensor(self, sensor: str) -> Mapping[str, BandAdjustment]:
        if sensor == self.reference:
            return {}
        return self.adjustments.get(sensor, {})

    def is_calibrated(self, sensor: str) -> bool:
        return bool(self.for_sensor(sensor))

    def apply(self, obs: Observation) -> Observation:
        """Return the observation on the reference basis.

        An uncalibrated sensor passes through unchanged rather than raising.
        That is a deliberate choice: refusing to composite because calibration
        is missing would make the package useless on day one, and the fact that
        an adjustment was not applied is recorded in provenance so it appears
        in the output rather than being lost.
        """
        adj = self.for_sensor(obs.sensor)
        if not adj:
            return obs
        from dataclasses import replace

        bands = dict(obs.bands)
        applied = []
        for name, a in adj.items():
            if name in bands and not a.is_identity:
                bands[name] = a.apply(bands[name])
                applied.append(name)
        if not applied:
            return obs
        detail = dict(obs.provenance.detail)
        detail["harmonised_to"] = self.reference
        detail["harmonised_bands"] = ",".join(sorted(applied))
        if self.fitted_on:
            detail["calibration_fitted_on"] = self.fitted_on
        return replace(
            obs,
            bands=bands,
            provenance=replace(obs.provenance, detail=detail),
        )

    # ------------------------------------------------------------------
    # Persistence, and why a calibration should outlive the season that fit it
    # ------------------------------------------------------------------
    #
    # The first version fitted the adjustment inside each season and used it
    # only there. That is defensible for locality: coefficients fitted over
    # Zimbabwe are better than coefficients fitted over Maryland.
    #
    # A live run showed the cost. The 2024/25 season produced 334,181 usable
    # coincident pixel pairs and fitted five bands cleanly. The 2025/26 season
    # produced 66 coincident scene pairs but only 130,992 overlapping valid
    # pixels, because cloud fell differently, and every band failed the
    # r-squared floor. The season with the best sensor coverage in four years
    # therefore ran completely uncalibrated, with measured residual offsets of
    # +0.038 in blue and +0.020 in red left in the data.
    #
    # The assumption behind per-season fitting is wrong. The difference between
    # OLI and MSI is a property of the two instruments' spectral response
    # functions. It does not change between one growing season and the next.
    # Fitting per season buys locality in space, which matters, and locality in
    # time, which does not.
    #
    # So a fitted harmoniser can be saved and reused, and a season that cannot
    # fit its own is not left uncorrected. What must never be lost is the
    # provenance: a composite corrected with coefficients from another season
    # has to say so, which is why ``fitted_on`` travels with the object and
    # into every caveat.

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "fitted_on": self.fitted_on,
            "adjustments": {
                sensor: {
                    band: {
                        "slope": a.slope,
                        "intercept": a.intercept,
                        "n_pairs": a.n_pairs,
                        "rmse": a.rmse,
                        "r2": a.r2,
                        "estimator": a.estimator,
                        "ols_slope": a.ols_slope,
                    }
                    for band, a in bands.items()
                }
                for sensor, bands in self.adjustments.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Harmoniser:
        return cls(
            reference=str(data["reference"]),
            fitted_on=data.get("fitted_on"),
            adjustments={
                sensor: {
                    band: BandAdjustment(band=band, **d)
                    for band, d in bands.items()
                }
                for sensor, bands in data.get("adjustments", {}).items()
            },
        )

    def save(self, path: str) -> None:
        import json

        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> Harmoniser:
        import json

        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def report(self) -> str:
        if not self.adjustments:
            return f"identity harmoniser (reference {self.reference}, nothing fitted)"
        origin = (
            "" if self.fitted_on is None else f", coefficients from {self.fitted_on}"
        )
        lines = [f"harmonised onto {self.reference}{origin}"]
        for sensor, bands in sorted(self.adjustments.items()):
            lines.append(f"  {sensor}:")
            for band in OPTICAL_BANDS:
                if band in bands:
                    lines.append(f"    {bands[band]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fitting the adjustment from coincident acquisitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoincidentPair:
    """Two observations of the same ground within a short window."""

    reference: Observation
    other: Observation

    @property
    def hours_apart(self) -> float:
        return abs(
            (self.reference.acquired - self.other.acquired).total_seconds() / 3600.0
        )


def find_coincident(
    stack: ObservationStack,
    reference_sensor: str,
    max_hours: float = 30.0,
) -> list[CoincidentPair]:
    """Pairs of observations close enough in time to show the same surface.

    Thirty hours by default rather than the tempting 'same day'. Sentinel-2
    crosses at about 10:30 local and Landsat at about 10:00, so a genuine
    same-orbit-cycle pair can straddle midnight UTC and a calendar day test
    drops it. Thirty hours keeps those and still excludes anything that could
    have been rained on in between.

    The pairing is deliberately many to many rather than nearest neighbour: if
    two Landsat scenes both fall within the window of one Sentinel-2 scene,
    both are real calibration evidence and discarding one to make the pairing
    tidy throws away data for no reason.
    """
    ref = [o for o in stack if o.sensor == reference_sensor and not o.is_radar]
    others = [o for o in stack if o.sensor != reference_sensor and not o.is_radar]
    pairs = []
    for r in ref:
        for o in others:
            pair = CoincidentPair(r, o)
            if pair.hours_apart <= max_hours:
                pairs.append(pair)
    return pairs


def fit_band(
    pairs: Sequence[CoincidentPair],
    band: str,
    *,
    min_pairs: int = 500,
    min_r2: float = 0.70,
    sample_per_pair: int = 20_000,
    trim: float = 0.02,
    estimator: str = "deming",
    rng: np.random.Generator | None = None,
) -> BandAdjustment | None:
    """Least squares fit of target = slope * source + intercept for one band.

    Only pixels both instruments called clear contribute. That is the whole
    trick: a cloud-contaminated pixel in either scene would drag the fit, and
    the validity masks already encode which pixels each instrument stands
    behind.

    ``trim`` drops the extreme tails of the residual distribution before the
    final fit. Even with both masks agreeing, a small population of pairs is
    genuinely different surface: a cloud edge one classifier missed, a field
    burned between the two passes, water level moved. Those are real changes,
    not calibration differences, and they belong out of a calibration fit. Two
    percent from each tail, applied once, and reported.

    Returns None rather than a bad fit when there is not enough evidence.
    A caller that gets None applies no adjustment, which is correct behaviour;
    a caller that gets a fit from nine pixels and applies it is not.
    """
    rng = rng or np.random.default_rng(20260905)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for pair in pairs:
        if not (pair.reference.has(band) and pair.other.has(band)):
            continue
        both = pair.reference.valid & pair.other.valid
        if not both.any():
            continue
        src = pair.other.band(band)[both]
        tgt = pair.reference.band(band)[both]
        keep = np.isfinite(src) & np.isfinite(tgt)
        src, tgt = src[keep], tgt[keep]
        if src.size > sample_per_pair:
            idx = rng.choice(src.size, sample_per_pair, replace=False)
            src, tgt = src[idx], tgt[idx]
        if src.size:
            xs.append(src)
            ys.append(tgt)

    if not xs:
        return None
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    if x.size < min_pairs:
        return None

    fit = _deming if estimator == "deming" else _ols
    slope, intercept = fit(x, y)
    if trim > 0 and x.size > 100:
        resid = y - (slope * x + intercept)
        lo, hi = np.quantile(resid, [trim, 1.0 - trim])
        keep = (resid >= lo) & (resid <= hi)
        if keep.sum() >= min_pairs:
            x, y = x[keep], y[keep]
            slope, intercept = fit(x, y)

    ols_slope, _ = _ols(x, y)
    pred = slope * x + intercept
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    # Refuse a fit the data cannot support.
    #
    # This guard was added after a live run, and the reason is specific.
    # Deming corrects the attenuation that noise in the predictor causes, and
    # the size of that correction scales inversely with how reliable the
    # predictor is. When the pair scatter is small the correction is gentle and
    # right. When the scatter is large the correction is enormous, and it is
    # only right if the assumed ratio of the two error variances is right.
    #
    # On real data over Harare the blue band fitted at r-squared 0.60 and
    # Deming returned slopes of 1.51 for Landsat 8 and 1.82 for Landsat 9.
    # Neither is a physical cross-sensor difference; blue is where the two
    # atmospheric corrections disagree most and where the equal-noise
    # assumption is least defensible, so the estimator was amplifying
    # disagreement rather than removing bias.
    #
    # The honest answer for such a band is that it cannot be calibrated from
    # this evidence. Returning None leaves it uncorrected and, because the
    # pipeline records uncalibrated sensors in every composite's caveats, says
    # so in the output. An adjustment that injects more error than it removes
    # is worse than none, and it is worse precisely because it looks like
    # diligence.
    if r2 is not None and r2 < min_r2:
        return None

    return BandAdjustment(
        band=band,
        slope=float(slope),
        intercept=float(intercept),
        n_pairs=int(x.size),
        rmse=rmse,
        r2=r2,
        estimator=estimator,
        ols_slope=float(ols_slope),
    )


def _deming(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> tuple[float, float]:
    """Errors-in-variables regression. The statistically correct estimator here.

    Ordinary least squares assumes the predictor is known exactly and all the
    error is in the response. That assumption is false in this application and
    the falsehood is not academic.

    Both sides of a cross-sensor pair are measurements. Landsat at 30 m and
    Sentinel-2 at 10 m are each resampled onto a common grid, they are
    geolocated to within a fraction of a pixel rather than exactly, they are
    acquired up to thirty hours apart so the surface may genuinely have
    changed, and each carries its own atmospheric correction error. There is
    substantial noise in x.

    Noise in the predictor biases the OLS slope toward zero. The bias is
    exactly the attenuation factor, so an OLS slope is the true slope
    multiplied by roughly the reliability of x, and the worse the scatter the
    flatter the fitted line. That produces the characteristic signature seen in
    the first run of this pipeline: Landsat 9 NIR fitted at slope 0.64 with an
    r-squared of 0.62, while Landsat 8 NIR on the same season fitted at 0.86
    with r-squared 0.79. Two nominally identical instruments cannot differ from
    Sentinel-2 by that much. The slopes were tracking the noise, not the
    radiometry.

    Deming regression minimises perpendicular distance instead of vertical
    distance, and is unbiased when the two error variances are in the assumed
    ratio ``lam``. With ``lam = 1`` it is orthogonal regression: appropriate
    here because there is no reason to believe either instrument is markedly
    noisier than the other after both have been resampled to the same grid.

    The closed form for lam = 1 is standard; the general form is carried so the
    assumption is visible and adjustable rather than hidden in a constant.
    """
    n = x.size
    if n < 2:
        return 1.0, 0.0
    xbar = float(x.mean())
    ybar = float(y.mean())
    dx = x - xbar
    dy = y - ybar
    sxx = float(np.dot(dx, dx)) / n
    syy = float(np.dot(dy, dy)) / n
    sxy = float(np.dot(dx, dy)) / n

    scale = max(abs(xbar), 1e-6)
    if sxx <= (scale * 1e-7) ** 2:
        return 1.0, 0.0
    if abs(sxy) < 1e-15:
        # No covariance: perpendicular regression has no defined direction to
        # prefer. Refuse rather than return whichever root the arithmetic
        # happens to produce.
        return 1.0, 0.0

    a = syy - lam * sxx
    slope = (a + float(np.sqrt(a * a + 4.0 * lam * sxy * sxy))) / (2.0 * sxy)
    return float(slope), float(ybar - slope * xbar)


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares in centred form.

    Written out rather than pulled from a library for two reasons: it keeps
    scipy off the dependency path for the core, and the degenerate case has to
    be refused explicitly, where numpy's polyfit emits a warning and returns a
    number anyway.

    Centred rather than the textbook sums-of-products form, and that is not a
    stylistic choice. The raw form computes ``n * sum(x^2) - sum(x)^2``, which
    for reflectance values clustered near 0.2 over tens of thousands of pixels
    subtracts two numbers agreeing to eleven significant figures. The true
    answer is zero when x is constant, and floating point returns something
    near 1e-8 instead: not small enough to trip an absolute guard, and large
    enough to divide an equally meaningless numerator by and return a slope of
    -0.0. A test caught exactly that.

    The centred form computes ``sum((x - xbar)^2)`` directly, which loses no
    precision to cancellation and makes the degeneracy test meaningful: compare
    the spread in x against the scale of x rather than against an absolute
    constant that means different things for reflectance and for decibels.
    """
    n = x.size
    if n < 2:
        return 1.0, 0.0
    xbar = float(x.mean())
    ybar = float(y.mean())
    dx = x - xbar
    sxx = float(np.dot(dx, dx))

    # Relative test: is the spread in x negligible compared with x itself?
    scale = max(abs(xbar), 1e-6)
    if sxx <= n * (scale * 1e-7) ** 2:
        return 1.0, 0.0

    slope = float(np.dot(dx, y - ybar)) / sxx
    return slope, ybar - slope * xbar


def fit_harmoniser(
    stack: ObservationStack,
    reference_sensor: str = "s2",
    bands: Iterable[str] = OPTICAL_BANDS,
    *,
    max_hours: float = 30.0,
    min_pairs: int = 500,
    min_r2: float = 0.70,
    estimator: str = "deming",
    label: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Harmoniser, dict[str, object]]:
    """Fit adjustments for every non-reference sensor in a stack.

    Returns the harmoniser and a diagnostic record. The record is returned
    separately rather than folded in because it belongs in the findings
    directory, not in the object that gets applied to pixels.
    """
    pairs = find_coincident(stack, reference_sensor, max_hours=max_hours)
    by_sensor: dict[str, list[CoincidentPair]] = {}
    for p in pairs:
        by_sensor.setdefault(p.other.sensor, []).append(p)

    adjustments: dict[str, dict[str, BandAdjustment]] = {}
    diagnostics: dict[str, object] = {
        "reference": reference_sensor,
        "max_hours": max_hours,
        "estimator": estimator,
        "min_r2": min_r2,
        "pair_count": {k: len(v) for k, v in by_sensor.items()},
        "fitted": {},
        "skipped": {},
    }

    for sensor, sensor_pairs in by_sensor.items():
        fitted: dict[str, BandAdjustment] = {}
        skipped: list[str] = []
        for band in bands:
            adj = fit_band(
                sensor_pairs, band, min_pairs=min_pairs, min_r2=min_r2,
                estimator=estimator, rng=rng,
            )
            if adj is None:
                skipped.append(band)
            else:
                fitted[band] = adj
        if fitted:
            adjustments[sensor] = fitted
        diagnostics["fitted"][sensor] = {  # type: ignore[index]
            b: {
                "slope": a.slope,
                "intercept": a.intercept,
                "n": a.n_pairs,
                "rmse": a.rmse,
                "r2": a.r2,
                "ols_slope": a.ols_slope,
            }
            for b, a in fitted.items()
        }
        diagnostics["skipped"][sensor] = skipped  # type: ignore[index]

    return (
        Harmoniser(
            reference=reference_sensor,
            adjustments=adjustments,
            fitted_on=None,
        ),
        {**diagnostics, "label": label},
    )


def harmonise_stack(stack: ObservationStack, harmoniser: Harmoniser) -> ObservationStack:
    from dataclasses import replace

    return replace(
        stack,
        observations=tuple(harmoniser.apply(o) for o in stack.observations),
    )


def sensor_offset_report(
    pairs: Sequence[CoincidentPair], band: str
) -> dict[str, float] | None:
    """The raw difference between two sensors on one band.

    Run before fitting, this says whether harmonisation was needed at all: if
    the mean difference on red is 0.0004 reflectance then the correction is
    noise and should be reported as unnecessary rather than applied for the
    look of the thing.

    Run after fitting, it is a much weaker check than it appears, and the
    weakness is worth stating because the number looks so convincing. Any
    least-squares-family fit passes through the joint mean of the data by
    construction, so the mean residual is driven to near zero whether the slope
    is right or badly wrong. A post-fit mean difference of 0.001 therefore
    confirms only that the fit did not diverge. The statistics that carry real
    information about fit quality are the RMSE and the r-squared, and the
    honest summary of a band is the pair of them, not the mean residual.
    """
    diffs: list[np.ndarray] = []
    for pair in pairs:
        if not (pair.reference.has(band) and pair.other.has(band)):
            continue
        both = pair.reference.valid & pair.other.valid
        if not both.any():
            continue
        d = pair.reference.band(band)[both] - pair.other.band(band)[both]
        d = d[np.isfinite(d)]
        if d.size:
            diffs.append(d)
    if not diffs:
        return None
    all_d = np.concatenate(diffs)
    return {
        "n": int(all_d.size),
        "mean_difference": float(all_d.mean()),
        "median_difference": float(np.median(all_d)),
        "std": float(all_d.std()),
        "p05": float(np.quantile(all_d, 0.05)),
        "p95": float(np.quantile(all_d, 0.95)),
    }


def coincidence_windows(
    stack: ObservationStack, reference_sensor: str = "s2"
) -> list[tuple[dt.date, str, str, float]]:
    """Every coincident pair as a readable row. For the findings write-up."""
    rows = []
    for p in find_coincident(stack, reference_sensor, max_hours=48.0):
        rows.append(
            (
                p.reference.acquired.date(),
                p.reference.sensor,
                p.other.sensor,
                round(p.hours_apart, 1),
            )
        )
    return sorted(rows)


def merge_calibration(
    own: Harmoniser, fallback: Harmoniser, fallback_label: str
) -> Harmoniser:
    """Fill the bands this season could not fit from a season that could.

    Band by band rather than all or nothing, because the two failures are
    different. A band that fitted cleanly this season should keep its own
    coefficients: they are local in space and time and there is no reason to
    replace them. A band that failed should take the carried ones rather than
    stay uncorrected, since the instrument difference is real and known even
    when this season's pairs were too few or too cloudy to measure it.

    The result records ``fitted_on`` whenever any carried coefficient was used,
    so a composite built from it says in its caveats that part of its
    calibration came from elsewhere.
    """
    merged: dict[str, dict[str, BandAdjustment]] = {
        sensor: dict(bands) for sensor, bands in own.adjustments.items()
    }
    borrowed = False
    for sensor, bands in fallback.adjustments.items():
        target = merged.setdefault(sensor, {})
        for band, adj in bands.items():
            if band not in target:
                target[band] = adj
                borrowed = True
    merged = {s: b for s, b in merged.items() if b}
    return Harmoniser(
        reference=own.reference,
        adjustments=merged,
        fitted_on=fallback_label if borrowed else own.fitted_on,
    )
