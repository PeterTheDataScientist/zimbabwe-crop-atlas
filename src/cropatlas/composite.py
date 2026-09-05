"""Reducing many partly cloudy looks into one usable image.

The central measurement in findings/01 was that scene level cloud cover is the
wrong statistic. A February Sentinel-2 scene over Harare reports 52% cloud, and
the instinct is to discard it. But the 48% that is clear may be exactly the
district you care about, and after three such scenes most of the district has
been seen at least once even though no single scene was usable.

So nothing here ever discards a scene. Every scene contributes its clear pixels
and nothing else, and the output records how many looks each pixel actually
got. That count is not a diagnostic bolted on afterwards. It is the honesty of
the product: a composite pixel built from eight clear looks and one built from
a single look are different in kind, and a map that renders them identically is
lying by omission.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cropatlas.observation import (
    Grid,
    ObservationStack,
    stack_bands,
)

Reducer = Literal["median", "mean", "max", "min", "first", "last"]


@dataclass(frozen=True)
class CompositeProvenance:
    """What went into a composite, in enough detail to defend a number.

    Findings/02 made this mandatory. The sensor set changes between seasons, so
    a user comparing a 2024 composite with a 2026 one has to be told that one
    of them had radar available and the other did not, and that they are
    therefore not the same measurement.
    """

    start: dt.datetime
    end: dt.datetime
    sensors: tuple[str, ...]
    scenes_by_sensor: Mapping[str, int]
    acquisition_days: int
    reducer: str
    harmonised_to: str | None = None
    uncalibrated_sensors: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def caveats(self) -> list[str]:
        """Everything a reader needs to know before trusting the picture."""
        out = list(self.notes)
        if self.uncalibrated_sensors:
            out.append(
                "no cross-sensor calibration applied for "
                + ", ".join(self.uncalibrated_sensors)
                + "; values from those sensors are on their own radiometric basis"
            )
        if len(self.sensors) == 1:
            out.append(f"single sensor ({self.sensors[0]}); no cross-sensor gain")
        return out

    def __str__(self) -> str:
        by = ", ".join(f"{k}={v}" for k, v in sorted(self.scenes_by_sensor.items()))
        return (
            f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}, "
            f"{self.acquisition_days} acquisition days ({by}), {self.reducer}"
        )


@dataclass(frozen=True)
class Composite:
    """One reduced image plus the evidence for it."""

    grid: Grid
    bands: Mapping[str, np.ndarray]
    clear_count: np.ndarray
    provenance: CompositeProvenance

    def band(self, name: str) -> np.ndarray:
        if name not in self.bands:
            raise KeyError(
                f"composite has no {name!r}; it carries "
                f"{', '.join(sorted(self.bands))}"
            )
        return self.bands[name]

    @property
    def blind_fraction(self) -> float:
        """Share of the grid never seen clearly by any sensor in the window.

        The number findings/01 was built on, and the one that decides whether a
        monthly map is publishable. 16.7% for Sentinel-2 alone in February 2025,
        0.0% once Landsat joined.
        """
        return float((self.clear_count == 0).mean())

    def coverage(self, at_least: int = 1) -> float:
        return float((self.clear_count >= at_least).mean())

    @property
    def mean_looks(self) -> float:
        return float(self.clear_count.mean())

    def confidence(self) -> np.ndarray:
        """Per-pixel confidence in [0, 1] from observation density alone.

        Deliberately crude and deliberately explicit. Three clear looks is
        taken as full confidence for a monthly composite, because with three
        looks a median has a real chance of rejecting one bad pixel and with
        two it does not. This is a convention, it is stated here, and it is one
        number to change rather than a judgement buried in a rendering
        function.
        """
        return np.clip(self.clear_count.astype(np.float32) / 3.0, 0.0, 1.0)

    def report(self) -> str:
        lines = [
            str(self.provenance),
            f"  mean clear looks   {self.mean_looks:.2f}",
            f"  never seen         {self.blind_fraction * 100:.1f}%",
            f"  3 or more looks    {self.coverage(3) * 100:.1f}%",
        ]
        for c in self.provenance.caveats():
            lines.append(f"  caveat: {c}")
        return "\n".join(lines)


def composite(
    stack: ObservationStack,
    bands: Sequence[str],
    *,
    reducer: Reducer = "median",
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    harmonised_to: str | None = None,
    uncalibrated: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> Composite:
    """Reduce a stack to one image, per pixel, over the requested bands.

    Median by default. Not mean, because the residual after cloud masking is
    almost always thin cloud and haze that the classifier missed, and those
    push reflectance up; a median with three or more looks rejects them and a
    mean averages them in. Not max-NDVI either, which is the classic choice for
    annual composites but biases toward the greenest day in the window and so
    systematically overstates a struggling crop.

    Only observations carrying every requested band contribute. A radar scene
    in the stack is therefore silently ignored by an optical composite, which
    is correct: gamma-0 decibels have no business being medianed with
    reflectance.
    """
    usable = stack.with_bands(*bands)
    if start is not None or end is not None:
        lo = start or dt.datetime.min
        hi = end or dt.datetime.max
        usable = usable.between(lo, hi)

    if not len(usable):
        empty = {
            b: np.full(stack.grid.shape, np.nan, dtype=np.float32) for b in bands
        }
        return Composite(
            grid=stack.grid,
            bands=empty,
            clear_count=np.zeros(stack.grid.shape, dtype=np.int16),
            provenance=CompositeProvenance(
                start=start or dt.datetime.min,
                end=end or dt.datetime.max,
                sensors=(),
                scenes_by_sensor={},
                acquisition_days=0,
                reducer=reducer,
                harmonised_to=harmonised_to,
                notes=(*notes, "no observations carried the requested bands"),
            ),
        )

    obs = list(usable)
    out: dict[str, np.ndarray] = {}
    for band in bands:
        cube = stack_bands(obs, band)
        out[band] = _reduce(cube, reducer)

    times = [o.acquired for o in obs]
    prov = CompositeProvenance(
        start=start or min(times),
        end=end or max(times),
        sensors=usable.sensors(),
        scenes_by_sensor=dict(usable.by_sensor()),
        acquisition_days=len(usable.acquisition_days()),
        reducer=reducer,
        harmonised_to=harmonised_to,
        uncalibrated_sensors=tuple(
            s for s in usable.sensors() if s in set(uncalibrated)
        ),
        notes=tuple(notes),
    )
    return Composite(
        grid=stack.grid,
        bands=out,
        clear_count=usable.clear_count(),
        provenance=prov,
    )


def _reduce(cube: np.ndarray, reducer: Reducer) -> np.ndarray:
    """Reduce a (time, y, x) cube down the time axis, ignoring NaN.

    The all-NaN column is the case that matters and the one numpy handles
    badly: ``np.nanmedian`` of an all-NaN slice emits a RuntimeWarning and
    returns NaN. NaN is the right answer, the warning is noise, and a pipeline
    that prints thousands of warnings gets its warnings ignored, including the
    ones that mean something. So it is suppressed here, at the one place where
    all-NaN is known to be legitimate, rather than globally.
    """
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if reducer == "median":
                val = np.nanmedian(cube, axis=0)
            elif reducer == "mean":
                val = np.nanmean(cube, axis=0)
            elif reducer == "max":
                val = np.nanmax(cube, axis=0)
            elif reducer == "min":
                val = np.nanmin(cube, axis=0)
            elif reducer == "first":
                val = _first_finite(cube)
            elif reducer == "last":
                val = _first_finite(cube[::-1])
            else:
                raise ValueError(f"unknown reducer {reducer!r}")
    return val.astype(np.float32)


def _first_finite(cube: np.ndarray) -> np.ndarray:
    """First finite value down axis 0, NaN where the whole column is NaN."""
    finite = np.isfinite(cube)
    any_finite = finite.any(axis=0)
    idx = np.argmax(finite, axis=0)
    out = np.take_along_axis(cube, idx[None, ...], axis=0)[0]
    return np.where(any_finite, out, np.nan)


def monthly_composites(
    stack: ObservationStack,
    bands: Sequence[str],
    season_start: dt.datetime,
    season_end: dt.datetime,
    **kwargs: object,
) -> dict[str, Composite]:
    """One composite per calendar month across a season.

    Monthly is the natural unit for Zimbabwean maize: planting responds to the
    onset of rains within a few weeks, tasselling is a February event, and
    harvest is April to May. A finer window has too few clear looks to median
    in the wet months, which is precisely what findings/01 measured.
    """
    out: dict[str, Composite] = {}
    cursor = dt.datetime(season_start.year, season_start.month, 1)
    while cursor < season_end:
        nxt = (
            dt.datetime(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else dt.datetime(cursor.year, cursor.month + 1, 1)
        )
        out[f"{cursor:%Y-%m}"] = composite(
            stack, bands, start=cursor, end=nxt, **kwargs  # type: ignore[arg-type]
        )
        cursor = nxt
    return out


def coverage_table(composites: Mapping[str, Composite]) -> str:
    """The findings/01 table, regenerated from real composites."""
    lines = [
        "| month   | scenes | mean looks | never seen | 3 or more | sensors |",
        "|---------|-------:|-----------:|-----------:|----------:|---------|",
    ]
    for month, c in sorted(composites.items()):
        n = sum(c.provenance.scenes_by_sensor.values())
        lines.append(
            f"| {month} | {n} | {c.mean_looks:.2f} | "
            f"{c.blind_fraction * 100:.1f}% | {c.coverage(3) * 100:.1f}% | "
            f"{'+'.join(c.provenance.sensors) or 'none'} |"
        )
    return "\n".join(lines)
