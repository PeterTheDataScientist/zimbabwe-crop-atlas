"""The season pipeline: ask every sensor, take what answered, say what you got.

This module is deliberately small. Almost everything it does is delegation, and
the one piece of real logic it owns is the part findings/02 forced into
existence: discovering which sensors actually hold data for the season being
asked about, rather than assuming a fixed set and silently producing a number
from fewer instruments than the caller believes.

The shape of a run:

    inventory   ask every sensor what it holds. Cheap: metadata only, no pixels.
    load        read the scenes onto one grid. Expensive, and the only
                expensive step.
    calibrate   fit the cross-sensor adjustment from coincident pairs.
    composite   reduce to monthly images, carrying the clear count.
    phenology   read the season off the smoothed index series.
    roll up     aggregate to zones, withholding where coverage is too thin.

Every stage returns something with its evidence attached, so a caller can stop
after ``inventory`` and have a publishable coverage audit, which is exactly
what findings/01 and 02 are.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cropatlas.composite import Composite, monthly_composites
from cropatlas.harmonise import (
    Harmoniser,
    fit_harmoniser,
    harmonise_stack,
    merge_calibration,
)
from cropatlas.indices import BY_NAME
from cropatlas.observation import RADAR_BANDS, Grid, Observation, ObservationStack
from cropatlas.phenology import Phenology, extract_phenology, to_daily_grid
from cropatlas.sensors import SceneRef, SensorUnavailable


@dataclass(frozen=True)
class SensorInventory:
    """What one sensor holds for one season. Metadata only."""

    sensor: str
    scenes: tuple[SceneRef, ...]
    reachable: bool
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.scenes)

    @property
    def days(self) -> tuple[dt.date, ...]:
        return tuple(sorted({r.acquired.date() for r in self.scenes}))

    @property
    def status(self) -> str:
        """The distinction findings/02 was written about.

        'Held nothing' and 'could not be reached' are different statements
        about the world, and reporting them the same way is how a network
        timeout becomes a published claim about satellite coverage.
        """
        if not self.reachable:
            return f"unreachable ({self.error})"
        if not self.scenes:
            return "held nothing for this season"
        return f"{self.count} scenes on {len(self.days)} days"


@dataclass(frozen=True)
class SeasonInventory:
    """The full coverage picture before a single pixel is read."""

    bbox: tuple[float, float, float, float]
    start: dt.datetime
    end: dt.datetime
    by_sensor: Mapping[str, SensorInventory]

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(
            k for k, v in self.by_sensor.items() if v.reachable and v.scenes
        )

    @property
    def empty(self) -> tuple[str, ...]:
        return tuple(
            k for k, v in self.by_sensor.items() if v.reachable and not v.scenes
        )

    @property
    def unreachable(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.by_sensor.items() if not v.reachable)

    def all_scenes(self) -> list[SceneRef]:
        out: list[SceneRef] = []
        for inv in self.by_sensor.values():
            out.extend(inv.scenes)
        return sorted(out, key=lambda r: r.acquired)

    def report(self) -> str:
        lines = [
            f"Season {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}, box {self.bbox}",
        ]
        for key, inv in sorted(self.by_sensor.items()):
            lines.append(f"  {key:4s} {inv.status}")
        if self.unreachable:
            lines.append(
                "  WARNING: "
                + ", ".join(self.unreachable)
                + " could not be reached; absence here is not evidence of absence"
            )
        return "\n".join(lines)

    def table(self) -> str:
        lines = [
            "| sensor | scenes | acquisition days | status |",
            "|--------|-------:|-----------------:|--------|",
        ]
        for key, inv in sorted(self.by_sensor.items()):
            state = (
                "unreachable"
                if not inv.reachable
                else ("held nothing" if not inv.scenes else "available")
            )
            lines.append(
                f"| {key} | {inv.count} | {len(inv.days)} | {state} |"
            )
        return "\n".join(lines)


def inventory(
    sensors: Sequence[Any],
    bbox: tuple[float, float, float, float],
    start: dt.datetime,
    end: dt.datetime,
) -> SeasonInventory:
    """Ask every sensor what it holds. No pixels are read.

    This is the whole of findings/01 and findings/02 as one function call, and
    it runs in seconds. Doing the coverage audit before writing a pipeline is
    what saved this project from building a radar fusion product on an archive
    that had no radar in it.
    """
    out: dict[str, SensorInventory] = {}
    for sensor in sensors:
        try:
            refs = sensor.search(bbox, start, end)
            out[sensor.key] = SensorInventory(
                sensor=sensor.key, scenes=tuple(refs), reachable=True
            )
        except SensorUnavailable as exc:
            out[sensor.key] = SensorInventory(
                sensor=sensor.key, scenes=(), reachable=False, error=str(exc)
            )
        except Exception as exc:
            out[sensor.key] = SensorInventory(
                sensor=sensor.key,
                scenes=(),
                reachable=False,
                error=f"{type(exc).__name__}: {exc}",
            )
    return SeasonInventory(bbox=bbox, start=start, end=end, by_sensor=out)


def load_stack(
    sensors: Sequence[Any],
    inv: SeasonInventory,
    grid: Grid,
    bands: Sequence[str],
    *,
    workers: int = 6,
    on_error: Callable[[SceneRef, Exception], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[ObservationStack, list[tuple[SceneRef, str]]]:
    """Read every scene in the inventory onto one grid.

    Threaded because the work is network bound: GDAL releases the GIL during
    range requests, so threads genuinely overlap here where they would not for
    computation. Six is not a tuned number; it is low enough to stay polite to
    a free public archive that this project depends on continuing to exist.

    A scene that fails to read is recorded and skipped rather than aborting the
    season. A single corrupt COG in a six month archive should cost one
    observation, not the run. The failures are returned rather than logged so
    the caller can decide whether the loss matters, and so a season that lost
    half its scenes cannot look like a season that lost none.
    """
    by_key = {s.key: s for s in sensors}
    refs = inv.all_scenes()
    failures: list[tuple[SceneRef, str]] = []
    observations: list[Observation] = []

    warnings: list[str] = []

    def one(ref: SceneRef) -> Observation | None:
        sensor = by_key[ref.sensor]
        usable = [b for b in bands if b in sensor.bands]
        if not usable:
            return None
        obs = sensor.load(ref, grid, bands=usable)
        # A sensor may offer a physical self-check on what it just produced.
        # Optional by design: the protocol stays narrow, and a sensor with
        # nothing to check simply does not have the method.
        check = getattr(sensor, "offset_sanity", None)
        if check is not None:
            warned = check(obs)
            if warned:
                warnings.append(f"{ref}: {warned}")
        return obs

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, r): r for r in refs}
        for fut in as_completed(futures):
            ref = futures[fut]
            done += 1
            if progress:
                progress(done, len(refs))
            try:
                obs = fut.result()
            except Exception as exc:
                failures.append((ref, f"{type(exc).__name__}: {exc}"))
                if on_error:
                    on_error(ref, exc)
                continue
            if obs is not None:
                observations.append(obs)

    if warnings:
        # Physical self-checks that fired. Surfaced through the failure channel
        # rather than a log, because a scene whose radiometry is suspect is a
        # data-quality fact the caller has to see, and a log line in a
        # background thread is a fact nobody sees.
        failures.extend(
            (SceneRef(sensor="check", scene_id=w.split(":", 1)[0], acquired=dt.datetime.min), w)
            for w in warnings
        )
    return ObservationStack.build(grid, observations), failures


@dataclass(frozen=True)
class SeasonResult:
    """Everything one season produced, with the evidence for all of it."""

    inventory: SeasonInventory
    grid: Grid
    stack: ObservationStack
    harmoniser: Harmoniser
    calibration: Mapping[str, Any]
    monthly: Mapping[str, Composite]
    index_name: str
    # Radar is composited separately and always. Gamma-0 decibels and surface
    # reflectance cannot share a composite: they are different physical
    # quantities on different scales, and a median over the two together is
    # arithmetic on a category error.
    monthly_radar: Mapping[str, Composite] = field(default_factory=dict)
    failures: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def index_series(self) -> tuple[list[dt.date], list[np.ndarray], list[np.ndarray]]:
        """Monthly index arrays with their clear counts, in date order."""
        idx = BY_NAME[self.index_name]
        dates, arrays, counts = [], [], []
        for month, comp in sorted(self.monthly.items()):
            if not idx.computable_from(comp.bands):
                continue
            y, m = int(month[:4]), int(month[5:7])
            dates.append(dt.date(y, m, 15))
            arrays.append(idx(comp.bands))
            counts.append(comp.clear_count)
        return dates, arrays, counts

    def mean_index_series(self) -> tuple[list[dt.date], list[float], list[float]]:
        """District mean of the index per month, with mean clear looks."""
        dates, arrays, counts = self.index_series()
        vals, looks = [], []
        for a, c in zip(arrays, counts, strict=True):
            with np.errstate(invalid="ignore"):
                vals.append(float(np.nanmean(a)) if np.isfinite(a).any() else float("nan"))
            looks.append(float(c.mean()))
        return dates, vals, looks

    def observation_series(
        self, min_coverage: float = 0.10
    ) -> tuple[list[dt.datetime], list[float], list[float], list[str]]:
        """District mean index per acquisition, not per month.

        Monthly composites give six points across a season, which is too few to
        locate a start of season to better than a fortnight and too few for the
        smoother to have anything to be stiff about. The stack itself holds
        eighty to a hundred observations across the same period, and those are
        the actual measurements: compositing to months and then fitting a curve
        throws away most of the temporal information that the three-sensor
        harmonisation was built to gather in the first place.

        Weight is the count of valid pixels, so a scene that saw a tenth of the
        district counts for a tenth of one that saw all of it. Scenes below
        ``min_coverage`` are dropped entirely rather than down-weighted,
        because a district mean from 3% of the pixels is not a small
        measurement of the district, it is a measurement of somewhere else.
        """
        idx = BY_NAME[self.index_name]
        times: list[dt.datetime] = []
        values: list[float] = []
        weights: list[float] = []
        # The sensor label travels with the point. Without it a figure has to
        # guess from the acquisition hour, and the guess is wrong often enough
        # to make the one diagnostic that matters unreadable: if the Landsat
        # points sit visibly above the Sentinel-2 points, the calibration
        # failed, and a mislabelled scatter hides exactly that.
        sensors: list[str] = []
        n_pixels = self.grid.shape[0] * self.grid.shape[1]

        for obs in self.stack:
            if not idx.computable_from(obs):
                continue
            coverage = obs.valid_fraction
            if coverage < min_coverage:
                continue
            arr = idx(obs)
            finite = np.isfinite(arr)
            if not finite.any():
                continue
            with np.errstate(invalid="ignore"):
                mean = float(np.nanmean(arr))
            if not np.isfinite(mean):
                continue
            times.append(obs.acquired)
            values.append(mean)
            weights.append(float(finite.sum()) / n_pixels)
            sensors.append(obs.sensor)
        return times, values, weights, sensors

    def district_phenology(
        self,
        season_start: dt.date,
        season_end: dt.date,
        *,
        per_observation: bool = True,
        **kwargs: Any,
    ) -> Phenology:
        """Season metrics for the district as a whole.

        Defaults to the per-acquisition series for the reason above. Passing
        ``per_observation=False`` runs it on monthly composites instead, which
        is worth having because the difference between the two answers is
        itself a measurement of how much the monthly step costs.
        """
        if per_observation:
            times, vals, weights, _ = self.observation_series()
            if len(times) >= 4:
                grid = to_daily_grid(
                    times, vals, season_start, season_end,
                    weights=[w * 6.0 for w in weights],
                )
                return extract_phenology(grid, **kwargs)

        dates, vals, looks = self.mean_index_series()
        times = [dt.datetime(d.year, d.month, d.day) for d in dates]
        grid = to_daily_grid(
            times,
            vals,
            season_start,
            season_end,
            weights=[min(x, 6.0) for x in looks],
        )
        return extract_phenology(grid, **kwargs)

    def report(self) -> str:
        lines = [self.inventory.report(), "", self.stack.summary(), ""]
        lines.append(self.harmoniser.report())
        lines.append("")
        for _month, comp in sorted(self.monthly.items()):
            lines.append(comp.report())
        if self.failures:
            lines.append("")
            lines.append(f"{len(self.failures)} scene(s) failed to read:")
            for scene, why in self.failures[:5]:
                lines.append(f"  {scene}: {why}")
        return "\n".join(lines)


def run_season(
    sensors: Sequence[Any],
    bbox: tuple[float, float, float, float],
    season_start: dt.date,
    season_end: dt.date,
    grid: Grid,
    *,
    bands: Sequence[str] = ("red", "nir", "blue", "swir1"),
    index: str = "ndvi",
    calibrate: bool = True,
    fallback_calibration: Harmoniser | None = None,
    fallback_label: str = "another season",
    workers: int = 6,
    progress: Callable[[str], None] | None = None,
) -> SeasonResult:
    """One season, end to end.

    ``calibrate`` fits the cross-sensor adjustment from coincident acquisitions
    inside this season rather than importing coefficients from a paper. If too
    few coincident pairs exist the fit returns nothing and the run proceeds
    uncalibrated, with the fact recorded in every composite's caveats. That is
    the honest degradation: an uncalibrated composite that says so is useful,
    and a composite silently corrected by numbers fitted over Maryland is not.
    """
    say = progress or (lambda _m: None)
    start = dt.datetime.combine(season_start, dt.time.min)
    end = dt.datetime.combine(season_end, dt.time.min)

    say("inventory")
    inv = inventory(sensors, bbox, start, end)
    say(inv.table())

    say("loading")
    stack, failures = load_stack(
        sensors, inv, grid, bands, workers=workers,
        progress=lambda d, n: say(f"  {d}/{n}") if d % 10 == 0 else None,
    )
    say(stack.summary())

    harmoniser = Harmoniser.identity("s2")
    calibration: dict[str, Any] = {"attempted": False}
    if calibrate and len(stack.optical()) > 1:
        say("calibrating")
        harmoniser, calibration = fit_harmoniser(
            stack.optical(), "s2",
            bands=[b for b in bands if b not in RADAR_BANDS],
        )
        calibration["attempted"] = True
        # A season that could not fit its own coefficients is not a season
        # that has to run uncorrected. The difference between OLI and MSI is a
        # property of the instruments, not of the weather, so coefficients
        # fitted on a season with better coincidence remain valid here. What
        # matters is that the borrowing is recorded rather than hidden.
        if fallback_calibration is not None:
            harmoniser = merge_calibration(
                harmoniser, fallback_calibration, fallback_label
            )
            calibration["fallback_used"] = harmoniser.fitted_on is not None
            calibration["fallback_label"] = fallback_label
        stack = harmonise_stack(stack, harmoniser)

    uncalibrated = tuple(
        s for s in stack.sensors() if s != "s2" and not harmoniser.is_calibrated(s)
    )

    say("compositing")
    # Split the requested bands by physical family before compositing.
    #
    # This is not tidiness. ``composite`` reduces the observations that carry
    # EVERY requested band, which is the right contract: a composite has to be
    # built from scenes that all measured the same things. Handing it optical
    # and radar bands together therefore selects the observations carrying
    # both, and no instrument carries both, so the result is silently empty.
    #
    # That happened on the first run of the 2025/26 season: 255 observations
    # loaded, 68 of them radar, and every monthly composite came back 100%
    # blind. The load was fine, the calibration was fine, and the output was
    # nothing. A pipeline fails this way when a correct component is given
    # inputs its contract excludes, which is why the split belongs here, at the
    # point that knows both families exist, rather than inside the compositor.
    optical_bands = [b for b in bands if b not in RADAR_BANDS]
    radar_bands = [b for b in bands if b in RADAR_BANDS]

    notes: tuple[str, ...] = ()
    if harmoniser.fitted_on:
        notes = (
            "part of the cross-sensor calibration was fitted on "
            f"{harmoniser.fitted_on}, not on this season",
        )
    monthly = monthly_composites(
        stack.optical(),
        optical_bands,
        start,
        end,
        harmonised_to="s2" if harmoniser.adjustments else None,
        uncalibrated=uncalibrated,
        notes=notes,
    )
    monthly_radar: dict[str, Composite] = {}
    if radar_bands and len(stack.radar()):
        monthly_radar = dict(
            monthly_composites(
                stack.radar(), radar_bands, start, end,
                notes=("gamma-0 decibels, not reflectance",),
            )
        )

    return SeasonResult(
        inventory=inv,
        grid=grid,
        stack=stack,
        harmoniser=harmoniser,
        calibration=calibration,
        monthly=monthly,
        monthly_radar=monthly_radar,
        index_name=index,
        failures=tuple((str(r), why) for r, why in failures),
    )
