"""The unit every sensor produces and every downstream stage consumes.

Design note, because this is the load bearing decision in the whole package.

The measurement in findings/02 established that the sensor set available over
Zimbabwe changes from season to season: three seasons with no radar at all,
then a season with a full radar record. Any pipeline that hardcodes its sensor
list is wrong for some part of the archive, and the failure is silent, which is
the worst kind.

So the boundary is drawn here. A sensor's job is to turn whatever it holds into
an ``Observation``: canonical band names, surface reflectance on one scale, an
explicit validity mask, and a provenance record. Everything after this line
works on ``Observation`` and cannot tell Sentinel-2 from Landsat 9 unless it
deliberately asks. That is parse-don't-validate applied at a sensor boundary:
the messy, per-instrument, bit-twiddling knowledge lives in one place per
sensor and is not allowed to leak.

The band names are the vocabulary. They are physical, not instrumental: ``red``
means the red band, whichever instrument measured it, resampled and bandpass
adjusted onto a common basis. An instrument that has no equivalent for a name
omits it rather than substituting something close, and downstream code that
needs it asks and handles the absence.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

# The canonical vocabulary. Physical bands, not instrument band numbers.
#
# Sentinel-2 carries three red-edge bands that Landsat has no equivalent for.
# They are in the vocabulary because they are the strongest bands for crop
# nitrogen and canopy structure, and dropping them to make every sensor look
# alike would throw away the reason Sentinel-2 is worth having. Sensors that
# lack them simply do not supply them.
OPTICAL_BANDS: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "swir1",
    "swir2",
)

# Radar is not reflectance and must never be composited into the same array as
# reflectance. It is a separate vocabulary carried on the same object so that a
# season with radar and a season without differ in what is present, not in what
# type of object comes back.
RADAR_BANDS: tuple[str, ...] = ("vv", "vh")

ALL_BANDS: tuple[str, ...] = OPTICAL_BANDS + RADAR_BANDS


class BandUnavailable(KeyError):
    """A band was requested that this observation's instrument does not carry.

    Distinct from a band that exists but is masked everywhere. The first is a
    structural fact about the sensor, the second is weather. Conflating them is
    how a pipeline silently reports a February with no data as a February with
    no crop.
    """


@dataclass(frozen=True)
class Grid:
    """Where the pixels are.

    Held as an affine transform plus CRS plus shape rather than as coordinate
    arrays, because two observations are on the same grid if and only if these
    three agree, and comparing three small tuples is cheap enough to do on
    every operation that combines observations.
    """

    transform: tuple[float, float, float, float, float, float]
    crs: str
    shape: tuple[int, int]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def pixel_area_m2(self) -> float:
        """Absolute pixel area, assuming a projected CRS in metres.

        Used for area weighted district roll-ups. Guarded rather than assumed:
        a geographic CRS gives a meaningless answer here, and silently
        reporting hectares computed from degrees is exactly the sort of error
        that survives review because the number looks plausible.
        """
        if self.crs.upper() in {"EPSG:4326", "OGC:CRS84", "CRS84"}:
            raise ValueError(
                f"pixel_area_m2 needs a projected CRS in metres, got {self.crs}. "
                "Reproject to the local UTM zone before computing areas."
            )
        a, _, _, _, e, _ = self.transform
        return abs(a * e)

    def same_as(self, other: Grid) -> bool:
        return (
            self.shape == other.shape
            and self.crs == other.crs
            and all(
                abs(x - y) < 1e-6 for x, y in zip(self.transform, other.transform, strict=True)
            )
        )


@dataclass(frozen=True)
class Provenance:
    """Where a number came from.

    Findings/02 forced this to be part of the data rather than a logging
    concern. A district mean computed from three sensors and one computed from
    four are different measurements, and a user comparing 2024 with 2026 has to
    be told which one they are holding. Making provenance a field means the
    only way to lose it is to write code that deliberately discards it.
    """

    sensor: str
    scene_id: str
    acquired: dt.datetime
    # Free-form, sensor specific, for anything worth keeping that has no home
    # in the common model: processing baseline, orbit direction, sun angle.
    detail: Mapping[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.sensor}:{self.scene_id}@{self.acquired:%Y-%m-%d}"


@dataclass(frozen=True)
class Observation:
    """One scene, parsed onto the canonical vocabulary.

    ``bands`` holds float32 arrays. Optical bands are surface reflectance in
    [0, 1]; radar bands are gamma-0 in decibels. ``valid`` is True where the
    pixel is usable, having already had cloud, shadow, saturation and no-data
    removed by the sensor that produced it.

    Invalid pixels are NaN in the band arrays as well as False in ``valid``.
    Carrying the information twice is deliberate: NaN makes an arithmetic slip
    propagate loudly instead of quietly averaging in a fill value, and the
    boolean mask makes counting cheap without a NaN scan.
    """

    grid: Grid
    bands: Mapping[str, np.ndarray]
    valid: np.ndarray
    provenance: Provenance

    def __post_init__(self) -> None:
        h, w = self.grid.shape
        if self.valid.shape != (h, w):
            raise ValueError(
                f"valid mask {self.valid.shape} does not match grid {(h, w)}"
            )
        if self.valid.dtype != np.bool_:
            raise TypeError(f"valid must be boolean, got {self.valid.dtype}")
        for name, arr in self.bands.items():
            if name not in ALL_BANDS:
                raise ValueError(
                    f"{name!r} is not a canonical band name. "
                    f"Known: {', '.join(ALL_BANDS)}"
                )
            if arr.shape != (h, w):
                raise ValueError(
                    f"band {name!r} has shape {arr.shape}, grid is {(h, w)}"
                )
            if arr.dtype != np.float32:
                raise TypeError(
                    f"band {name!r} must be float32, got {arr.dtype}. "
                    "Sensors are responsible for scaling and casting."
                )

    @property
    def acquired(self) -> dt.datetime:
        return self.provenance.acquired

    @property
    def sensor(self) -> str:
        return self.provenance.sensor

    @property
    def is_radar(self) -> bool:
        return any(b in RADAR_BANDS for b in self.bands)

    def band(self, name: str) -> np.ndarray:
        try:
            return self.bands[name]
        except KeyError as exc:
            raise BandUnavailable(
                f"{self.sensor} has no {name!r} band. "
                f"It carries: {', '.join(sorted(self.bands))}"
            ) from exc

    def has(self, *names: str) -> bool:
        return all(n in self.bands for n in names)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def masked_to(self, mask: np.ndarray) -> Observation:
        """Narrow validity, for example to a crop mask or a district boundary.

        Returns a new observation. Nothing in this package mutates an
        observation in place: a scene is evidence, and evidence that changes
        under you is not evidence.
        """
        if mask.shape != self.valid.shape:
            raise ValueError(f"mask {mask.shape} does not match {self.valid.shape}")
        keep = self.valid & mask.astype(bool)
        bands = {
            n: np.where(keep, a, np.float32(np.nan)).astype(np.float32)
            for n, a in self.bands.items()
        }
        return replace(self, bands=bands, valid=keep)


@dataclass(frozen=True)
class ObservationStack:
    """Every observation of one place over one period, from every sensor.

    Sorted by acquisition time, because phenology only means anything in order
    and re-sorting at every stage is how an off-by-one crept into three
    different places in an earlier draft.

    The stack does not require every observation to carry the same bands. A
    2025/26 stack holds optical scenes with nine bands and radar scenes with
    two, and the compositor selects what it needs. This is the direct
    consequence of the finding that the sensor set varies by season.
    """

    grid: Grid
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        for obs in self.observations:
            if not obs.grid.same_as(self.grid):
                raise ValueError(
                    f"{obs.provenance} is on a different grid from the stack. "
                    "Reproject and resample at the sensor boundary, not here."
                )
        times = [o.acquired for o in self.observations]
        if times != sorted(times):
            raise ValueError("observations must be in acquisition order")

    @classmethod
    def build(cls, grid: Grid, observations: Iterable[Observation]) -> ObservationStack:
        return cls(grid, tuple(sorted(observations, key=lambda o: o.acquired)))

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def optical(self) -> ObservationStack:
        return replace(
            self, observations=tuple(o for o in self.observations if not o.is_radar)
        )

    def radar(self) -> ObservationStack:
        return replace(
            self, observations=tuple(o for o in self.observations if o.is_radar)
        )

    def with_bands(self, *names: str) -> ObservationStack:
        """Only those observations carrying every named band.

        The natural way to write a stage that needs red and nir: ask for them,
        and get a stack where every member has them, instead of guarding each
        access. An empty result is a legitimate answer and downstream code has
        to handle it, which is the honest shape for an archive with holes.
        """
        return replace(
            self, observations=tuple(o for o in self.observations if o.has(*names))
        )

    def between(self, start: dt.datetime, end: dt.datetime) -> ObservationStack:
        return replace(
            self,
            observations=tuple(
                o for o in self.observations if start <= o.acquired < end
            ),
        )

    def sensors(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for o in self.observations:
            seen[o.sensor] = None
        return tuple(seen)

    def by_sensor(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for o in self.observations:
            counts[o.sensor] = counts.get(o.sensor, 0) + 1
        return counts

    def clear_count(self) -> np.ndarray:
        """Per-pixel count of valid observations across the stack.

        The single most useful diagnostic in the package, and the one finding
        01 was built on. A composite is only as trustworthy as this array, so
        it travels with every composite rather than being recomputed on demand
        by whoever remembers to.
        """
        if not self.observations:
            return np.zeros(self.grid.shape, dtype=np.int16)
        total = np.zeros(self.grid.shape, dtype=np.int16)
        for obs in self.observations:
            total += obs.valid.astype(np.int16)
        return total

    def acquisition_days(self) -> tuple[dt.date, ...]:
        return tuple(sorted({o.acquired.date() for o in self.observations}))

    def summary(self) -> str:
        if not self.observations:
            return "empty stack"
        first = self.observations[0].acquired.date()
        last = self.observations[-1].acquired.date()
        by = ", ".join(f"{k}={v}" for k, v in sorted(self.by_sensor().items()))
        return (
            f"{len(self)} observations {first} to {last} "
            f"on {len(self.acquisition_days())} days ({by})"
        )


def stack_bands(stack: Sequence[Observation], band: str) -> np.ndarray:
    """Lift one band across a stack into a (time, y, x) cube.

    NaN where invalid, which is what every reducer in this package expects.
    Kept as a function rather than a method because it materialises a cube that
    can be large, and a call site that allocates hundreds of megabytes should
    look like it does.
    """
    if not stack:
        raise ValueError("cannot stack an empty sequence")
    missing = [str(o.provenance) for o in stack if not o.has(band)]
    if missing:
        raise BandUnavailable(
            f"{len(missing)} observation(s) lack band {band!r}, first: {missing[0]}. "
            "Filter with ObservationStack.with_bands before stacking."
        )
    return np.stack([o.band(band) for o in stack]).astype(np.float32)
