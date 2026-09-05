"""Digital Earth Africa adapters: Sentinel-2, Landsat 8 and 9, Sentinel-1 RTC.

Each of these is a thin translation layer. Everything genuinely difficult lives
elsewhere: masking in ``masks``, reprojection in ``io.cog``, compositing in
``composite``. What is left here is per-instrument knowledge that has nowhere
else to go, and there is more of it than anyone expects.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cropatlas import masks
from cropatlas.io import cog, stac
from cropatlas.observation import Grid, Observation, Provenance
from cropatlas.sensors import SceneRef, SensorUnavailable


@dataclass(frozen=True)
class BandMap:
    """Canonical band name to archive asset name."""

    mapping: Mapping[str, str]

    def assets_for(self, bands: Sequence[str]) -> dict[str, str]:
        out = {}
        for b in bands:
            if b in self.mapping:
                out[b] = self.mapping[b]
        return out

    @property
    def canonical(self) -> tuple[str, ...]:
        return tuple(self.mapping)


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------

S2_BANDS = BandMap(
    {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "rededge1": "B05",
        "rededge2": "B06",
        "rededge3": "B07",
        "nir": "B08",
        "swir1": "B11",
        "swir2": "B12",
    }
)


@dataclass
class Sentinel2:
    """DE Africa ``s2_l2a``. The reference sensor.

    Reference because it has the finest resolution of the three and because it
    is the only one carrying red edge. Harmonising Sentinel-2 down onto the
    Landsat band set would mean the reference basis had bands the reference
    could not measure, which is the wrong way round.

    The processing baseline trap. From baseline 04.00, dated 25 January 2022,
    ESA added a -1000 radiometric offset to L2A products. Reflectance is
    ``(DN + BOA_ADD_OFFSET) / 10000``, not ``DN / 10000``. Getting this wrong
    shifts every value by 0.1 reflectance, which is enormous: larger than the
    entire cross-sensor difference the harmonisation exists to correct, and
    large enough to move NDVI by 0.15. It does not look like a bug, it looks
    like a slightly odd crop.

    Because different archives handle the offset differently and the property
    is not always published, this adapter reads the baseline where it exists,
    applies the documented rule, and records what it did in provenance. The
    empirical harmonisation fit is then the independent check: an unhandled
    0.1 offset would show up as an intercept two orders of magnitude larger
    than any real cross-sensor difference, which is a very loud failure.
    """

    key: str = "s2"
    collection: str = "s2_l2a"
    band_map: BandMap = field(default_factory=lambda: S2_BANDS)
    scl_usable: frozenset[int] = masks.SCL_USABLE_DEFAULT
    # A ground distance, not a pixel count. Specifying it in pixels made the
    # cloud mask depend on the grid resolution, so the same season composited
    # at 20 m and at 200 m produced different coverage statistics from
    # identical inputs. 40 m is two Sentinel-2 pixels at native resolution and
    # is about where the SCL stops under-calling cloud edges.
    cloud_buffer_m: float = 40.0
    assume_offset_applied: bool | None = None

    @property
    def bands(self) -> tuple[str, ...]:
        return self.band_map.canonical

    def search(
        self,
        bbox: tuple[float, float, float, float],
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[SceneRef]:
        try:
            items = stac.search(self.collection, bbox, start, end)
        except Exception as exc:
            raise SensorUnavailable(f"{self.key} search failed: {exc}") from exc
        return [
            SceneRef(
                sensor=self.key,
                scene_id=item["id"],
                acquired=stac.item_datetime(item),
                handle={"item": item},
                cloud_cover=item["properties"].get("eo:cloud_cover"),
            )
            for item in items
        ]

    def load(
        self, ref: SceneRef, grid: Grid, bands: Sequence[str] | None = None
    ) -> Observation:
        item: dict[str, Any] = ref.handle["item"]  # type: ignore[assignment]
        wanted = list(bands) if bands else list(self.bands)

        resolution = abs(grid.transform[0])
        buffer_px = masks.dilation_pixels(self.cloud_buffer_m, resolution)
        scl = cog.read_categorical(stac.asset_href(item, "SCL"), grid)
        clear = masks.sentinel2_clear(
            scl, usable=self.scl_usable, dilate_cloud=buffer_px
        )

        offset, baseline = self._offset_for(item)
        out: dict[str, np.ndarray] = {}
        valid = clear.copy()
        for name, asset in self.band_map.assets_for(wanted).items():
            raw = cog.read_onto_grid(stac.asset_href(item, asset), grid)
            refl = ((raw + offset) / 10_000.0).astype(np.float32)
            ok = masks.valid_range(refl)
            valid &= ok
            out[name] = refl

        for name in out:
            out[name] = np.where(valid, out[name], np.float32(np.nan)).astype(np.float32)

        return Observation(
            grid=grid,
            bands=out,
            valid=valid,
            provenance=Provenance(
                sensor=self.key,
                scene_id=ref.scene_id,
                acquired=ref.acquired,
                detail={
                    "platform": str(item["properties"].get("platform", "")),
                    "processing_baseline": baseline,
                    "boa_add_offset": str(offset),
                    "scene_cloud_cover": str(ref.cloud_cover),
                    "cloud_buffer_m": str(self.cloud_buffer_m),
                    "cloud_buffer_px": str(buffer_px),
                },
            ),
        )

    def _offset_for(self, item: dict[str, Any]) -> tuple[float, str]:
        """The BOA_ADD_OFFSET still to apply, and where that answer came from.

        The question is not 'is this a post-baseline-04.00 product', which is
        what an earlier version of this method asked. It is 'has the offset
        already been applied by whoever built this archive', which is a
        different question with a different answer.

        DE Africa's ``s2_l2a`` is built on the Element84 cloud-optimised
        rebuild, which normalises every product onto the pre-baseline scale and
        publishes ``sentinel:boa_offset_applied: true`` to say so. Reading the
        baseline instead and applying -1000 subtracts the offset a second time,
        which puts red reflectance at -0.008 where Landsat reads +0.079 over
        the same ground on the same day.

        That is exactly the failure this class's docstring predicted, and it
        was caught the way the docstring said it would be: the cross-sensor
        comparison showed a 0.087 discrepancy, two orders of magnitude larger
        than any real difference between OLI and MSI. The lesson is narrow and
        useful. Ask the archive what it did rather than inferring it from the
        instrument's processing history, and when an archive publishes a flag
        that answers your question directly, look for that flag first.

        Order of preference: an explicit caller override, the archive's own
        flag, the processing baseline, then the acquisition date. Each step
        down is less certain and says so in the returned reason, which lands in
        provenance and therefore in the output.
        """
        if self.assume_offset_applied is True:
            return 0.0, "caller: already applied"
        if self.assume_offset_applied is False:
            return -1000.0, "caller: not applied"

        props = item.get("properties", {})

        flag = props.get("sentinel:boa_offset_applied")
        if flag is None:
            flag = props.get("s2:boa_offset_applied")
        if flag is not None:
            if bool(flag):
                return 0.0, "archive flag: offset already applied"
            return -1000.0, "archive flag: offset not applied"

        baseline = (
            props.get("s2:processing_baseline")
            or props.get("sentinel:processing_baseline")
            or ""
        )
        if baseline:
            try:
                if float(baseline) >= 4.0:
                    return -1000.0, f"baseline {baseline}, no archive flag"
                return 0.0, f"baseline {baseline}, no archive flag"
            except ValueError:
                pass

        cutover = dt.datetime(2022, 1, 25)
        if stac.item_datetime(item) >= cutover:
            return -1000.0, "inferred from date, no flag or baseline published"
        return 0.0, "inferred from date, no flag or baseline published"

    # Physical floors for a median over clear land pixels. Both are set well
    # below any real scene and well above what a wrongly applied 0.1 offset
    # leaves behind, so the check fires on the bug and not on unusual weather.
    #
    # Red is the diagnostic band, and that is worth saying because the obvious
    # choice is NIR. NIR over a canopy is 0.25 to 0.40, so subtracting 0.1
    # leaves 0.15 to 0.30, which is low but entirely plausible: a check on NIR
    # alone does not fire on the real bug. Red over a canopy is 0.02 to 0.10,
    # so the same 0.1 subtraction drives the median negative, which nothing
    # physical does. The band with the smallest true value is the one where a
    # fixed additive error is most visible. A first version of this check used
    # NIR and a test caught that it would have missed the very bug it was
    # written for.
    MIN_PLAUSIBLE_MEDIAN_RED = 0.010
    MIN_PLAUSIBLE_MEDIAN_NIR = 0.100

    def offset_sanity(self, obs: Observation) -> str | None:
        """A physical check that the offset decision was right.

        The metadata route above is correct for this archive today. It is not
        correct for every archive forever, and getting it wrong is silent: the
        imagery still renders, the crop just looks slightly odd.

        This is the independent test that needs no metadata at all. Returns a
        warning string, or None when the scene looks physically sensible.
        """
        if not obs.valid.any():
            return None

        for band, floor in (
            ("red", self.MIN_PLAUSIBLE_MEDIAN_RED),
            ("nir", self.MIN_PLAUSIBLE_MEDIAN_NIR),
        ):
            if not obs.has(band):
                continue
            vals = obs.band(band)[obs.valid]
            vals = vals[np.isfinite(vals)]
            if vals.size < 100:
                continue
            median = float(np.median(vals))
            if median < floor:
                return (
                    f"median {band} is {median:+.4f} over clear land, below the "
                    f"physical floor of {floor:.3f}; the BOA offset has probably "
                    "been applied twice"
                )
        return None


# ---------------------------------------------------------------------------
# Landsat 8 and 9
# ---------------------------------------------------------------------------

LANDSAT_BANDS = BandMap(
    {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7",
    }
)

# Collection 2 Level 2 surface reflectance is stored as scaled integers.
# reflectance = DN * 0.0000275 - 0.2. Published by USGS and fixed for the
# collection; named here rather than inlined so it is greppable when
# Collection 3 changes it.
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2


@dataclass
class Landsat:
    """DE Africa ``ls8_sr`` and ``ls9_sr``. Same instrument family, same code.

    Landsat 8 and 9 fly the same OLI design in the same orbit eight days apart,
    which is exactly why they matter here: they fail on different days from
    each other and from Sentinel-2. Findings/01 measured the result, February
    clear looks going from 1.76 to 4.02 and the blind fraction from 16.7% to
    zero.

    They are separate ``Sensor`` instances rather than one combined adapter,
    because the harmonisation fits them separately. Treating them as one sensor
    would assume their radiometry is identical, and whether it is is a question
    this pipeline can answer rather than assume.
    """

    key: str
    collection: str
    band_map: BandMap = field(default_factory=lambda: LANDSAT_BANDS)
    exclude_shadow: bool = True
    exclude_cirrus: bool = True

    @property
    def bands(self) -> tuple[str, ...]:
        return self.band_map.canonical

    def search(
        self,
        bbox: tuple[float, float, float, float],
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[SceneRef]:
        try:
            items = stac.search(self.collection, bbox, start, end)
        except Exception as exc:
            raise SensorUnavailable(f"{self.key} search failed: {exc}") from exc
        return [
            SceneRef(
                sensor=self.key,
                scene_id=item["id"],
                acquired=stac.item_datetime(item),
                handle={"item": item},
                cloud_cover=item["properties"].get("eo:cloud_cover"),
            )
            for item in items
        ]

    def load(
        self, ref: SceneRef, grid: Grid, bands: Sequence[str] | None = None
    ) -> Observation:
        item: dict[str, Any] = ref.handle["item"]  # type: ignore[assignment]
        wanted = list(bands) if bands else list(self.bands)

        qa = cog.read_categorical(stac.asset_href(item, "QA_PIXEL"), grid)
        clear = masks.landsat_clear(
            qa.astype(np.uint16),
            exclude_shadow=self.exclude_shadow,
            exclude_cirrus=self.exclude_cirrus,
        )

        out: dict[str, np.ndarray] = {}
        valid = clear.copy()
        for name, asset in self.band_map.assets_for(wanted).items():
            raw = cog.read_onto_grid(stac.asset_href(item, asset), grid)
            # DN 0 is fill in Collection 2 and would scale to -0.2, a
            # physically impossible reflectance that valid_range then removes.
            # Removing it here as well makes the intent explicit rather than
            # relying on a range check to catch a nodata value by accident.
            refl = (raw * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET).astype(np.float32)
            refl = np.where(raw == 0, np.float32(np.nan), refl).astype(np.float32)
            valid &= masks.valid_range(refl)
            out[name] = refl

        for name in out:
            out[name] = np.where(valid, out[name], np.float32(np.nan)).astype(np.float32)

        return Observation(
            grid=grid,
            bands=out,
            valid=valid,
            provenance=Provenance(
                sensor=self.key,
                scene_id=ref.scene_id,
                acquired=ref.acquired,
                detail={
                    "platform": str(item["properties"].get("platform", "")),
                    "collection": str(
                        item["properties"].get("landsat:collection_number", "")
                    ),
                    "scene_cloud_cover": str(ref.cloud_cover),
                    "shadow_excluded": str(self.exclude_shadow),
                },
            ),
        )


def landsat8() -> Landsat:
    return Landsat(key="l8", collection="ls8_sr")


def landsat9() -> Landsat:
    return Landsat(key="l9", collection="ls9_sr")


# ---------------------------------------------------------------------------
# Sentinel-1 RTC
# ---------------------------------------------------------------------------


@dataclass
class Sentinel1RTC:
    """DE Africa ``s1_rtc``. Radiometrically terrain corrected gamma-0.

    Nothing over the Zimbabwean maize belt from 2022/23 through 2024/25, and a
    full season from November 2025 onward: findings/02, confirmed against CDSE
    and ASF. This adapter exists so that the 2025/26 season and everything
    after it can use radar, and so the seasons that have none report that as a
    fact rather than as a failure.

    Radar values are gamma-0 converted to decibels here. Compositing in linear
    power and converting afterwards is the physically correct order, because
    decibels are logarithmic and the mean of logs is not the log of the mean.
    That conversion happens in the compositor for radar bands; this adapter
    hands over decibels because that is the unit everything downstream reports
    in, and the linear-domain reduction is a compositing concern rather than a
    reading one.
    """

    key: str = "s1"
    collection: str = "s1_rtc"
    min_db: float = -35.0
    max_db: float = 5.0

    @property
    def bands(self) -> tuple[str, ...]:
        return ("vv", "vh")

    def search(
        self,
        bbox: tuple[float, float, float, float],
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[SceneRef]:
        try:
            items = stac.search(self.collection, bbox, start, end)
        except Exception as exc:
            raise SensorUnavailable(f"{self.key} search failed: {exc}") from exc
        return [
            SceneRef(
                sensor=self.key,
                scene_id=item["id"],
                acquired=stac.item_datetime(item),
                handle={"item": item},
                cloud_cover=None,  # radar does not care, and saying so matters
            )
            for item in items
        ]

    def load(
        self, ref: SceneRef, grid: Grid, bands: Sequence[str] | None = None
    ) -> Observation:
        item: dict[str, Any] = ref.handle["item"]  # type: ignore[assignment]
        wanted = [b for b in (bands or self.bands) if b in self.bands]

        out: dict[str, np.ndarray] = {}
        valid = np.ones(grid.shape, dtype=bool)
        for name in wanted:
            linear = cog.read_onto_grid(stac.asset_href(item, name), grid)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = (10.0 * np.log10(np.where(linear > 0, linear, np.nan))).astype(
                    np.float32
                )
            ok = np.isfinite(db) & (db >= self.min_db) & (db <= self.max_db)
            valid &= ok
            out[name] = np.where(ok, db, np.float32(np.nan)).astype(np.float32)

        for name in out:
            out[name] = np.where(valid, out[name], np.float32(np.nan)).astype(np.float32)

        return Observation(
            grid=grid,
            bands=out,
            valid=valid,
            provenance=Provenance(
                sensor=self.key,
                scene_id=ref.scene_id,
                acquired=ref.acquired,
                detail={
                    "platform": str(item["properties"].get("platform", "")),
                    "units": "gamma0 dB",
                },
            ),
        )


def default_sensors() -> list[Any]:
    """The four adapters, in the order the pipeline should try them.

    Deliberately a function rather than a module constant. A constant would
    invite a caller to mutate the shared list, and the whole argument of
    findings/02 is that the sensor set is a per-run decision.
    """
    return [Sentinel2(), landsat8(), landsat9(), Sentinel1RTC()]
