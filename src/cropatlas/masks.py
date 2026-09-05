"""Cloud masking, per instrument, as pure functions over arrays.

Kept out of the sensor classes and away from the network so that the part most
likely to be wrong is the part easiest to test. Every function here takes an
array of quality flags and returns a boolean array where True means usable.

The two instruments encode quality in completely different ways. Sentinel-2
ships a per-pixel classification with one label per pixel. Landsat ships a
bitfield with sixteen independent flags. Neither is convertible into the other,
and pretending otherwise by mapping both onto a single 'cloudy' boolean loses
the distinctions that matter, so both are implemented on their own terms and
only the output is common.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Sentinel-2: the Scene Classification Layer
# ---------------------------------------------------------------------------

SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_BARE_SOIL = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW = 11

SCL_LABELS = {
    SCL_NO_DATA: "no data",
    SCL_SATURATED: "saturated or defective",
    SCL_DARK_AREA: "dark area or topographic shadow",
    SCL_CLOUD_SHADOW: "cloud shadow",
    SCL_VEGETATION: "vegetation",
    SCL_BARE_SOIL: "not vegetated",
    SCL_WATER: "water",
    SCL_UNCLASSIFIED: "unclassified",
    SCL_CLOUD_MEDIUM: "cloud, medium probability",
    SCL_CLOUD_HIGH: "cloud, high probability",
    SCL_THIN_CIRRUS: "thin cirrus",
    SCL_SNOW: "snow or ice",
}

# Classes kept as usable observations of the land surface.
#
# Vegetation and bare soil are the obvious two. Water is kept because a
# cropland pixel that floods is a real observation of a real event, and
# dropping it would quietly delete waterlogging from the record.
#
# Unclassified (7) is the judgement call. Over Zimbabwean smallholder plots the
# classifier lands on 7 fairly often at field edges and on mixed pixels, and
# excluding it discards real observations in exactly the fragmented landscape
# this project is about. It is included by default and separable, so the cost
# of that choice can be measured rather than argued about.
SCL_USABLE_DEFAULT: frozenset[int] = frozenset(
    {SCL_VEGETATION, SCL_BARE_SOIL, SCL_WATER, SCL_UNCLASSIFIED}
)

# The conservative alternative, for the sensitivity check.
SCL_USABLE_STRICT: frozenset[int] = frozenset(
    {SCL_VEGETATION, SCL_BARE_SOIL, SCL_WATER}
)


def sentinel2_clear(
    scl: np.ndarray,
    usable: frozenset[int] = SCL_USABLE_DEFAULT,
    dilate_cloud: int = 0,
) -> np.ndarray:
    """True where the Sentinel-2 SCL says the surface was seen.

    ``dilate_cloud`` grows the cloud and shadow classes by that many pixels.
    The SCL is known to under-call cloud edges, and a thin ring of
    contaminated pixels around every cloud biases reflectance upward in a way
    that survives compositing because it is present in most passes. Two pixels
    at 20 m is the usual compromise. Off by default so the effect can be
    measured before it is applied.
    """
    if scl.ndim != 2:
        raise ValueError(f"SCL must be 2-D, got shape {scl.shape}")
    codes = scl.astype(np.int16)
    clear = np.isin(codes, list(usable))
    if dilate_cloud > 0:
        contaminating = np.isin(
            codes,
            [SCL_CLOUD_SHADOW, SCL_CLOUD_MEDIUM, SCL_CLOUD_HIGH, SCL_THIN_CIRRUS],
        )
        clear &= ~_dilate(contaminating, dilate_cloud)
    return clear


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation without a scipy dependency in the hot path.

    A separable maximum over shifted copies. Small radii only, which is all
    cloud edge buffering ever needs.
    """
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def sentinel2_class_shares(scl: np.ndarray) -> dict[str, float]:
    """What the classifier actually said, as fractions. For diagnostics.

    Worth having because 'the scene was 60% cloud' and 'the scene was 60%
    unclassified' are very different problems and a single clear fraction
    hides which one you have.
    """
    codes = scl.astype(np.int16)
    n = codes.size
    if n == 0:
        return {}
    return {
        label: float((codes == code).sum()) / n
        for code, label in SCL_LABELS.items()
        if (codes == code).any()
    }


# ---------------------------------------------------------------------------
# Landsat Collection 2: the QA_PIXEL bitfield
# ---------------------------------------------------------------------------

QA_FILL = 0
QA_DILATED_CLOUD = 1
QA_CIRRUS = 2
QA_CLOUD = 3
QA_CLOUD_SHADOW = 4
QA_SNOW = 5
QA_CLEAR = 6
QA_WATER = 7

QA_BIT_NAMES = {
    QA_FILL: "fill",
    QA_DILATED_CLOUD: "dilated cloud",
    QA_CIRRUS: "cirrus",
    QA_CLOUD: "cloud",
    QA_CLOUD_SHADOW: "cloud shadow",
    QA_SNOW: "snow",
    QA_CLEAR: "clear",
    QA_WATER: "water",
}


def landsat_clear(
    qa: np.ndarray,
    *,
    exclude_shadow: bool = True,
    exclude_cirrus: bool = True,
) -> np.ndarray:
    """True where Landsat Collection 2 QA_PIXEL says the surface was seen.

    A correction worth recording. Findings/01 used bit 6 alone. Bit 6 is
    defined as 'cloud and dilated cloud bits are not set', which says nothing
    about cloud shadow or cirrus, both of which have their own bits and both of
    which corrupt surface reflectance. Bit 6 alone therefore lets shadow
    through, and shadow is the failure that matters most here because a shadowed
    crop pixel reads as low NDVI and is indistinguishable from a failed crop.

    This masks bit 6 set, and bits 0, 4 and 2 clear. The February numbers in
    findings/01 are consequently a slight overcount of usable Landsat looks; the
    corrected figure is measured in findings/03 rather than assumed here.
    """
    if qa.ndim != 2:
        raise ValueError(f"QA_PIXEL must be 2-D, got shape {qa.shape}")
    bits = qa.astype(np.uint16)
    clear = _bit(bits, QA_CLEAR) & ~_bit(bits, QA_FILL)
    if exclude_shadow:
        clear &= ~_bit(bits, QA_CLOUD_SHADOW)
    if exclude_cirrus:
        clear &= ~_bit(bits, QA_CIRRUS)
    return clear


def _bit(arr: np.ndarray, n: int) -> np.ndarray:
    return ((arr >> np.uint16(n)) & np.uint16(1)).astype(bool)


def landsat_bit_shares(qa: np.ndarray) -> dict[str, float]:
    """Fraction of pixels with each QA bit set. For diagnostics."""
    bits = qa.astype(np.uint16)
    n = bits.size
    if n == 0:
        return {}
    return {name: float(_bit(bits, b).sum()) / n for b, name in QA_BIT_NAMES.items()}


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


# Both bounds are conventions rather than physical constants, and both are set
# from a measurement rather than from instinct.
#
# The floor is the one that matters and the one that is easy to get wrong.
# Reflectance cannot physically be negative, so 0.0 looks like the obviously
# correct floor. It is not. Sen2Cor over-corrects for aerosol over dark
# vegetated targets, which pushes retrieved blue reflectance slightly below
# zero on perfectly good pixels. Measured on a low-cloud Sentinel-2 scene over
# Harare, 11 November 2024, across pixels the SCL itself called clear:
#
#   band   median      1st pct    fraction < 0
#   blue   -0.0092     -0.0481    62%
#   red    +0.0497     -0.0256    11%
#   nir    +0.1535     +0.0647     0%
#   swir1  +0.2651     +0.1312     0%
#
# A floor of 0.0 therefore discards nearly two thirds of the clear pixels in
# any scene where blue is requested, and because validity accumulates across
# bands it destroys the whole observation. That was a real bug in this
# package, found by comparing a pipeline run against an earlier hand
# measurement; the numbers above are why the floor is where it is.
#
# -0.1 keeps the over-correction and still rejects genuine failures, which sit
# far lower. The ceiling above 1.0 is the mirror case: bright surfaces and
# specular returns legitimately exceed unity, and clipping them would fabricate
# data.
REFLECTANCE_FLOOR = -0.1
REFLECTANCE_CEILING = 1.6


def valid_range(
    reflectance: np.ndarray,
    low: float = REFLECTANCE_FLOOR,
    high: float = REFLECTANCE_CEILING,
) -> np.ndarray:
    """True where a surface reflectance value is plausible after correction.

    See the note on REFLECTANCE_FLOOR above: the floor is negative on purpose,
    and setting it to zero silently destroys most of the blue band.
    """
    return np.isfinite(reflectance) & (reflectance >= low) & (reflectance <= high)


def dilation_pixels(distance_m: float, resolution_m: float) -> int:
    """Convert a ground buffer distance into whole pixels for this grid.

    The bug this exists to prevent: a buffer specified as '2 pixels' means 40 m
    on a 20 m grid and 400 m on a 200 m grid, so the same code produces a
    different cloud mask at every resolution and the coverage statistics stop
    being comparable between runs. A buffer is a physical distance and has to
    be stated as one.

    Rounds to nearest, so a buffer smaller than half a pixel becomes no
    dilation at all, which is the honest outcome: you cannot buffer by less
    than you can resolve.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution must be positive, got {resolution_m}")
    return round(distance_m / resolution_m)
