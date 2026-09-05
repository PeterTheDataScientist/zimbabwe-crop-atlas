"""Vegetation and moisture indices, computed from harmonised bands.

Each index is a named function with its bands declared, so a caller can ask
whether an index is computable from what a sensor supplied before trying. That
matters here more than usual: the red edge indices are the strongest signals
available for crop condition and only one of the three sensors carries them, so
'can I compute this' is a question with a different answer every season.

All of them return NaN where inputs are NaN, which propagates the validity mask
without needing to carry it separately.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cropatlas.observation import Observation


@dataclass(frozen=True)
class Index:
    name: str
    bands: tuple[str, ...]
    compute: Callable[..., np.ndarray]
    description: str
    valid_range: tuple[float, float] = (-1.0, 1.0)
    clamp_negative: bool = True

    def __call__(self, source: Mapping[str, np.ndarray] | Observation) -> np.ndarray:
        bands = source.bands if isinstance(source, Observation) else source
        missing = [b for b in self.bands if b not in bands]
        if missing:
            raise KeyError(
                f"{self.name} needs {', '.join(self.bands)}; "
                f"missing {', '.join(missing)}"
            )
        arrays = [np.asarray(bands[b], dtype=np.float32) for b in self.bands]
        if self.clamp_negative:
            arrays = [np.maximum(a, np.float32(0.0)) for a in arrays]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.compute(*arrays)
        lo, hi = self.valid_range
        return np.where((out >= lo) & (out <= hi), out, np.nan).astype(np.float32)

    # Why clamp_negative exists, and why it defaults to True.
    #
    # Surface reflectance is allowed to come back slightly negative, because
    # Sen2Cor over-corrects aerosol over dark vegetation and those pixels are
    # good observations (see masks.REFLECTANCE_FLOOR). That is correct for
    # compositing, where a value of -0.009 carries real information.
    #
    # It is not correct for a ratio. NDVI is (nir - red) / (nir + red), and a
    # negative red inflates the numerator while shrinking the denominator, so
    # red = -0.05 with nir = 0.30 gives NDVI = 1.40. That is not a very green
    # pixel, it is arithmetic on an artefact. The range check then turns it
    # into NaN, and the pixel disappears from the map.
    #
    # This was not hypothetical. Fixing the reflectance floor made a whole
    # season's February zone statistics collapse from reportable to withheld,
    # because the composite had data everywhere and NDVI had NaN in most of it.
    # Two individually correct decisions produced a wrong answer where they
    # met, which is the characteristic failure of a pipeline and the reason the
    # end-to-end run has to be checked against a hand measurement.
    #
    # Clamping at zero is the right repair rather than widening the NDVI range,
    # because reflectance genuinely cannot be negative: zero is the physical
    # bound and therefore the best estimate of the true value for a pixel whose
    # correction overshot. The pixel keeps its place in the composite and gets
    # a defensible index.

    def computable_from(self, source: Mapping[str, np.ndarray] | Observation) -> bool:
        bands = source.bands if isinstance(source, Observation) else source
        return all(b in bands for b in self.bands)


def _normalised(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), the shape most of these share."""
    denom = a + b
    return np.where(np.abs(denom) < 1e-6, np.nan, (a - b) / denom)


NDVI = Index(
    name="ndvi",
    bands=("nir", "red"),
    compute=_normalised,
    description=(
        "Normalised difference vegetation index. The default crop signal, and "
        "the one everything else is compared against. Saturates over dense "
        "canopy, which for Zimbabwean smallholder maize is rarely the binding "
        "problem, but it is why EVI and the red edge indices are also here."
    ),
)

EVI = Index(
    name="evi",
    bands=("nir", "red", "blue"),
    compute=lambda nir, red, blue: 2.5
    * (nir - red)
    / np.where(np.abs(nir + 6.0 * red - 7.5 * blue + 1.0) < 1e-6, np.nan,
               nir + 6.0 * red - 7.5 * blue + 1.0),
    description=(
        "Enhanced vegetation index. Resists soil background and aerosol better "
        "than NDVI and does not saturate as early. The blue term is why it is "
        "more sensitive than NDVI to a residual haze the cloud mask missed, so "
        "it is most useful alongside NDVI rather than instead of it."
    ),
    valid_range=(-1.0, 2.5),
)

SAVI = Index(
    name="savi",
    bands=("nir", "red"),
    compute=lambda nir, red: (1.5 * (nir - red))
    / np.where(np.abs(nir + red + 0.5) < 1e-6, np.nan, nir + red + 0.5),
    description=(
        "Soil adjusted vegetation index, L = 0.5. Built for exactly this "
        "landscape: partial canopy over bare soil, which is what a smallholder "
        "maize plot looks like from space for the first six weeks after "
        "planting. NDVI over sparse canopy is dominated by the soil line and "
        "SAVI is the standard correction for it."
    ),
    valid_range=(-1.5, 1.5),
)

NDMI = Index(
    name="ndmi",
    bands=("nir", "swir1"),
    compute=_normalised,
    description=(
        "Normalised difference moisture index. Canopy water content. Drops "
        "before NDVI does under water stress, because a stressed plant loses "
        "leaf water before it loses greenness, which makes this the earliest "
        "optical warning available for a mid-season dry spell."
    ),
)

NBR = Index(
    name="nbr",
    bands=("nir", "swir2"),
    compute=_normalised,
    description=(
        "Normalised burn ratio. Included not for fire but for residue burning, "
        "which is common between seasons here and produces a sharp signature "
        "that would otherwise be misread as an anomalous crop failure."
    ),
)

NDRE = Index(
    name="ndre",
    bands=("nir", "rededge1"),
    compute=_normalised,
    description=(
        "Normalised difference red edge. Sentinel-2 only. Tracks canopy "
        "chlorophyll and therefore nitrogen status, and unlike NDVI it keeps "
        "responding after the canopy closes. The single strongest argument for "
        "keeping Sentinel-2 as the reference sensor rather than harmonising "
        "everything down to the Landsat band set."
    ),
)

CIRE = Index(
    name="cire",
    bands=("nir", "rededge1"),
    compute=lambda nir, re1: np.where(np.abs(re1) < 1e-6, np.nan, nir / re1) - 1.0,
    description=(
        "Red edge chlorophyll index. Sentinel-2 only. Near linear in canopy "
        "chlorophyll content across a wider range than NDRE, at the cost of "
        "being unbounded and therefore noisier at low canopy cover."
    ),
    valid_range=(-1.0, 20.0),
)

ALL_INDICES: tuple[Index, ...] = (NDVI, EVI, SAVI, NDMI, NBR, NDRE, CIRE)
BY_NAME: Mapping[str, Index] = {i.name: i for i in ALL_INDICES}


def available_indices(source: Mapping[str, np.ndarray] | Observation) -> tuple[str, ...]:
    """Which indices this observation can actually produce.

    The honest way to drive a UI: offer what the data supports for the season
    the user selected, rather than offering everything and rendering an empty
    map for the red edge indices in a Landsat-only month.
    """
    return tuple(i.name for i in ALL_INDICES if i.computable_from(source))


def compute_all(
    source: Mapping[str, np.ndarray] | Observation,
    names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    wanted = names if names is not None else available_indices(source)
    out = {}
    for n in wanted:
        idx = BY_NAME[n]
        if idx.computable_from(source):
            out[n] = idx(source)
    return out
