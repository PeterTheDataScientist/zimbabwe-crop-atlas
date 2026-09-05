"""Reading a window of a cloud optimised GeoTIFF onto a target grid.

The whole point of a COG is that you can read the part you want without
downloading the file. Everything here exists to make sure that stays true:
one WarpedVRT per asset, a window computed in the source CRS, and no full-scene
reads anywhere.

Reprojection is not optional in this pipeline and it is worth saying why.
Sentinel-2 over Harare arrives on EPSG:32736 with 10 m pixels. The Landsat
scenes covering the same ground arrive on EPSG:32636 with 30 m pixels and
negative northings. Those are different grids in every respect, and any
cross-sensor arithmetic before they are on a common grid is comparing pixels
that are not looking at the same ground.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cropatlas.observation import Grid


def read_onto_grid(
    href: str,
    grid: Grid,
    *,
    resampling: str = "bilinear",
    fill: float = np.nan,
    dtype: str = "float32",
) -> np.ndarray:
    """Read one asset, reprojected and resampled onto the target grid.

    ``resampling`` must be nearest for anything categorical. Bilinear
    interpolation of a Scene Classification Layer produces class 6.5, which is
    not a class, and the resulting mask is quietly wrong in a way that shows up
    as speckle at cloud edges rather than as an error.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    from cropatlas.io.stac import GDAL_ENV

    how = getattr(Resampling, resampling)
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src, WarpedVRT(
        src,
        crs=grid.crs,
        transform=_affine(grid.transform),
        width=grid.width,
        height=grid.height,
        resampling=how,
        src_nodata=src.nodata,
        nodata=None,
    ) as vrt:
        data = vrt.read(
            1,
            window=Window(0, 0, grid.width, grid.height),
            out_dtype=dtype,
            masked=True,
        )
    arr = np.asarray(data.filled(fill) if hasattr(data, "filled") else data)
    return arr.astype(dtype)


def read_categorical(href: str, grid: Grid, fill: int = 0) -> np.ndarray:
    """Read a class or bitfield asset with nearest neighbour resampling.

    Separated from ``read_onto_grid`` rather than left as a parameter, because
    a caller who forgets to pass ``resampling='nearest'`` for an SCL band gets
    a subtly wrong cloud mask and no error. Making it a different function
    means the mistake is not available.
    """
    return read_onto_grid(
        href, grid, resampling="nearest", fill=float(fill), dtype="float64"
    ).astype(np.int32)


def _affine(transform: tuple[float, float, float, float, float, float]) -> Any:
    from affine import Affine

    return Affine(*transform)


def grid_for_bbox(
    bbox: tuple[float, float, float, float],
    resolution_m: float,
    crs: str | None = None,
) -> Grid:
    """A projected grid covering a lon/lat box at a stated resolution.

    The CRS is chosen as the UTM zone containing the box centre unless one is
    given. Working in UTM rather than in degrees is what makes ``pixel_area_m2``
    meaningful, and area is what every district statistic ultimately rests on.

    Resolution is an argument with real cost attached. The Harare test box is
    about 42 by 33 km: at 10 m that is 4200 by 3300 pixels per band per scene,
    and a season of three sensors across six bands is tens of gigabytes over
    the network. At 60 m the same season is a few hundred megabytes and every
    district statistic is unchanged, because a Zimbabwean maize plot is
    typically under two hectares and neither resolution resolves individual
    fields reliably. Choose 10 or 20 m when field boundaries are the subject,
    60 m when district phenology is.
    """
    w, s, e, n = bbox
    if crs is None:
        zone = int((w + e) / 2 // 6) + 31
        south = (s + n) / 2 < 0
        crs = f"EPSG:{32700 + zone if south else 32600 + zone}"

    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tf.transform([w, e, w, e], [s, s, n, n])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Snap the origin outward to a multiple of the resolution. Two grids built
    # from slightly different boxes then land on the same pixel centres, which
    # is what makes two districts comparable and two seasons stackable.
    minx = np.floor(minx / resolution_m) * resolution_m
    maxy = np.ceil(maxy / resolution_m) * resolution_m
    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))

    return Grid(
        transform=(resolution_m, 0.0, float(minx), 0.0, -resolution_m, float(maxy)),
        crs=crs,
        shape=(height, width),
    )
