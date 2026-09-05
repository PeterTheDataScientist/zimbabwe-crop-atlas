"""Generate every figure from measured results. Nothing here is illustrative.

The line, bar and timeline figures are drawn from findings/phase1-results.json,
which the pipeline wrote. The map and scatter figures need pixel arrays that no
summary file can carry, so those re-run one month of the pipeline: cheap,
because a single month at 200 m is about thirty scenes.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cropatlas import figures
from cropatlas.composite import composite
from cropatlas.harmonise import find_coincident
from cropatlas.io import cog, stac
from cropatlas.phenology import _banded_whittaker, to_daily_grid
from cropatlas.pipeline import inventory, load_stack
from cropatlas.sensors.deafrica import Sentinel2, landsat8, landsat9

BOX = (30.95, -18.05, 31.35, -17.75)
OUT = "figures"
RESULTS = "findings/phase1-results.json"

MONTH_LABEL = {
    "11": "Nov", "12": "Dec", "01": "Jan", "02": "Feb",
    "03": "Mar", "04": "Apr", "05": "May", "10": "Oct",
}


def label(month_key: str) -> str:
    return f"{MONTH_LABEL.get(month_key[5:7], month_key[5:7])} {month_key[2:4]}"


def fig_timeline(data: dict) -> None:
    sweep = data.get("inventory_sweep")
    if not sweep:
        print("  no inventory sweep in results, skipping timeline")
        return
    seasons = list(sweep)
    counts = {
        key: [sweep[s]["scenes"].get(key, 0) for s in seasons]
        for key in ("s2", "l8", "l9", "s1")
    }
    p = figures.sensor_timeline(
        seasons, counts,
        title="Scenes held over the Harare box, by season and instrument",
        path=f"{OUT}/01-sensor-timeline.png",
    )
    print(f"  {p}")


def fig_cliff(data: dict, season: str, radar: bool) -> None:
    s = data["seasons"][season]
    gains = s["landsat_gain"]
    months = sorted(gains)
    rad = None
    if radar and s.get("radar"):
        rad_monthly = s["radar"]["monthly"]
        rad = [rad_monthly.get(m, {}).get("mean_looks", 0.0) for m in months]
    p = figures.coverage_cliff(
        [label(m) for m in months],
        [gains[m]["s2_looks"] for m in months],
        [gains[m]["all_looks"] for m in months],
        [gains[m]["blind_s2"] for m in months],
        [gains[m]["blind_all"] for m in months],
        radar_looks=rad,
        title=f"What each sensor set could see, {season} season, Harare",
        path=f"{OUT}/02-coverage-{season.replace('/', '-')}.png",
    )
    print(f"  {p}")


def fig_february(season_start: dt.date, tag: str) -> None:
    """The February clear-count map, Sentinel-2 alone against all three."""
    grid = cog.grid_for_bbox(BOX, 200.0)
    feb_start = dt.datetime(season_start.year + 1, 2, 1)
    feb_end = dt.datetime(season_start.year + 1, 3, 1)

    sensors = [Sentinel2(), landsat8(), landsat9()]
    inv = inventory(sensors, BOX, feb_start, feb_end)
    stack, _ = load_stack(sensors, inv, grid, ["red", "nir"], workers=6)

    s2_only = type(stack)(grid, tuple(o for o in stack if o.sensor == "s2"))
    a = composite(s2_only, ["red", "nir"])
    b = composite(stack, ["red", "nir"])

    figures.clear_count_map(
        a.clear_count,
        f"February {season_start.year + 1}: Sentinel-2 alone",
        f"mean {a.mean_looks:.2f} clear looks per pixel, "
        f"{a.coverage(3) * 100:.0f}% of the district got three or more",
        path=f"{OUT}/03-february-s2-{tag}.png",
    )
    figures.clear_count_map(
        b.clear_count,
        f"February {season_start.year + 1}: with Landsat 8 and 9",
        f"mean {b.mean_looks:.2f} clear looks per pixel, "
        f"{b.coverage(3) * 100:.0f}% of the district got three or more",
        path=f"{OUT}/04-february-all-{tag}.png",
    )
    print(f"  {OUT}/03-february-s2-{tag}.png and 04-february-all-{tag}.png")

    # The calibration scatter, from the same month's coincident pairs.
    pairs = find_coincident(stack.optical(), "s2")
    l8_pairs = [p for p in pairs if p.other.sensor == "l8"]
    if l8_pairs:
        xs, ys = [], []
        for pair in l8_pairs:
            both = pair.reference.valid & pair.other.valid
            if both.any():
                xs.append(pair.other.band("red")[both])
                ys.append(pair.reference.band("red")[both])
        if xs:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            if x.size > 500:
                from cropatlas.harmonise import _deming, _ols

                sl, ic = _deming(x.astype(float), y.astype(float))
                ols_sl, _ = _ols(x.astype(float), y.astype(float))
                pred = sl * x + ic
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
                p = figures.calibration_scatter(
                    y, x, sl, ic, ols_sl, "red", "Landsat 8", r2,
                    path=f"{OUT}/05-calibration-{tag}.png",
                )
                print(f"  {p}")


def fig_phenology(data: dict, season: str, season_start: dt.date,
                  season_end: dt.date, tag: str) -> None:
    s = data["seasons"][season]
    series = s.get("observation_series")
    if not series:
        print(f"  no observation series for {season}")
        return
    times = [dt.datetime.fromisoformat(r["when"]) for r in series]
    vals = [r["ndvi"] for r in series]
    weights = [r["coverage"] * 6.0 for r in series]

    grid = to_daily_grid(times, vals, season_start, season_end, weights=weights)
    smoothed = _banded_whittaker(grid.values, grid.weights, 100.0)

    ph = s.get("phenology", {})
    marks = {}
    for name, key in (("start", "start_of_season"), ("peak", "peak"),
                      ("end", "end_of_season")):
        v = ph.get(key)
        if v:
            marks[name] = dt.date.fromisoformat(v)

    # Results written before the series carried a sensor label can still be
    # drawn correctly: an inventory is metadata only, so matching acquisition
    # times back to their instrument costs a few seconds rather than a reload.
    if any("sensor" not in r for r in series):
        inv = inventory(
            [Sentinel2(), landsat8(), landsat9()],
            BOX,
            dt.datetime.combine(season_start, dt.time.min),
            dt.datetime.combine(season_end, dt.time.min),
        )
        by_time = {ref.acquired: ref.sensor for ref in inv.all_scenes()}
        sensors = [by_time.get(t, "s2") for t in times]
    else:
        sensors = [r["sensor"] for r in series]
    p = figures.phenology_curve(
        times, vals, sensors,
        smoothed_days=grid.days, smoothed_values=smoothed,
        season_start=season_start, marks=marks,
        title=(
            f"Harare district NDVI, {season} season. "
            f"{len(times)} acquisitions, confidence {ph.get('confidence', 0):.2f}"
        ),
        path=f"{OUT}/06-phenology-{tag}.png",
    )
    print(f"  {p}")


def main() -> int:
    stac.apply_gdal_env()
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS) as fh:
        data = json.load(fh)

    print("figures:")
    fig_timeline(data)
    for season, start, end, radar in (
        ("2024/25", dt.date(2024, 11, 1), dt.date(2025, 5, 1), False),
        ("2025/26", dt.date(2025, 11, 1), dt.date(2026, 5, 1), True),
    ):
        if season not in data.get("seasons", {}):
            continue
        tag = season.replace("/", "-")
        fig_cliff(data, season, radar)
        fig_phenology(data, season, start, end, tag)
    fig_february(dt.date(2024, 11, 1), "2024-25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
