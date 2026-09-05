"""Phase 1 end to end over Harare Metropolitan, for two contrasting seasons.

2024/25 has no radar. 2025/26 has a full radar season. Running both through the
same code is the demonstration that the sensor set is an input rather than an
assumption, and the comparison is the headline result.

Everything measured here goes into findings/03. Nothing is asserted that the
run did not produce.

Usage:
    python scripts/run_phase1.py [--resolution 100] [--season 2024/25]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cropatlas.composite import coverage_table
from cropatlas.district import (
    coverage_bias_check,
    grid_zones,
    roll_up,
    statistics_table,
    summarise,
)
from cropatlas.harmonise import (
    find_coincident,
    sensor_offset_report,
)
from cropatlas.indices import NDVI
from cropatlas.io import cog, stac
from cropatlas.pipeline import inventory, run_season
from cropatlas.sensors.deafrica import (
    Sentinel1RTC,
    Sentinel2,
    landsat8,
    landsat9,
)

# Harare and Goromonzi, the same box every finding in this project uses.
BOX = (30.95, -18.05, 31.35, -17.75)

SEASONS = {
    "2021/22": (dt.date(2021, 11, 1), dt.date(2022, 5, 1)),
    "2022/23": (dt.date(2022, 11, 1), dt.date(2023, 5, 1)),
    "2023/24": (dt.date(2023, 11, 1), dt.date(2024, 5, 1)),
    "2024/25": (dt.date(2024, 11, 1), dt.date(2025, 5, 1)),
    "2025/26": (dt.date(2025, 11, 1), dt.date(2026, 5, 1)),
}

OPTICAL_BANDS_WANTED = ("blue", "red", "nir", "swir1")
RADAR_BANDS_WANTED = ("vv", "vh")


def optical_sensors() -> list:
    return [Sentinel2(), landsat8(), landsat9()]


def all_sensors() -> list:
    return [Sentinel2(), landsat8(), landsat9(), Sentinel1RTC()]


def jsonable(obj):
    """Make numpy and infinity safe for json.dump."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        if math.isnan(f):
            return None
        return f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    return obj


def run_inventory_sweep() -> dict:
    """The coverage audit across every season, from the pipeline itself.

    Findings/01 and 02 were measured with throwaway scripts. Reproducing them
    through the shipped code is the check that the package actually does what
    the findings claim, and it is the difference between a result and an
    anecdote.
    """
    print("\n=== Inventory sweep: what each season actually holds ===\n")
    out = {}
    rows = []
    for label, (s, e) in SEASONS.items():
        inv = inventory(
            all_sensors(),
            BOX,
            dt.datetime.combine(s, dt.time.min),
            dt.datetime.combine(e, dt.time.min),
        )
        counts = {k: v.count for k, v in inv.by_sensor.items()}
        days = {k: len(v.days) for k, v in inv.by_sensor.items()}
        out[label] = {
            "scenes": counts,
            "days": days,
            "available": list(inv.available),
            "empty": list(inv.empty),
            "unreachable": list(inv.unreachable),
        }
        rows.append((label, counts, days))
        print(f"  {label}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\n| season  | s2 | l8 | l9 | s1 | optical days | radar days |")
    print("|---------|---:|---:|---:|---:|-------------:|-----------:|")
    for label, counts, days in rows:
        opt_days = max(days.get("s2", 0), 0) + days.get("l8", 0) + days.get("l9", 0)
        print(
            f"| {label} | {counts.get('s2', 0)} | {counts.get('l8', 0)} | "
            f"{counts.get('l9', 0)} | {counts.get('s1', 0)} | ~{opt_days} | "
            f"{days.get('s1', 0)} |"
        )
    return out


def run_one_season(
    label: str,
    resolution: float,
    with_radar: bool,
    fallback=None,
    fallback_label: str = "",
) -> tuple[dict, object]:
    start, end = SEASONS[label]
    grid = cog.grid_for_bbox(BOX, resolution)
    print(f"\n=== Season {label} at {resolution:.0f} m ===")
    print(f"grid {grid.crs} {grid.shape} pixel {grid.pixel_area_m2:.0f} m2")

    sensors = all_sensors() if with_radar else optical_sensors()
    bands = (*OPTICAL_BANDS_WANTED, *(RADAR_BANDS_WANTED if with_radar else ()))
    t0 = time.time()
    res = run_season(
        sensors, BOX, start, end, grid,
        bands=bands, index="ndvi", workers=6,
        fallback_calibration=fallback,
        fallback_label=fallback_label or "another season",
        progress=lambda m: print(f"  {m}") if "\n" not in m else None,
    )
    print(f"loaded in {time.time() - t0:.0f}s: {res.stack.summary()}")
    if res.failures:
        print(f"  {len(res.failures)} scene(s) failed to read")

    record: dict = {
        "season": label,
        "resolution_m": resolution,
        "grid": {"crs": grid.crs, "shape": list(grid.shape)},
        "stack": res.stack.summary(),
        "sensors": list(res.stack.sensors()),
        "scenes_by_sensor": dict(res.stack.by_sensor()),
        "failures": len(res.failures),
    }

    # --- cross-sensor calibration, measured over Zimbabwe ---
    print("\n-- cross-sensor calibration, fitted on this season --")
    print(res.harmoniser.report())
    record["calibration"] = jsonable(res.calibration)

    pairs = find_coincident(res.stack.optical(), "s2")
    offsets = {}
    for band in OPTICAL_BANDS_WANTED:
        rep = sensor_offset_report(pairs, band)
        if rep:
            offsets[band] = rep
            print(
                f"   residual after harmonisation, {band}: "
                f"mean {rep['mean_difference']:+.5f}  n={rep['n']:,}"
            )
    record["residual_offsets"] = jsonable(offsets)

    # --- coverage, the findings/01 table regenerated ---
    print("\n-- monthly coverage, all sensors --")
    table_all = coverage_table(res.monthly)
    print(table_all)
    record["coverage_all"] = {
        m: {
            "scenes": sum(c.provenance.scenes_by_sensor.values()),
            "mean_looks": c.mean_looks,
            "blind_fraction": c.blind_fraction,
            "coverage_3plus": c.coverage(3),
            "sensors": list(c.provenance.sensors),
        }
        for m, c in sorted(res.monthly.items())
    }

    # --- the same season with Sentinel-2 alone, for the gain ---
    from cropatlas.composite import monthly_composites

    s2_only = res.stack.optical()
    s2_only = type(s2_only)(
        s2_only.grid, tuple(o for o in s2_only if o.sensor == "s2")
    )
    monthly_s2 = monthly_composites(
        s2_only, list(OPTICAL_BANDS_WANTED),
        dt.datetime.combine(start, dt.time.min),
        dt.datetime.combine(end, dt.time.min),
    )
    print("\n-- the Landsat gain, measured --")
    print("| month   | S2 alone | with L8+L9 | gain | blind S2 | blind both |")
    print("|---------|---------:|-----------:|-----:|---------:|-----------:|")
    gains = {}
    for month in sorted(res.monthly):
        a = monthly_s2[month]
        b = res.monthly[month]
        gains[month] = {
            "s2_looks": a.mean_looks,
            "all_looks": b.mean_looks,
            "gain": b.mean_looks - a.mean_looks,
            "blind_s2": a.blind_fraction,
            "blind_all": b.blind_fraction,
        }
        print(
            f"| {month} | {a.mean_looks:.2f} | {b.mean_looks:.2f} | "
            f"{b.mean_looks - a.mean_looks:+.2f} | {a.blind_fraction * 100:.1f}% | "
            f"{b.blind_fraction * 100:.1f}% |"
        )
    record["landsat_gain"] = jsonable(gains)

    # --- phenology from the district mean series ---
    print("\n-- district phenology --")
    dates, vals, looks = res.mean_index_series()
    for d, v, lk in zip(dates, vals, looks, strict=True):
        print(f"   {d:%Y-%m}  ndvi {v:.3f}  mean looks {lk:.2f}")
    obs_times, obs_vals, obs_w, obs_sensors = res.observation_series()
    print(f"   per-acquisition series: {len(obs_times)} points "
          f"(monthly series has {len(dates)})")
    ph = res.district_phenology(start, end, per_observation=True)
    print(ph.report())
    ph_monthly = res.district_phenology(start, end, per_observation=False)
    print(f"   for comparison, from monthly composites: peak "
          f"{ph_monthly.date_of(ph_monthly.peak_of_season)}, "
          f"confidence {ph_monthly.confidence:.2f}")
    record["observation_series"] = jsonable(
        [{"when": t.isoformat(), "ndvi": v, "coverage": w, "sensor": sn}
         for t, v, w, sn in zip(obs_times, obs_vals, obs_w, obs_sensors, strict=True)]
    )
    record["phenology_monthly"] = jsonable(
        {
            "peak": ph_monthly.date_of(ph_monthly.peak_of_season),
            "confidence": ph_monthly.confidence,
        }
    )
    record["ndvi_series"] = jsonable(
        [{"month": d.isoformat(), "ndvi": v, "looks": lk}
         for d, v, lk in zip(dates, vals, looks, strict=True)]
    )
    record["phenology"] = jsonable(
        {
            "start_of_season": ph.date_of(ph.start_of_season),
            "peak": ph.date_of(ph.peak_of_season),
            "end_of_season": ph.date_of(ph.end_of_season),
            "length_days": ph.length_of_season,
            "amplitude": ph.amplitude,
            "baseline": ph.baseline,
            "integral": ph.integral,
            "confidence": ph.confidence,
            "largest_gap_days": ph.largest_gap_days,
            "notes": list(ph.notes),
        }
    )

    # --- what radar adds, in the season that has it ---
    radar = res.stack.radar()
    if len(radar):
        print("\n-- radar, the fourth sensor --")
        print(f"   {radar.summary()}")
        rad_monthly = monthly_composites(
            radar, list(RADAR_BANDS_WANTED),
            dt.datetime.combine(start, dt.time.min),
            dt.datetime.combine(end, dt.time.min),
        )
        print("| month   | radar passes | mean looks | never seen | optical looks |")
        print("|---------|-------------:|-----------:|-----------:|--------------:|")
        rad_rows = {}
        for month in sorted(rad_monthly):
            rc = rad_monthly[month]
            oc = res.monthly.get(month)
            n = sum(rc.provenance.scenes_by_sensor.values())
            rad_rows[month] = {
                "passes": n,
                "mean_looks": rc.mean_looks,
                "blind": rc.blind_fraction,
                "optical_looks": oc.mean_looks if oc else None,
            }
            print(
                f"| {month} | {n} | {rc.mean_looks:.2f} | "
                f"{rc.blind_fraction * 100:.1f}% | "
                f"{oc.mean_looks:.2f} |" if oc else "|"
            )
        record["radar"] = jsonable(
            {"summary": radar.summary(), "monthly": rad_rows}
        )
        # Radar sees through cloud, so its coverage should be flat across the
        # season where optical collapses. That contrast is the whole argument
        # for having it, and it is worth measuring rather than asserting.
        feb_r = rad_rows.get(f"{start.year + 1}-02")
        dec_r = rad_rows.get(f"{start.year}-12")
        if feb_r and dec_r and dec_r["mean_looks"]:
            ratio = feb_r["mean_looks"] / dec_r["mean_looks"]
            print(f"   February/December radar look ratio: {ratio:.2f}")
            print("   (optical ratio: "
                  f"{feb_r['optical_looks'] / dec_r['optical_looks']:.2f})")
            record["radar"]["feb_dec_ratio"] = jsonable(ratio)
    else:
        print("\n-- radar: nothing held for this season --")
        record["radar"] = None

    # --- zone roll-up on the February composite, where it is hardest ---
    feb = next((m for m in sorted(res.monthly) if m.endswith("-02")), None)
    if feb:
        comp = res.monthly[feb]
        ndvi = NDVI(comp.bands) if NDVI.computable_from(comp.bands) else None
        if ndvi is not None:
            zones = grid_zones(res.grid, 4, 4)
            stats = roll_up(ndvi, comp, zones)
            print(f"\n-- {feb} zone roll-up, 4x4 lattice --")
            print(statistics_table(stats))
            s = summarise(stats)
            print(f"   {s['reported']}/{s['zones']} zones reportable")
            record["february_zones"] = jsonable(
                {
                    "month": feb,
                    "summary": s,
                    "withheld": [z.zone for z in stats if not z.reported],
                }
            )

            # Is the missing data structured? Use NDVI's own spatial mean as a
            # stand-in covariate: a proper run uses elevation, but the question
            # of whether unseen ground differs from seen ground is answerable
            # with whatever covariate is to hand and is worth asking always.
            jan = next((m for m in sorted(res.monthly) if m.endswith("-01")), None)
            if jan and NDVI.computable_from(res.monthly[jan].bands):
                jan_ndvi = NDVI(res.monthly[jan].bands)
                bias = coverage_bias_check(
                    ndvi, comp.clear_count, jan_ndvi, "january ndvi"
                )
                print(f"   coverage bias check: {jsonable(bias)}")
                record["february_coverage_bias"] = jsonable(bias)

    return record, res.harmoniser


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=100.0)
    ap.add_argument("--seasons", nargs="*", default=["2024/25", "2025/26"])
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    stac.apply_gdal_env()
    out: dict = {"box": BOX, "generated": dt.datetime.now().isoformat()}

    if not args.skip_sweep:
        out["inventory_sweep"] = run_inventory_sweep()

    out["seasons"] = {}
    # Seasons run in order, and each carries its fitted calibration forward.
    # The 2025/26 season could not fit its own from 66 coincident pairs, and
    # the instrument difference it needs was measured cleanly on 2024/25.
    carried = None
    carried_label = ""
    for label in args.seasons:
        with_radar = label in ("2021/22", "2025/26")
        record, harmoniser = run_one_season(
            label, args.resolution, with_radar,
            fallback=carried, fallback_label=carried_label,
        )
        out["seasons"][label] = record
        if harmoniser.adjustments and harmoniser.fitted_on is None:
            carried, carried_label = harmoniser, label
            os.makedirs("findings", exist_ok=True)
            harmoniser.save(
                f"findings/calibration-{label.replace('/', '-')}.json"
            )

    os.makedirs("findings", exist_ok=True)
    path = "findings/phase1-results.json"
    with open(path, "w") as fh:
        json.dump(jsonable(out), fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
