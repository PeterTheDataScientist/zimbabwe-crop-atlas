"""The 2025/26 radar return: is it real, which satellite, and does it hold?

The source-archive count found zero Sentinel-1 over the test box for three
growing seasons and then 134 products in 2025/26. Before that becomes a
finding it has to survive three questions:

  1. Is it spread across the season, or one burst that happens to fall inside
     the window? A usable time series needs regular passes.
  2. Which platform? If it is Sentinel-1C the return is structural, because
     1C restored the constellation to two satellites. If it is 1A alone the
     return is a plan change that could be reversed.
  3. Does it hold outside the test box? Harare could be a local exception.

Also re-runs the whole-country grid that findings/01 used, so the two
measurements are directly comparable.
"""

from __future__ import annotations

import collections
import json
import urllib.parse
import urllib.request

BOX = (30.95, -18.05, 31.35, -17.75)
TIMEOUT = 90


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "cropatlas/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "replace")


def wkt_box(box) -> str:
    w, s, e, n = box
    ring = [(w, s), (e, s), (e, n), (w, n), (w, s)]
    return "POLYGON((" + ",".join(f"{x} {y}" for x, y in ring) + "))"


def cdse_products(start: str, end: str, box=BOX, top: int = 1000) -> list[dict]:
    """Pull the product records themselves, not just a count.

    A count cannot tell you whether eight scenes are eight weekly passes or
    eight frames of one day. The names carry platform and timestamp, so one
    listing answers both questions.
    """
    filt = (
        "Collection/Name eq 'SENTINEL-1'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_box(box)}')"
        f" and ContentDate/Start gt {start}T00:00:00.000Z"
        f" and ContentDate/Start lt {end}T00:00:00.000Z"
    )
    qs = urllib.parse.urlencode(
        {"$filter": filt, "$top": str(top), "$orderby": "ContentDate/Start asc"}
    )
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{qs}"
    return json.loads(_get(url)).get("value", [])


def summarise(products: list[dict]) -> dict:
    by_month: collections.Counter = collections.Counter()
    by_platform: collections.Counter = collections.Counter()
    by_type: collections.Counter = collections.Counter()
    days: set[str] = set()
    for p in products:
        name = p.get("Name", "")
        started = p.get("ContentDate", {}).get("Start", "")
        if started:
            by_month[started[:7]] += 1
            days.add(started[:10])
        # S1A_IW_GRDH_1SDV_20251104T032... -> platform is the first token
        by_platform[name[:3]] += 1
        for t in ("GRDH", "GRDM", "SLC", "OCN", "RAW"):
            if f"_{t}_" in name or f"_{t}" in name:
                by_type[t] += 1
                break
    return {
        "products": len(products),
        "distinct_days": len(days),
        "by_month": dict(sorted(by_month.items())),
        "by_platform": dict(by_platform),
        "by_type": dict(by_type),
        "days": sorted(days),
    }


def main() -> int:
    out: dict = {}

    print("== 2025/26 season over the test box ==")
    prods = cdse_products("2025-11-01", "2026-05-01")
    s = summarise(prods)
    out["box_2025_26"] = s
    print(f"  {s['products']} products on {s['distinct_days']} distinct days")
    print(f"  by month:    {s['by_month']}")
    print(f"  by platform: {s['by_platform']}")
    print(f"  by type:     {s['by_type']}")

    print("\n== the three empty seasons, re-checked with a listing ==")
    for label, a, b in [
        ("2022/23", "2022-11-01", "2023-05-01"),
        ("2023/24", "2023-11-01", "2024-05-01"),
        ("2024/25", "2024-11-01", "2025-05-01"),
    ]:
        p = cdse_products(a, b)
        print(f"  {label}: {len(p)} products")
        out[f"box_{label.replace('/', '_')}"] = summarise(p)

    print("\n== does the return hold outside Harare? four boxes across the maize belt ==")
    regions = {
        "Harare/Goromonzi": BOX,
        "Chinhoyi/Makonde": (29.6, -17.6, 30.2, -17.1),
        "Gweru/Midlands": (29.6, -19.8, 30.2, -19.3),
        "Bulawayo/Matabeleland": (28.3, -20.4, 28.9, -19.9),
    }
    out["regions_2025_26"] = {}
    for nm, bx in regions.items():
        p = cdse_products("2025-11-01", "2026-05-01", box=bx)
        sm = summarise(p)
        out["regions_2025_26"][nm] = sm
        print(f"  {nm:24s} {sm['products']:4d} products, {sm['distinct_days']:3d} days")

    print("\n== is it still running now, into the 2026 dry season? ==")
    p = cdse_products("2026-05-01", "2026-09-05")
    sm = summarise(p)
    out["box_2026_dry"] = sm
    print(f"  May to Sep 2026: {sm['products']} products on {sm['distinct_days']} days")
    print(f"  by month: {sm['by_month']}")

    with open("findings/radar-return-2025-26.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote findings/radar-return-2025-26.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
