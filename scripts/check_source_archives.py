"""Count Sentinel-1 scenes over Zimbabwe in the SOURCE archives.

Digital Earth Africa holds nothing for 2022/23 onward (measured, findings/01).
DE Africa is a curated regional mirror, so an empty mirror is not evidence of
an empty ESA archive. This asks the two archives that hold everything ESA
acquired: the Copernicus Data Space Ecosystem, and the Alaska Satellite
Facility. Both answer unauthenticated for catalogue counts.

If the source archives are also empty, the radar finding is closed for good and
the reason is acquisition, not mirroring. If they hold scenes DE Africa lacks,
radar returns as a Phase 2 option and DE Africa has a gap worth reporting.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

# Same test box as findings/01, so the counts are comparable line for line.
BOX = (30.95, -18.05, 31.35, -17.75)  # w, s, e, n  Harare and Goromonzi

SEASONS = [
    ("2021/22", "2021-11-01", "2022-05-01"),
    ("2022/23", "2022-11-01", "2023-05-01"),
    ("2023/24", "2023-11-01", "2024-05-01"),
    ("2024/25", "2024-11-01", "2025-05-01"),
    ("2025/26", "2025-11-01", "2026-05-01"),
]

TIMEOUT = 60


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "cropatlas/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "replace")


def wkt_box(box: tuple[float, float, float, float]) -> str:
    w, s, e, n = box
    ring = [(w, s), (e, s), (e, n), (w, n), (w, s)]
    return "POLYGON((" + ",".join(f"{x} {y}" for x, y in ring) + "))"


def cdse_count(start: str, end: str, collection: str = "SENTINEL-1") -> int | None:
    """Count products in the Copernicus Data Space Ecosystem OData catalogue.

    OData rather than STAC: the OData endpoint returns an exact $count for a
    spatial and temporal filter in one request, which is all we need. We are
    counting, not downloading.
    """
    filt = (
        f"Collection/Name eq '{collection}'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_box(BOX)}')"
        f" and ContentDate/Start gt {start}T00:00:00.000Z"
        f" and ContentDate/Start lt {end}T00:00:00.000Z"
    )
    qs = urllib.parse.urlencode({"$filter": filt, "$count": "True", "$top": "1"})
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{qs}"
    try:
        body = json.loads(_get(url))
    except urllib.error.HTTPError as exc:
        print(f"    CDSE HTTP {exc.code}: {exc.read()[:200]!r}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"    CDSE error: {exc}", file=sys.stderr)
        return None
    return body.get("@odata.count")


def cdse_modes(start: str, end: str) -> dict[str, int]:
    """Break a season down by product type, so IW GRD is not confused with SLC."""
    out: dict[str, int] = {}
    for ptype in ("GRD", "SLC", "RTC"):
        filt = (
            "Collection/Name eq 'SENTINEL-1'"
            f" and contains(Name,'{ptype}')"
            f" and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_box(BOX)}')"
            f" and ContentDate/Start gt {start}T00:00:00.000Z"
            f" and ContentDate/Start lt {end}T00:00:00.000Z"
        )
        qs = urllib.parse.urlencode({"$filter": filt, "$count": "True", "$top": "1"})
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{qs}"
        try:
            out[ptype] = json.loads(_get(url)).get("@odata.count", -1)
        except Exception:
            out[ptype] = -1
    return out


def asf_count(start: str, end: str) -> int | None:
    """Count granules in the ASF DAAC search API.

    ASF is NASA's mirror of the full Sentinel-1 archive. It indexes what ESA
    acquired, independently of what ESA's own portal chooses to serve, so
    agreement between ASF and CDSE is a genuine cross-check rather than two
    readings of one catalogue.
    """
    w, s, e, n = BOX
    qs = urllib.parse.urlencode(
        {
            "platform": "Sentinel-1",
            "bbox": f"{w},{s},{e},{n}",
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T00:00:00Z",
            "output": "count",
        }
    )
    url = f"https://api.daac.asf.alaska.edu/services/search/param?{qs}"
    try:
        return int(_get(url).strip())
    except Exception as exc:
        print(f"    ASF error: {exc}", file=sys.stderr)
        return None


def main() -> int:
    print(f"Box {BOX}  (same as findings/01)\n")
    rows = []
    for label, start, end in SEASONS:
        print(f"  {label} ...", flush=True)
        rows.append(
            {
                "season": label,
                "cdse_s1": cdse_count(start, end),
                "asf_s1": asf_count(start, end),
                "modes": cdse_modes(start, end),
            }
        )

    print("\n| season  | CDSE S1 | ASF S1 | GRD | SLC | DE Africa s1_rtc |")
    print("|---------|--------:|-------:|----:|----:|-----------------:|")
    known = {"2021/22": 8, "2022/23": 0, "2023/24": 0, "2024/25": 0, "2025/26": None}
    for r in rows:
        de = known.get(r["season"])
        de_s = "not measured" if de is None else str(de)
        print(
            f"| {r['season']} | {r['cdse_s1']} | {r['asf_s1']} |"
            f" {r['modes'].get('GRD')} | {r['modes'].get('SLC')} | {de_s} |"
        )

    with open("findings/source-archive-counts.json", "w") as fh:
        json.dump({"box": BOX, "rows": rows}, fh, indent=2)
    print("\nwrote findings/source-archive-counts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
