"""Existing at ESA and being usable are different things.

CDSE and ASF hold Sentinel-1 GRD: raw-ish detected amplitude in slant geometry,
uncalibrated for terrain. Turning that into analysis-ready gamma-0 needs orbit
files, a DEM, radiometric terrain correction and speckle handling. That is a
week of work and a lot of compute, and it is exactly what Digital Earth Africa's
s1_rtc product already is.

So the question that decides whether radar enters Phase 1 is not "did ESA
acquire it" (answered: yes, from Nov 2025). It is "has DE Africa processed it",
because DE Africa is the only free source of analysis-ready radar over Africa
that this pipeline can read straight into an array.
"""

from __future__ import annotations

import json
import urllib.request

STAC = "https://explorer.digitalearth.africa/stac"
BOX = [30.95, -18.05, 31.35, -17.75]
TIMEOUT = 90


def search(collection: str, start: str, end: str, box=None, limit: int = 500) -> list:
    body = json.dumps(
        {
            "collections": [collection],
            "bbox": box or BOX,
            "datetime": f"{start}T00:00:00Z/{end}T00:00:00Z",
            "limit": limit,
        }
    ).encode()
    req = urllib.request.Request(
        f"{STAC}/search",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "cropatlas/0.1"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read()).get("features", [])


def main() -> int:
    out = {}
    print("DE Africa s1_rtc, by season, over the test box:\n")
    print("| season  | s1_rtc scenes | distinct days |")
    print("|---------|--------------:|--------------:|")
    for label, a, b in [
        ("2021/22", "2021-11-01", "2022-05-01"),
        ("2022/23", "2022-11-01", "2023-05-01"),
        ("2023/24", "2023-11-01", "2024-05-01"),
        ("2024/25", "2024-11-01", "2025-05-01"),
        ("2025/26", "2025-11-01", "2026-05-01"),
        ("2026 dry", "2026-05-01", "2026-09-05"),
    ]:
        try:
            feats = search("s1_rtc", a, b)
        except Exception as exc:
            print(f"| {label} | error: {exc} | |")
            continue
        days = sorted({f["properties"]["datetime"][:10] for f in feats})
        out[label] = {"scenes": len(feats), "days": days}
        print(f"| {label} | {len(feats)} | {len(days)} |")

    # If s1_rtc is behind, the monthly mosaic may still carry the season.
    print("\nDE Africa s1_monthly_mosaic, 2025/26:")
    try:
        feats = search("s1_monthly_mosaic", "2025-11-01", "2026-05-01")
        months = sorted({f["properties"]["datetime"][:7] for f in feats})
        out["mosaic_2025_26"] = {"scenes": len(feats), "months": months}
        print(f"  {len(feats)} scenes covering {months}")
    except Exception as exc:
        print(f"  error: {exc}")

    # How far forward does the whole s1_rtc collection actually run, anywhere?
    # A collection that stops in 2022 globally is a discontinued product, not a
    # regional gap, and that distinction changes what we tell people.
    print("\nGlobal s1_rtc recency check (whole of Africa bbox, one month windows):")
    africa = [-20.0, -35.0, 55.0, 38.0]
    for label, a, b in [
        ("Feb 2024", "2024-02-01", "2024-03-01"),
        ("Feb 2025", "2025-02-01", "2025-03-01"),
        ("Feb 2026", "2026-02-01", "2026-03-01"),
        ("Jul 2026", "2026-07-01", "2026-08-01"),
    ]:
        try:
            feats = search("s1_rtc", a, b, box=africa, limit=1)
            # limit=1 only tells us empty or not; that is the question here.
            print(f"  {label}: {'has data' if feats else 'EMPTY across Africa'}")
            out[f"africa_{label}"] = bool(feats)
        except Exception as exc:
            print(f"  {label}: error {exc}")

    with open("findings/deafrica-s1-ingest.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote findings/deafrica-s1-ingest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
