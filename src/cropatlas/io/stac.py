"""Talking to Digital Earth Africa's STAC catalogue.

Two things learned the hard way and recorded so nobody repeats them.

``GET /stac/collections`` returns an HTML error page, not JSON. Collections
have to be enumerated through the catalogue's child links. Searching works, but
only as ``POST /stac/search`` with a JSON body; the GET form silently ignores
some parameters.

Reading the COGs needs ``AWS_DEFAULT_REGION=af-south-1`` set alongside
``AWS_NO_SIGN_REQUEST=YES``. Without the region GDAL fails with "location
constraint is incompatible for the region specific endpoint", which reads like
a permissions problem and is not one. The buckets are public; they are simply
not in the default region.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

STAC_ROOT = "https://explorer.digitalearth.africa/stac"
DEFAULT_TIMEOUT = 120

# Applied to every rasterio environment in this package. Set once, here, so
# that a new call site cannot forget the region and get a misleading error.
GDAL_ENV: dict[str, str] = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "AWS_DEFAULT_REGION": "af-south-1",
    "AWS_S3_ENDPOINT": "s3.af-south-1.amazonaws.com",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "268435456",
}


def apply_gdal_env() -> None:
    """Set the GDAL environment in this process.

    Rasterio's ``Env`` context manager is the tidier mechanism, but the
    settings have to survive into worker threads and into any library that
    opens a dataset without going through our code, so they go in the
    environment as well.
    """
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)


def search(
    collection: str,
    bbox: tuple[float, float, float, float],
    start: dt.datetime,
    end: dt.datetime,
    *,
    limit: int = 500,
    max_pages: int = 20,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Every item in a collection intersecting the box in the period.

    Paged, because a season of Sentinel-2 over a district exceeds any single
    page and a truncated search looks exactly like a thin archive. Findings/02
    exists because an empty answer was mistaken for a fact about the world;
    quietly returning page one would be the same mistake with extra steps.
    """
    out: list[dict[str, Any]] = []
    token: str | None = None
    for _ in range(max_pages):
        body: dict[str, Any] = {
            "collections": [collection],
            "bbox": list(bbox),
            "datetime": f"{_iso(start)}/{_iso(end)}",
            "limit": limit,
        }
        if token:
            body["token"] = token
        page = _post(f"{STAC_ROOT}/search", body, timeout=timeout)
        feats = page.get("features", [])
        out.extend(feats)
        token = _next_token(page)
        if not token or not feats:
            break
    return out


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _post(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "cropatlas/0.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _next_token(page: dict[str, Any]) -> str | None:
    for link in page.get("links", []):
        if link.get("rel") == "next":
            body = link.get("body") or {}
            tok = body.get("token")
            if tok:
                return str(tok)
            href = link.get("href", "")
            if "token=" in href:
                return href.split("token=", 1)[1].split("&", 1)[0]
    return None


def collections(timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    """Enumerate collections through the catalogue's child links.

    ``/stac/collections`` returns HTML. This walks the root catalogue instead,
    which is the documented STAC traversal and happens to be the one that
    works here.
    """
    req = urllib.request.Request(
        STAC_ROOT, headers={"User-Agent": "cropatlas/0.1"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        root = json.loads(resp.read())
    return sorted(
        link["href"].rstrip("/").rsplit("/", 1)[-1]
        for link in root.get("links", [])
        if link.get("rel") == "child"
    )


def asset_href(item: dict[str, Any], name: str) -> str:
    """The readable URL for one asset, translated for GDAL.

    DE Africa publishes ``s3://`` hrefs. GDAL wants ``/vsis3/``. Doing the
    translation in one place means a sensor never has to think about it, and
    means switching to the HTTPS endpoints later is a change here rather than
    in every adapter.
    """
    assets = item.get("assets", {})
    if name not in assets:
        raise KeyError(
            f"asset {name!r} not in item {item.get('id')}; "
            f"it has {', '.join(sorted(assets))}"
        )
    href = assets[name]["href"]
    if href.startswith("s3://"):
        return "/vsis3/" + href[len("s3://") :]
    return href


def item_datetime(item: dict[str, Any]) -> dt.datetime:
    raw = item["properties"]["datetime"]
    # STAC datetimes here appear with and without fractional seconds and always
    # in UTC. fromisoformat handles both once the Z is normalised, and it is
    # worth doing rather than reaching for a dateutil dependency for one field.
    cleaned = raw.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(cleaned)
    return parsed.replace(tzinfo=None)


def chunk(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
