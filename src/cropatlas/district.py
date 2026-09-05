"""Rolling pixels up to districts without losing what the pixels knew.

The temptation at this stage is to take a mean and move on. That mean is where
most products like this quietly stop being trustworthy, for a reason that is
easy to state and easy to forget: the pixels that are missing are not missing
at random.

February cloud over Zimbabwe sits on the high ground and along the convergence
zone. Those are not random pixels, they are wetter, higher and often more
productive than the district average. A mean over the pixels that happened to
be visible is therefore biased, and the bias has a sign, and it moves between
months. Findings/01 measured 16.7% of the test district never seen at all in
February on Sentinel-2 alone.

So every statistic here travels with the coverage that produced it, and the
aggregator refuses to report a district number when coverage is below a stated
floor rather than reporting a number computed from a third of the pixels and
letting the caller notice the small print. Refusing is a feature. A blank cell
that says why it is blank is worth more than a plausible number nobody can
audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cropatlas.composite import Composite
from cropatlas.observation import Grid


@dataclass(frozen=True)
class Zone:
    """A named area on the grid: a district, a ward, a scheme, a field."""

    name: str
    mask: np.ndarray
    code: str | None = None

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise TypeError(f"zone {self.name!r} mask must be boolean")

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())

    def area_ha(self, grid: Grid) -> float:
        return self.pixel_count * grid.pixel_area_m2 / 10_000.0


@dataclass(frozen=True)
class ZoneStatistic:
    """One number for one zone, with everything needed to judge it."""

    zone: str
    statistic: str
    value: float | None
    # Evidence
    pixels_in_zone: int
    pixels_observed: int
    mean_looks: float
    reported: bool
    withheld_reason: str | None = None
    # Spread, present only when the value is
    p10: float | None = None
    p90: float | None = None
    std: float | None = None

    @property
    def coverage(self) -> float:
        return self.pixels_observed / self.pixels_in_zone if self.pixels_in_zone else 0.0

    def __str__(self) -> str:
        if not self.reported:
            return f"{self.zone}: withheld ({self.withheld_reason})"
        return (
            f"{self.zone}: {self.statistic}={self.value:.3f} "
            f"(coverage {self.coverage * 100:.0f}%, {self.mean_looks:.1f} looks)"
        )


def zone_statistic(
    values: np.ndarray,
    clear_count: np.ndarray,
    zone: Zone,
    *,
    statistic: str = "mean",
    min_coverage: float = 0.60,
    min_mean_looks: float = 1.0,
) -> ZoneStatistic:
    """Aggregate one array over one zone, or refuse and say why.

    ``min_coverage`` at 0.60 is a convention and is stated as one. It is set
    where it is because below about that level the visible pixels in a
    Zimbabwean February are dominated by the low, dry parts of a district and
    the resulting mean is measurably biased, not merely noisy. Callers who want
    the number anyway can lower it deliberately, which is the point: the choice
    is visible in the call rather than absent from the code.
    """
    in_zone = zone.mask
    n_zone = int(in_zone.sum())
    if n_zone == 0:
        return ZoneStatistic(
            zone=zone.name, statistic=statistic, value=None,
            pixels_in_zone=0, pixels_observed=0, mean_looks=0.0,
            reported=False, withheld_reason="zone has no pixels on this grid",
        )

    vals = values[in_zone]
    looks = clear_count[in_zone]
    observed = np.isfinite(vals)
    n_obs = int(observed.sum())
    mean_looks = float(looks.mean())
    coverage = n_obs / n_zone

    if coverage < min_coverage:
        return ZoneStatistic(
            zone=zone.name, statistic=statistic, value=None,
            pixels_in_zone=n_zone, pixels_observed=n_obs, mean_looks=mean_looks,
            reported=False,
            withheld_reason=(
                f"only {coverage * 100:.0f}% of the zone was seen, floor is "
                f"{min_coverage * 100:.0f}%; cloud gaps here are not random and "
                "a mean over the visible pixels would be biased"
            ),
        )
    if mean_looks < min_mean_looks:
        return ZoneStatistic(
            zone=zone.name, statistic=statistic, value=None,
            pixels_in_zone=n_zone, pixels_observed=n_obs, mean_looks=mean_looks,
            reported=False,
            withheld_reason=(
                f"mean clear looks {mean_looks:.2f} below floor {min_mean_looks}"
            ),
        )

    good = vals[observed]
    value = {
        "mean": np.mean,
        "median": np.median,
        "p10": lambda a: np.quantile(a, 0.10),
        "p90": lambda a: np.quantile(a, 0.90),
        "std": np.std,
    }[statistic](good)

    return ZoneStatistic(
        zone=zone.name, statistic=statistic, value=float(value),
        pixels_in_zone=n_zone, pixels_observed=n_obs, mean_looks=mean_looks,
        reported=True,
        p10=float(np.quantile(good, 0.10)),
        p90=float(np.quantile(good, 0.90)),
        std=float(np.std(good)),
    )


def roll_up(
    values: np.ndarray,
    composite: Composite,
    zones: Sequence[Zone],
    **kwargs: object,
) -> list[ZoneStatistic]:
    return [
        zone_statistic(values, composite.clear_count, z, **kwargs)  # type: ignore[arg-type]
        for z in zones
    ]


def grid_zones(grid: Grid, rows: int, cols: int) -> list[Zone]:
    """Regular tiles over the grid.

    Not a substitute for administrative boundaries. Its purpose is to make the
    coverage bias visible: run the same statistic over a regular lattice and the
    withheld cells trace the shape of the cloud, which is a far more convincing
    demonstration that the gaps are structured than any amount of prose.
    """
    h, w = grid.shape
    zones = []
    for r in range(rows):
        for c in range(cols):
            mask = np.zeros((h, w), dtype=bool)
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            mask[y0:y1, x0:x1] = True
            zones.append(Zone(name=f"r{r}c{c}", mask=mask, code=f"{r}:{c}"))
    return zones


def coverage_bias_check(
    values: np.ndarray,
    clear_count: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str = "elevation",
) -> dict[str, object]:
    """Is what we could not see systematically different from what we could?

    Compares a covariate between seen and unseen pixels. If unseen pixels are on
    average higher, or wetter, or steeper, then the missing data is structured
    and the district mean is biased rather than merely uncertain.

    This is the check that turns 'coverage was 83%' into a statement about
    whether the 83% is representative, and it is the difference between a
    product that reports uncertainty and one that reports the right kind of
    uncertainty.

    The standardised difference is Cohen's d: the gap between the two group
    means in units of their pooled spread. Roughly, 0.2 is a difference you
    would struggle to see, 0.8 is one you could not miss.

    The degenerate case is the one that matters and it is counter-intuitive. If
    both groups have almost no internal spread, the pooled denominator goes to
    zero. An earlier version guarded that by returning 0.0, which inverted the
    meaning: two perfectly uniform groups separated by 400 m of elevation are
    not unbiased, they are the most completely separated case possible, and the
    guard reported the strongest available signal as no signal. A district
    where cloud sits exactly on one terrain class would have passed silently.
    It now reports infinite separation, which is what a zero denominator and a
    non-zero numerator mean.
    """
    seen = np.isfinite(values) & (clear_count > 0)
    unseen = ~seen
    base: dict[str, object] = {"covariate": covariate_name}

    if not seen.any() or not unseen.any():
        return {
            **base,
            "seen_n": int(seen.sum()),
            "unseen_n": int(unseen.sum()),
            "difference": 0.0,
            "standardised_difference": 0.0,
            "perfectly_separated": False,
            "note": "nothing unseen" if not unseen.any() else "nothing seen",
        }

    a = covariate[seen]
    b = covariate[unseen]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            **base,
            "seen_n": int(a.size),
            "unseen_n": int(b.size),
            "difference": 0.0,
            "standardised_difference": 0.0,
            "perfectly_separated": False,
            "note": "covariate missing for one group",
        }

    difference = float(b.mean() - a.mean())
    pooled = float(np.sqrt((a.var() + b.var()) / 2.0))
    scale = max(abs(float(a.mean())), abs(float(b.mean())), 1e-9)
    degenerate = pooled <= scale * 1e-9

    if degenerate:
        separated = abs(difference) > scale * 1e-9
        std_diff = float("inf") * np.sign(difference) if separated else 0.0
    else:
        separated = False
        std_diff = difference / pooled

    return {
        **base,
        "seen_n": int(a.size),
        "unseen_n": int(b.size),
        "seen_mean": float(a.mean()),
        "unseen_mean": float(b.mean()),
        "difference": difference,
        "pooled_sd": pooled,
        "standardised_difference": float(std_diff),
        "perfectly_separated": bool(separated),
    }


def statistics_table(stats: Sequence[ZoneStatistic]) -> str:
    lines = [
        "| zone | value | coverage | mean looks | status |",
        "|------|------:|---------:|-----------:|--------|",
    ]
    for s in stats:
        if s.reported:
            lines.append(
                f"| {s.zone} | {s.value:.3f} | {s.coverage * 100:.0f}% | "
                f"{s.mean_looks:.1f} | reported |"
            )
        else:
            lines.append(
                f"| {s.zone} | withheld | {s.coverage * 100:.0f}% | "
                f"{s.mean_looks:.1f} | {s.withheld_reason} |"
            )
    return "\n".join(lines)


def summarise(stats: Sequence[ZoneStatistic]) -> Mapping[str, float]:
    reported = [s for s in stats if s.reported]
    return {
        "zones": len(stats),
        "reported": len(reported),
        "withheld": len(stats) - len(reported),
        "withheld_fraction": (len(stats) - len(reported)) / len(stats) if stats else 0.0,
        "mean_coverage": float(np.mean([s.coverage for s in stats])) if stats else 0.0,
    }
