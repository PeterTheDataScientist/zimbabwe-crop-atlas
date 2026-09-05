"""Figures that make the measurements legible without the reader doing work.

Design rule for everything here: the figure has to carry its own caveat. A map
of February NDVI that renders unseen ground in the same visual language as
observed ground is a lie told in a colourmap, and it is the single most common
way products like this mislead. So coverage is never a separate panel. It is
drawn into the thing it qualifies, or the thing it qualifies is not drawn.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

import numpy as np

# Chosen for the two constraints that actually bind: legible when printed in
# greyscale, and safe for the most common colour vision deficiencies. That
# rules out the red-green ramps these maps are usually drawn with, which is
# unfortunate given the subject and not negotiable.
SENSOR_COLOURS = {
    "s2": "#1b6ca8",
    "l8": "#e07b39",
    "l9": "#8b5fbf",
    "s1": "#2a9d5c",
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
WARN = "#c1442e"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def coverage_cliff(
    months: Sequence[str],
    s2_looks: Sequence[float],
    all_looks: Sequence[float],
    s2_blind: Sequence[float],
    all_blind: Sequence[float],
    radar_looks: Sequence[float] | None = None,
    title: str = "",
    path: str = "coverage-cliff.png",
) -> str:
    """The central figure: what each sensor set can actually see, month by month.

    Two panels sharing an x axis, because the two questions are different.
    How many looks did a pixel get, and what share of the district got none at
    all. The second is the one that decides whether a map is publishable and it
    is almost never plotted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    x = np.arange(len(months))

    ax1.plot(x, s2_looks, "o-", color=SENSOR_COLOURS["s2"], lw=2,
             label="Sentinel-2 alone", zorder=3)
    ax1.plot(x, all_looks, "o-", color=SENSOR_COLOURS["l8"], lw=2,
             label="with Landsat 8 and 9", zorder=3)
    if radar_looks is not None:
        ax1.plot(x, radar_looks, "o--", color=SENSOR_COLOURS["s1"], lw=2,
                 label="Sentinel-1 radar", zorder=3)
    ax1.fill_between(x, s2_looks, all_looks, color=SENSOR_COLOURS["l8"], alpha=0.12)
    ax1.axhline(3, color=MUTED, ls=":", lw=1)
    ax1.text(
        len(months) - 0.05, 3.15, "three looks: enough for a median to reject one bad pixel",
        ha="right", va="bottom", fontsize=8, color=MUTED,
    )
    ax1.set_ylabel("mean clear looks per pixel", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax1)
    if title:
        ax1.set_title(title, fontsize=12, color=INK, loc="left", pad=12)

    width = 0.38
    ax2.bar(x - width / 2, np.array(s2_blind) * 100, width,
            color=SENSOR_COLOURS["s2"], label="Sentinel-2 alone")
    ax2.bar(x + width / 2, np.array(all_blind) * 100, width,
            color=SENSOR_COLOURS["l8"], label="with Landsat")
    worst = int(np.argmax(s2_blind))
    if s2_blind[worst] > 0.05:
        ax2.annotate(
            f"{s2_blind[worst] * 100:.0f}% of the district never seen",
            xy=(worst - width / 2, s2_blind[worst] * 100),
            xytext=(worst - 1.6, s2_blind[worst] * 100 + 4),
            fontsize=8.5, color=WARN,
            arrowprops={"arrowstyle": "->", "color": WARN, "lw": 1},
        )
    ax2.set_ylabel("never seen (%)", fontsize=10, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, fontsize=9)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def clear_count_map(
    counts: np.ndarray,
    title: str,
    subtitle: str = "",
    path: str = "clear-count.png",
) -> str:
    """Where the district was seen, and where it was not.

    Pixels never seen are drawn in a colour that is not on the ramp, so they
    read as absent rather than as a low value. That distinction is the entire
    point: zero looks is not a small number of looks, it is no measurement, and
    a continuous ramp starting at zero invites the eye to interpolate across it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    levels = [1, 2, 3, 5, 8, 13, 21]
    shades = ["#dbe7f0", "#a9c8e0", "#6ba3cc", "#3a7fb5", "#1b6ca8", "#0d4a75"]
    cmap = ListedColormap(shades)
    cmap.set_under("#f0d9d4")  # never seen: off the ramp entirely
    norm = BoundaryNorm(levels, cmap.N)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(counts, cmap=cmap, norm=norm, interpolation="nearest")
    blind = float((counts == 0).mean())

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    if subtitle:
        ax.text(0, -0.045, subtitle, transform=ax.transAxes,
                fontsize=9, color=MUTED, va="top")

    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, extend="min")
    cbar.set_label("clear observations", fontsize=9, color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.outline.set_visible(False)

    fig.text(
        0.02, 0.02,
        f"pink: never seen this month ({blind * 100:.1f}% of the district)",
        fontsize=8.5, color=WARN,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def phenology_curve(
    times: Sequence[dt.datetime],
    values: Sequence[float],
    sensors: Sequence[str],
    smoothed_days: np.ndarray | None = None,
    smoothed_values: np.ndarray | None = None,
    season_start: dt.date | None = None,
    marks: Mapping[str, dt.date] | None = None,
    title: str = "",
    path: str = "phenology.png",
) -> str:
    """The season, with every observation shown as the thing it was.

    Points are coloured by the instrument that made them, because the whole
    argument for harmonisation is that a reader should not be able to pick out
    which sensor produced which point. If the Landsat points sit visibly above
    or below the Sentinel-2 points, the calibration failed, and the figure says
    so without needing a statistic.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5))

    by_sensor: dict[str, tuple[list, list]] = {}
    for t, v, s in zip(times, values, sensors, strict=True):
        xs, ys = by_sensor.setdefault(s, ([], []))
        xs.append(t)
        ys.append(v)
    for sensor, (xs, ys) in sorted(by_sensor.items()):
        ax.plot(
            xs, ys, "o", ms=4.5, alpha=0.75,
            color=SENSOR_COLOURS.get(sensor, MUTED),
            label=f"{sensor} ({len(xs)})", zorder=2,
        )

    if smoothed_days is not None and smoothed_values is not None and season_start:
        dates = [season_start + dt.timedelta(days=int(d)) for d in smoothed_days]
        ax.plot(dates, smoothed_values, "-", color=INK, lw=2,
                label="Whittaker fit", zorder=3)

    if marks:
        for name, when in marks.items():
            if when is None:
                continue
            ax.axvline(when, color=MUTED, ls="--", lw=1, zorder=1)
            ax.text(
                when, ax.get_ylim()[1], f" {name}",
                rotation=90, va="top", ha="left", fontsize=8, color=MUTED,
            )

    ax.set_ylabel("NDVI, district mean", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="lower center")
    _style(ax)
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def sensor_timeline(
    seasons: Sequence[str],
    counts: Mapping[str, Sequence[int]],
    title: str = "",
    path: str = "sensor-timeline.png",
) -> str:
    """Which instruments existed over this ground, season by season.

    The findings/02 story as one picture: three seasons with no radar at all,
    then a full radar season. This is the figure that justifies the whole
    architecture, so it is worth drawing plainly and letting the gap speak.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["s2", "l8", "l9", "s1"]
    labels = {
        "s2": "Sentinel-2",
        "l8": "Landsat 8",
        "l9": "Landsat 9",
        "s1": "Sentinel-1 radar",
    }
    fig, ax = plt.subplots(figsize=(9, 3.6))

    for row, key in enumerate(order):
        vals = counts.get(key, [0] * len(seasons))
        for col, n in enumerate(vals):
            if n > 0:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.42, row - 0.36), 0.84, 0.72,
                        facecolor=SENSOR_COLOURS[key], alpha=0.85, lw=0,
                    )
                )
                ax.text(col, row, str(n), ha="center", va="center",
                        fontsize=9, color="white", weight="bold")
            else:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.42, row - 0.36), 0.84, 0.72,
                        facecolor="#f4f4f4", edgecolor=GRID, lw=0.8,
                    )
                )
                ax.text(col, row, "none", ha="center", va="center",
                        fontsize=8, color=WARN)

    ax.set_xlim(-0.6, len(seasons) - 0.4)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[k] for k in order], fontsize=9.5)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def calibration_scatter(
    ref_values: np.ndarray,
    other_values: np.ndarray,
    slope: float,
    intercept: float,
    ols_slope: float | None,
    band: str,
    other_sensor: str,
    r2: float,
    path: str = "calibration.png",
) -> str:
    """What the cross-sensor fit is actually fitting.

    Both fitted lines are drawn where an OLS slope is available, because the
    gap between them is the measured attenuation bias and it is easier to see
    than to explain.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(ref_values.size, 40_000)
    idx = np.random.default_rng(0).choice(ref_values.size, n, replace=False)
    x = other_values[idx]
    y = ref_values[idx]

    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.hexbin(x, y, gridsize=70, cmap="Blues", mincnt=1, linewidths=0)

    lo = float(np.nanpercentile(np.concatenate([x, y]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([x, y]), 99.5))
    line = np.linspace(lo, hi, 50)
    ax.plot(line, line, ":", color=MUTED, lw=1.2, label="1:1")
    ax.plot(line, slope * line + intercept, "-", color=WARN, lw=2,
            label=f"Deming  x{slope:.3f} {intercept:+.4f}")
    if ols_slope is not None:
        ax.plot(line, ols_slope * line + (y.mean() - ols_slope * x.mean()),
                "--", color=INK, lw=1.5, label=f"OLS  x{ols_slope:.3f}")

    ax.set_xlabel(f"{other_sensor} {band} reflectance", fontsize=10, color=INK)
    ax.set_ylabel(f"Sentinel-2 {band} reflectance", fontsize=10, color=INK)
    ax.set_title(
        f"Coincident acquisitions, {band}: {other_sensor} against Sentinel-2\n"
        f"r-squared {r2:.3f} on {ref_values.size:,} pixel pairs",
        fontsize=11, color=INK, loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path
