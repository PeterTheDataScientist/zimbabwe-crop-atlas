"""Sensors declare what they hold. The pipeline never assumes.

This is the architectural answer to findings/02. Over the Zimbabwe maize belt
the available instruments changed twice in five years: radar vanished in 2022
and returned in November 2025. A pipeline with a hardcoded sensor list is
wrong for part of that archive, and it fails silently, producing a plausible
looking number from fewer instruments than the caller believes.

So the sensor set is discovered, not declared. Each sensor answers two
questions: what do you hold for this box and this period, and turn this scene
into an ``Observation``. A sensor that holds nothing answers with an empty
list, which is a legitimate answer that the compositor handles, not an error.

The protocol is deliberately narrow. Two methods, no configuration surface, no
inheritance requirement. This is the same constraint used on the ERP adapters
in another project: a narrow protocol means a new sensor is an afternoon and a
mocked sensor in a test is six lines, and neither can reach into the pipeline
and change how compositing works.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cropatlas.observation import Grid, Observation


@dataclass(frozen=True)
class SceneRef:
    """A scene that exists, before anything has been read.

    Search returns these rather than observations so that the expensive part,
    pulling pixels over the network, happens only for scenes the caller
    actually wants. It also means a coverage audit like findings/01 is a search
    with no reads at all, which is why that measurement took minutes.
    """

    sensor: str
    scene_id: str
    acquired: dt.datetime
    # Whatever the sensor needs to fetch this later: asset hrefs, tile id.
    # Opaque to everything outside the sensor that produced it.
    handle: Mapping[str, object] = field(default_factory=dict)
    # Scene level cloud estimate where the archive supplies one. Advisory only:
    # findings/01 measured that scene level cloud badly overstates the problem
    # for a single district, so this is used for ordering and reporting, never
    # for discarding a scene.
    cloud_cover: float | None = None

    def __str__(self) -> str:
        cc = "" if self.cloud_cover is None else f" cloud={self.cloud_cover:.0f}%"
        return f"{self.sensor}:{self.scene_id}@{self.acquired:%Y-%m-%d}{cc}"


@runtime_checkable
class Sensor(Protocol):
    """What the pipeline needs from an instrument. Nothing more."""

    @property
    def key(self) -> str:
        """Short stable identifier, used in provenance. e.g. 's2', 'l8'."""

    @property
    def bands(self) -> tuple[str, ...]:
        """Canonical band names this sensor can supply."""

    def search(
        self,
        bbox: tuple[float, float, float, float],
        start: dt.datetime,
        end: dt.datetime,
    ) -> Sequence[SceneRef]:
        """Scenes held for this box and period. Empty is a valid answer."""

    def load(self, ref: SceneRef, grid: Grid) -> Observation:
        """Read one scene onto the requested grid.

        The sensor owns reprojection, resampling, scaling, cloud masking and
        band renaming. By the time an ``Observation`` leaves this method it
        carries canonical band names, surface reflectance in [0, 1] or gamma-0
        in decibels, and a validity mask with cloud already removed.
        """


class SensorUnavailable(RuntimeError):
    """A sensor could not answer, as distinct from answering with nothing.

    A network failure and an empty archive must not look the same. Finding 02
    exists because an empty answer was read as a fact about the world when it
    was a fact about one catalogue. This exception keeps the two apart so the
    pipeline can report 'radar held nothing' and 'radar could not be reached'
    as the different statements they are.
    """
