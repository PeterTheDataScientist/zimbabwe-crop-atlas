"""Pipeline tests with fake sensors. No network anywhere in this file.

The sensor protocol is narrow enough that a fake is a dozen lines, which is the
point of having made it narrow. These tests can therefore exercise the case
that matters most and that a live test could never produce on demand: a season
where one sensor holds nothing and another cannot be reached at all.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from cropatlas.observation import Grid
from cropatlas.pipeline import inventory, load_stack, run_season
from cropatlas.sensors import SceneRef, SensorUnavailable
from tests.conftest import make_obs


class FakeSensor:
    """A sensor that holds exactly what it was told to hold."""

    def __init__(
        self,
        key: str,
        days: list[int],
        *,
        bands: tuple[str, ...] = ("red", "nir"),
        raise_on_search: bool = False,
        fail_scene: str | None = None,
        offset: float = 0.0,
        base_month: int = 1,
    ) -> None:
        self.key = key
        self.bands = bands
        self._days = days
        self._raise = raise_on_search
        self._fail = fail_scene
        self._offset = offset
        self._month = base_month

    def search(self, bbox, start, end):
        if self._raise:
            raise SensorUnavailable(f"{self.key}: connection reset")
        return [
            SceneRef(
                sensor=self.key,
                scene_id=f"{self.key}-{d:03d}",
                acquired=dt.datetime(2025, self._month, 1) + dt.timedelta(days=d),
                handle={"day": d},
            )
            for d in self._days
        ]

    def load(self, ref, grid, bands=None):
        if self._fail and ref.scene_id == self._fail:
            raise OSError("corrupt COG: unexpected end of file")
        wanted = [b for b in (bands or self.bands) if b in self.bands]
        rng = np.random.default_rng(abs(hash(ref.scene_id)) % 2**32)
        arrays = {
            b: (rng.uniform(0.05, 0.4, grid.shape) + self._offset).astype(np.float32)
            for b in wanted
        }
        valid = np.ones(grid.shape, bool)
        day = int(ref.handle["day"])
        valid[(day * 3) % 30 : (day * 3) % 30 + 6, :] = False
        return make_obs(
            grid, self.key, ref.acquired, bands=arrays, valid=valid,
            scene_id=ref.scene_id,
        )


BOX = (30.95, -18.05, 31.35, -17.75)
START = dt.datetime(2025, 1, 1)
END = dt.datetime(2025, 3, 1)


class TestInventory:
    def test_separates_held_nothing_from_unreachable(self) -> None:
        """The distinction findings/02 exists to enforce.

        A network timeout must never be published as a statement about
        satellite coverage. This is that rule expressed as a test.
        """
        inv = inventory(
            [
                FakeSensor("s2", [1, 8, 15]),
                FakeSensor("l8", []),
                FakeSensor("s1", [], raise_on_search=True),
            ],
            BOX, START, END,
        )
        assert inv.available == ("s2",)
        assert inv.empty == ("l8",)
        assert inv.unreachable == ("s1",)
        assert "held nothing" in inv.by_sensor["l8"].status
        assert "unreachable" in inv.by_sensor["s1"].status

    def test_report_warns_loudly_about_unreachable_sensors(self) -> None:
        inv = inventory(
            [FakeSensor("s2", [1]), FakeSensor("s1", [], raise_on_search=True)],
            BOX, START, END,
        )
        assert "absence here is not evidence of absence" in inv.report()

    def test_an_unexpected_exception_is_unreachable_not_a_crash(self) -> None:
        class Broken:
            key = "x"
            bands = ("red",)

            def search(self, *a):
                raise ValueError("something nobody anticipated")

            def load(self, *a, **k):
                raise NotImplementedError

        inv = inventory([Broken()], BOX, START, END)
        assert inv.unreachable == ("x",)
        assert "ValueError" in inv.by_sensor["x"].error

    def test_a_season_with_nothing_at_all_is_a_valid_answer(self) -> None:
        """2022/23 through 2024/25 for radar. The pipeline reports it rather
        than failing."""
        inv = inventory([FakeSensor("s1", [])], BOX, START, END)
        assert inv.available == ()
        assert inv.all_scenes() == []
        assert "held nothing" in inv.table().replace("held nothing", "held nothing")

    def test_scenes_come_back_in_time_order(self) -> None:
        inv = inventory(
            [FakeSensor("s2", [20, 1, 9]), FakeSensor("l8", [5, 14])],
            BOX, START, END,
        )
        times = [r.acquired for r in inv.all_scenes()]
        assert times == sorted(times)
        assert len(times) == 5


class TestLoading:
    def test_a_corrupt_scene_costs_one_observation_not_the_season(
        self, grid: Grid
    ) -> None:
        sensors = [FakeSensor("s2", [1, 8, 15, 22], fail_scene="s2-008")]
        inv = inventory(sensors, BOX, START, END)
        stack, failures = load_stack(sensors, inv, grid, ["red", "nir"], workers=2)
        assert len(stack) == 3
        assert len(failures) == 1
        assert "corrupt COG" in failures[0][1]

    def test_failures_are_returned_not_swallowed(self, grid: Grid) -> None:
        """A season that lost half its scenes must not look like one that lost
        none. The caller decides whether the loss matters."""
        sensors = [FakeSensor("s2", [1, 8], fail_scene="s2-001")]
        inv = inventory(sensors, BOX, START, END)
        _, failures = load_stack(sensors, inv, grid, ["red", "nir"], workers=2)
        assert len(failures) == 1

    def test_a_sensor_without_the_requested_band_is_skipped_silently(
        self, grid: Grid
    ) -> None:
        """Asking for red edge in a season that had only Landsat is not an
        error, it is a season with no red edge."""
        sensors = [FakeSensor("l8", [1, 8], bands=("red", "nir"))]
        inv = inventory(sensors, BOX, START, END)
        stack, failures = load_stack(sensors, inv, grid, ["rededge1"], workers=2)
        assert len(stack) == 0
        assert failures == []


class TestRunSeason:
    def test_end_to_end_with_three_sensors(self, grid: Grid) -> None:
        sensors = [
            FakeSensor("s2", [2, 12, 22, 32, 42]),
            FakeSensor("l8", [4, 20, 36]),
            FakeSensor("l9", [8, 28, 48]),
        ]
        res = run_season(
            sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir"),
        )
        assert set(res.stack.sensors()) == {"s2", "l8", "l9"}
        assert sorted(res.monthly) == ["2025-01", "2025-02"]
        assert res.monthly["2025-01"].provenance.acquisition_days > 0
        assert "s2" in res.report()

    def test_a_radar_only_season_still_produces_an_inventory(
        self, grid: Grid
    ) -> None:
        """The 2023/24 case inverted: the pipeline must not require optical."""
        sensors = [FakeSensor("s2", []), FakeSensor("l8", [])]
        res = run_season(
            sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir"),
        )
        assert res.inventory.available == ()
        assert res.monthly["2025-01"].blind_fraction == 1.0

    def test_uncalibrated_sensors_are_named_in_the_composite_caveats(
        self, grid: Grid
    ) -> None:
        """One coincident pair is not a calibration. The composite has to say
        which sensors went in uncorrected."""
        sensors = [
            FakeSensor("s2", [2, 20, 40]),
            FakeSensor("l8", [11, 30, 50]),  # never within 30 h of an s2 pass
        ]
        res = run_season(
            sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir"),
        )
        assert not res.harmoniser.is_calibrated("l8")
        caveats = " ".join(res.monthly["2025-01"].provenance.caveats())
        assert "l8" in caveats

    def test_calibration_runs_when_coincident_pairs_exist(self, grid: Grid) -> None:
        sensors = [
            FakeSensor("s2", [2, 12, 22, 32]),
            FakeSensor("l8", [2, 12, 22, 32], offset=0.03),
        ]
        res = run_season(
            sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir"),
        )
        assert res.calibration["attempted"] is True
        assert res.calibration["pair_count"]["l8"] == 4

    def test_mean_index_series_is_monthly_and_ordered(self, grid: Grid) -> None:
        sensors = [FakeSensor("s2", [2, 12, 22, 32, 42, 52])]
        res = run_season(
            sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir"),
        )
        dates, vals, looks = res.mean_index_series()
        assert dates == sorted(dates)
        assert len(dates) == len(vals) == len(looks) == 2
        assert all(np.isfinite(v) for v in vals)

    def test_asking_for_optical_and_radar_together_does_not_empty_the_season(
        self, grid: Grid
    ) -> None:
        """The 2025/26 bug, as a test.

        ``composite`` reduces the observations carrying EVERY requested band,
        which is the right contract. Handing it optical and radar bands
        together therefore selects scenes carrying both, and no instrument
        carries both, so the result is silently empty: 255 observations loaded,
        68 of them radar, and every monthly composite 100% blind. A correct
        component given inputs its contract excludes.
        """
        sensors = [
            FakeSensor("s2", [2, 12, 22, 32]),
            FakeSensor("s1", [5, 17, 29], bands=("vv", "vh")),
        ]
        res = run_season(
            sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir", "vv", "vh"),
        )
        assert res.monthly["2025-01"].blind_fraction < 1.0, (
            "the optical composite must not be emptied by asking for radar too"
        )
        assert res.monthly["2025-01"].provenance.sensors == ("s2",)
        assert res.monthly_radar["2025-01"].provenance.sensors == ("s1",)
        assert res.monthly_radar["2025-01"].blind_fraction < 1.0

    def test_radar_composite_is_absent_when_no_radar_was_held(
        self, grid: Grid
    ) -> None:
        """2022/23 through 2024/25. An empty mapping, not an empty composite
        that a chart would render as a flat line at zero."""
        res = run_season(
            [FakeSensor("s2", [2, 12, 22])], BOX,
            dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir", "vv"),
        )
        assert res.monthly_radar == {}

    def test_radar_composite_is_labelled_as_decibels(self, grid: Grid) -> None:
        res = run_season(
            [
                FakeSensor("s2", [2, 12]),
                FakeSensor("s1", [5, 17], bands=("vv", "vh")),
            ],
            BOX, dt.date(2025, 1, 1), dt.date(2025, 2, 1), grid,
            bands=("red", "nir", "vv", "vh"),
        )
        notes = " ".join(res.monthly_radar["2025-01"].provenance.caveats())
        assert "decibels" in notes

    def test_phenology_survives_a_mixed_stack(self, grid: Grid) -> None:
        """The per-acquisition series reads the stack directly rather than the
        composites, which is why it kept working through the bug above. Worth
        pinning: that isolation is the reason one broken stage did not take the
        season with it."""
        res = run_season(
            [
                FakeSensor("s2", list(range(2, 56, 4))),
                FakeSensor("s1", [5, 17, 29], bands=("vv", "vh")),
            ],
            BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
            bands=("red", "nir", "vv", "vh"),
        )
        times, vals, weights, sensors = res.observation_series()
        assert len(times) >= 10
        assert set(sensors) == {"s2"}, (
            "radar carries no NDVI, so only optical scenes reach the series"
        )
        assert len(sensors) == len(times)
        assert all(np.isfinite(v) for v in vals)
        assert all(w > 0 for w in weights)

    def test_observation_series_drops_a_scene_that_saw_almost_nothing(
        self, grid: Grid
    ) -> None:
        """A district mean from 3% of the pixels is not a small measurement of
        the district, it is a measurement of somewhere else."""
        sensors = [FakeSensor("s2", [2, 12, 22, 32])]
        inv = inventory(sensors, BOX, START, END)
        stack, _ = load_stack(sensors, inv, grid, ["red", "nir"], workers=2)

        sliver = np.zeros(grid.shape, bool)
        sliver[:1, :2] = True
        narrowed = type(stack)(
            grid, tuple(o.masked_to(sliver) for o in stack.observations)
        )
        from dataclasses import replace

        res = replace(
            run_season(
                sensors, BOX, dt.date(2025, 1, 1), dt.date(2025, 3, 1), grid,
                bands=("red", "nir"),
            ),
            stack=narrowed,
        )
        assert res.observation_series()[0] == []
        assert res.observation_series()[3] == []

    def test_progress_callback_is_invoked(self, grid: Grid) -> None:
        seen: list[str] = []
        run_season(
            [FakeSensor("s2", [2, 12])], BOX, dt.date(2025, 1, 1),
            dt.date(2025, 2, 1), grid, bands=("red", "nir"),
            progress=seen.append,
        )
        assert "inventory" in seen and "compositing" in seen
