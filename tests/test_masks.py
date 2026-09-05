from __future__ import annotations

import numpy as np
import pytest

from cropatlas import masks


class TestSentinel2SCL:
    def test_keeps_land_classes_and_drops_cloud(self) -> None:
        scl = np.array(
            [
                [masks.SCL_VEGETATION, masks.SCL_BARE_SOIL, masks.SCL_WATER],
                [masks.SCL_CLOUD_HIGH, masks.SCL_CLOUD_SHADOW, masks.SCL_THIN_CIRRUS],
                [masks.SCL_NO_DATA, masks.SCL_SATURATED, masks.SCL_UNCLASSIFIED],
            ],
            dtype=np.uint8,
        )
        clear = masks.sentinel2_clear(scl)
        assert clear[0].all(), "vegetation, soil and water are observations"
        assert not clear[1].any(), "cloud, shadow and cirrus are not"
        assert not clear[2, 0] and not clear[2, 1]
        assert clear[2, 2], "unclassified kept by default over fragmented cropland"

    def test_strict_set_drops_unclassified(self) -> None:
        scl = np.full((4, 4), masks.SCL_UNCLASSIFIED, dtype=np.uint8)
        assert masks.sentinel2_clear(scl).all()
        assert not masks.sentinel2_clear(scl, usable=masks.SCL_USABLE_STRICT).any()

    def test_dilation_eats_the_ring_around_a_cloud(self) -> None:
        """SCL under-calls cloud edges; the buffer has to actually reach them."""
        scl = np.full((9, 9), masks.SCL_VEGETATION, dtype=np.uint8)
        scl[4, 4] = masks.SCL_CLOUD_HIGH
        assert masks.sentinel2_clear(scl).sum() == 80
        one = masks.sentinel2_clear(scl, dilate_cloud=1)
        assert one.sum() == 81 - 5, "centre plus its four neighbours"
        two = masks.sentinel2_clear(scl, dilate_cloud=2)
        assert two.sum() == 81 - 13, "the diamond of radius two"

    def test_dilation_does_not_grow_land_classes(self) -> None:
        scl = np.full((7, 7), masks.SCL_VEGETATION, dtype=np.uint8)
        assert masks.sentinel2_clear(scl, dilate_cloud=3).all()

    def test_rejects_a_cube(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            masks.sentinel2_clear(np.zeros((2, 4, 4), np.uint8))

    def test_class_shares_sum_to_one(self) -> None:
        scl = np.array([[4, 4, 8, 9]], dtype=np.uint8)
        shares = masks.sentinel2_class_shares(scl)
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["vegetation"] == pytest.approx(0.5)


class TestLandsatQA:
    @staticmethod
    def qa(*bits: int) -> np.ndarray:
        v = 0
        for b in bits:
            v |= 1 << b
        return np.full((3, 3), v, dtype=np.uint16)

    def test_clear_bit_alone_is_not_enough(self) -> None:
        """The correction to findings/01.

        Bit 6 means 'cloud and dilated cloud not set'. It says nothing about
        shadow. A shadowed crop pixel reads as low NDVI and is
        indistinguishable from a failed crop, so letting it through is worse
        than losing the pixel.
        """
        shadowed = self.qa(masks.QA_CLEAR, masks.QA_CLOUD_SHADOW)
        assert not masks.landsat_clear(shadowed).any()
        assert masks.landsat_clear(shadowed, exclude_shadow=False).all()

    def test_cirrus_excluded_by_default(self) -> None:
        cirrus = self.qa(masks.QA_CLEAR, masks.QA_CIRRUS)
        assert not masks.landsat_clear(cirrus).any()
        assert masks.landsat_clear(cirrus, exclude_cirrus=False).all()

    def test_fill_is_never_clear(self) -> None:
        assert not masks.landsat_clear(self.qa(masks.QA_CLEAR, masks.QA_FILL)).any()

    def test_a_genuinely_clear_pixel_passes(self) -> None:
        assert masks.landsat_clear(self.qa(masks.QA_CLEAR)).all()

    def test_water_does_not_disqualify(self) -> None:
        """A flooded field is an observation of a flooded field."""
        assert masks.landsat_clear(self.qa(masks.QA_CLEAR, masks.QA_WATER)).all()

    def test_cloud_without_clear_bit_is_dropped(self) -> None:
        assert not masks.landsat_clear(self.qa(masks.QA_CLOUD)).any()

    def test_bit_shares_reports_each_flag(self) -> None:
        qa = np.array([[1 << masks.QA_CLEAR, 1 << masks.QA_CLOUD]], dtype=np.uint16)
        shares = masks.landsat_bit_shares(qa)
        assert shares["clear"] == pytest.approx(0.5)
        assert shares["cloud"] == pytest.approx(0.5)


class TestValidRange:
    def test_allows_slightly_above_one(self) -> None:
        """Atmospheric correction legitimately exceeds 1.0 over bright ground.

        Clipping to 1.0 fabricates data; the cut is where correction failures
        begin, not where physics says reflectance stops.
        """
        arr = np.array([[-0.5, 0.0, 0.5, 1.05, 1.7, np.nan]], dtype=np.float32)
        keep = masks.valid_range(arr)
        assert list(keep[0]) == [False, True, True, True, False, False]

    def test_keeps_slightly_negative_blue(self) -> None:
        """The regression test for a bug that silently destroyed most of the data.

        Sen2Cor over-corrects aerosol over dark vegetation, so retrieved blue
        reflectance on genuinely clear pixels has a median near -0.009 and runs
        to -0.05. A floor of 0.0 looks obviously right and throws away 62% of
        the clear pixels in a scene, and because validity accumulates across
        bands it destroys the whole observation. Measured on Sentinel-2 over
        Harare, 11 November 2024.
        """
        realistic_blue = np.array(
            [[-0.0481, -0.0092, 0.0, 0.02, 0.27]], dtype=np.float32
        )
        assert masks.valid_range(realistic_blue).all(), (
            "the blue band of a clear scene must survive the range check"
        )
        assert masks.REFLECTANCE_FLOOR < -0.05

    def test_still_rejects_a_real_correction_failure(self) -> None:
        arr = np.array([[-0.4, -0.15, 2.5]], dtype=np.float32)
        assert not masks.valid_range(arr).any()


class TestDilationDistance:
    def test_buffer_is_a_distance_not_a_pixel_count(self) -> None:
        """The bug: '2 pixels' means 40 m at 20 m resolution and 400 m at
        200 m, so the same code produced a different cloud mask at every
        resolution and coverage statistics stopped being comparable."""
        assert masks.dilation_pixels(40.0, 20.0) == 2
        assert masks.dilation_pixels(40.0, 10.0) == 4
        assert masks.dilation_pixels(40.0, 30.0) == 1

    def test_a_sub_pixel_buffer_becomes_no_dilation(self) -> None:
        """You cannot buffer by less than you can resolve, and pretending
        otherwise is where the resolution dependence came from."""
        assert masks.dilation_pixels(40.0, 200.0) == 0
        assert masks.dilation_pixels(40.0, 100.0) == 0

    def test_rejects_a_nonsense_resolution(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            masks.dilation_pixels(40.0, 0.0)

    def test_the_same_buffer_masks_the_same_ground_at_two_resolutions(self) -> None:
        """The property the fix exists to guarantee, stated directly.

        A 200 m buffer around one cloudy pixel should cover about the same
        ground whether the grid is 20 m or 50 m. Expressed as a pixel count it
        did not; expressed as a distance it does.
        """
        fine = masks.dilation_pixels(200.0, 20.0) * 20.0
        coarse = masks.dilation_pixels(200.0, 50.0) * 50.0
        assert fine == pytest.approx(coarse, abs=25.0)
