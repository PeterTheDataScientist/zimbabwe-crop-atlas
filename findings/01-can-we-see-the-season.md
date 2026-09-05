# Can we see Zimbabwe's growing season from orbit?

Measured 4 September 2026, before any pipeline was written, because the whole
project rested on the answer. Test area is a 0.40 by 0.30 degree box over
Harare and Goromonzi, prime maize country. All figures are per pixel on a
240 by 300 grid, from Digital Earth Africa holdings.

## The question

Zimbabwe's maize yield is largely settled during tasselling in February, which
is also the wettest month. If cloud hides the crop in February, everything
downstream is guesswork. The plan written before this measurement assumed
Sentinel-1 radar would solve it, because radar sees through cloud.

## Finding 1: the radar answer does not exist here

Sentinel-1 RTC acquisitions over the test box, by growing season:

| season  | s1_rtc | s1_monthly_mosaic | s2_l2a |
|---------|-------:|------------------:|-------:|
| 2021/22 |      8 |                 2 |     72 |
| 2022/23 |      0 |                 0 |     60 |
| 2023/24 |      0 |                 0 |     63 |
| 2024/25 |      0 |                 0 |     72 |

Zero radar acquisitions in the growing season for three consecutive years. A
one degree grid over the whole country in February 2025 shows the same shape:
eastern Zimbabwe has some coverage, the western half and the central maize belt
have none. This is consistent with the loss of Sentinel-1B in December 2021 and
the revised acquisition plan that followed.

**The radar thesis is dead for this application.** Not weakened, dead. No amount
of engineering recovers data that was never acquired.

## Finding 2: the cloud problem is real, and now quantified

Scene level cloud cover from Sentinel-2 metadata, 2024/25 season:

| month    | scenes | under 20% cloud | median cloud |
|----------|-------:|----------------:|-------------:|
| Nov 2024 |     12 |               6 |        19.7% |
| Dec 2024 |     18 |              16 |         0.0% |
| Jan 2025 |     12 |               3 |        40.2% |
| Feb 2025 |     10 |               1 |        51.8% |
| Mar 2025 |     15 |               2 |        42.3% |
| Apr 2025 |     17 |              12 |         6.9% |

Scene level cloud is measured over a whole tile, so it overstates the problem
for any one district. The honest measurement is per pixel, compositing every
pass and masking with the scene classification band:

| month    | mean clear looks | share of district with none | share with 3 or more |
|----------|-----------------:|----------------------------:|---------------------:|
| Nov 2024 |             4.05 |                        0.0% |                85.3% |
| Dec 2024 |             8.77 |                        0.0% |               100.0% |
| Jan 2025 |             3.08 |                        0.2% |                59.4% |
| Feb 2025 |             1.76 |                       16.7% |                23.2% |
| Mar 2025 |             2.08 |                        9.4% |                28.6% |
| Apr 2025 |             7.31 |                        0.0% |               100.0% |

**One sixth of the district is never seen at all during February**, and only
23% of it gets three or more clear looks in the month that decides the yield.
The problem is real. It is simply not the problem radar was going to solve.

## Finding 3: Landsat closes it

Landsat 8 and 9 fly different orbits at different local times from Sentinel-2,
so they fail on different days. Adding both, masked with the Landsat QA_PIXEL
band:

| month    | S2 alone | with Landsat 8 and 9 | gain  | blind on S2 | blind on both |
|----------|---------:|---------------------:|------:|------------:|--------------:|
| Jan 2025 |     3.08 |                 7.50 | +4.42 |        0.2% |          0.0% |
| Feb 2025 |     1.76 |                 4.02 | +2.26 |       16.7% |          0.0% |
| Mar 2025 |     2.08 |                 5.92 | +3.84 |        9.4% |          0.1% |

February clear looks more than double, and the blind fraction goes from one
sixth of the district to nothing measurable.

## What this changes

The technical centrepiece is no longer radar and optical fusion. It is
multi-sensor optical harmonisation across Sentinel-2, Landsat 8 and Landsat 9,
which is the approach NASA's Harmonized Landsat Sentinel product takes, applied
to Zimbabwean smallholder cropland with the gain measured rather than assumed.

That is a better result than the original plan, for three reasons. It is
measured rather than asserted. It works on data that exists today. And the
finding that the obvious answer was unavailable, established in an afternoon
before writing a pipeline, is worth more than a season spent building on it.
