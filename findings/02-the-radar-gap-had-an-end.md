# The radar gap was real, and it ended

Measured 5 September 2026. Finding 01 declared the radar thesis dead on the
evidence of one catalogue. That was one catalogue too few. This checks the
source archives, and the conclusion changes.

## What finding 01 could and could not support

Finding 01 measured Digital Earth Africa's holdings and found zero Sentinel-1
acquisitions over the Harare test box for three consecutive growing seasons. It
concluded the radar thesis was dead.

That conclusion overreached. DE Africa is a curated regional mirror. An empty
mirror is consistent with an empty archive, but it is equally consistent with a
region that simply fell below somebody's ingest threshold. Finding 01 measured
a mirror and reported on an archive. Correcting that took twenty minutes.

## The source archives, checked independently

Two archives hold everything ESA acquired, and they are independent of each
other and of DE Africa. The Copernicus Data Space Ecosystem is ESA's own. The
Alaska Satellite Facility is NASA's mirror, indexed separately. Counts are over
the same test box finding 01 used.

| season  | CDSE | ASF | DE Africa | verdict |
|---------|-----:|----:|----------:|---------|
| 2021/22 |   16 | 104 |        16 | thin |
| 2022/23 |    0 |   0 |         0 | **nothing acquired** |
| 2023/24 |    0 |   0 |         0 | **nothing acquired** |
| 2024/25 |    0 |   0 |         0 | **nothing acquired** |
| 2025/26 |  134 | 501 |        68 | **full season** |

The counts differ because the archives index different product levels. ASF
counts every granule type, CDSE counts products, DE Africa counts terrain
corrected scenes. What matters is the agreement on zero and the agreement on
return, and on both they agree exactly.

**Finding 01's gap is confirmed and its cause is now known.** It was not a
mirroring failure. For three growing seasons nothing was acquired over central
Zimbabwe by anybody. Sentinel-1B failed in December 2021, and a
single-satellite constellation rations acquisitions. Southern Africa was
rationed out.

**And the rationing ended.** From November 2025 the test box has 17 distinct
acquisition days across the growing season, roughly one pass every ten days,
every month covered:

| month    | Nov 25 | Dec 25 | Jan 26 | Feb 26 | Mar 26 | Apr 26 |
|----------|-------:|-------:|-------:|-------:|-------:|-------:|
| products |     16 |     24 |     16 |     22 |     16 |     40 |

By platform: 118 products from Sentinel-1C, 16 from Sentinel-1D. Neither
satellite existed when the gap opened. The constellation was rebuilt and
coverage came back with it.

## Three checks before believing it

**Is it Harare only?** No. Four boxes across the maize belt, same season:

| box                   | products | acquisition days |
|-----------------------|---------:|-----------------:|
| Harare and Goromonzi  |      134 |               17 |
| Chinhoyi and Makonde  |      162 |               32 |
| Gweru and Midlands    |      150 |               15 |
| Bulawayo, Matabeleland|      250 |               32 |

**Is it still running?** Yes. May to September 2026 has 102 products on 13
days, including six already in the first days of September.

**Is it usable, or only archived?** This is the question that decides
everything, and it is the one most easily skipped. CDSE and ASF hold GRD:
detected amplitude in slant geometry, not corrected for terrain. Turning GRD
into analysis-ready gamma-0 needs orbit files, a DEM, radiometric terrain
correction and speckle handling. That is a week of work and serious compute.

DE Africa's `s1_rtc` is that work already done. And DE Africa has ingested the
new season: 68 terrain corrected scenes on 17 days, matching the CDSE
acquisition days exactly, plus all 12 monthly mosaics for the season, plus 52
scenes into the 2026 dry season.

A global check confirms `s1_rtc` was never discontinued: it has data across
Africa in February 2024, February 2025, February 2026 and July 2026 throughout
the years it held nothing over Zimbabwe. The gap was regional acquisition, not
a dead product.

## What this does to the design

The archive is not uniform, and no amount of wishing makes it uniform:

| season  | Sentinel-2 | Landsat 8 and 9 | Sentinel-1 |
|---------|:----------:|:---------------:|:----------:|
| 2021/22 |     yes    |       yes       | 4 days only |
| 2022/23 |     yes    |       yes       |   none     |
| 2023/24 |     yes    |       yes       |   none     |
| 2024/25 |     yes    |       yes       |   none     |
| 2025/26 |     yes    |       yes       | full season |
| 2026/27 |     yes    |       yes       |  expected  |

A pipeline hardcoded to four sensors returns nothing for 2023. A pipeline
hardcoded to optical throws away the best instrument for 2025/26 and every
season after it. Both are wrong, and the second is the more expensive mistake
because it looks like it is working.

So the sensor set is not a constant in this design, it is an input. Sensors
declare what they hold for a requested season, the compositor takes whatever
answered, and every output carries a provenance record naming the sensors that
contributed and the ones that had nothing. A number computed from three sensors
and a number computed from four are not the same number, and the pipeline is
required to say which one it handed you.

That constraint arrived from a measurement rather than from taste, which is the
only way an architectural constraint is worth anything.

## What finding 01 got right, and the lesson

Finding 01 was right that the historical baseline has to be optical. Seasons
2022/23 through 2024/25 have no radar and never will, so the Landsat
harmonisation that finding 01 measured is still what makes those seasons
readable, and February's blind fraction still goes from 16.7% to zero because
of it.

Finding 01 was wrong to say dead. The honest statement was: absent from the
mirror, cause unknown, source not yet checked. Twenty minutes of checking turned
a dead end into a dated boundary with data on the far side of it.

The lesson is cheap to state and easy to skip: an absence measured in one
catalogue is a fact about that catalogue. Before reporting an absence as a
property of the world, check a source that does not share the first one's
reasons for being empty.
