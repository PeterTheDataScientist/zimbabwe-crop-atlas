# Phase 1: what the pipeline measured, and the four bugs it found on the way

Measured 5 September 2026 over the Harare and Goromonzi box, the same
0.40 by 0.30 degree area every finding in this project uses. Grid is
EPSG:32736 at 200 m, 169 by 214 pixels. Two seasons: 2024/25, which has no
radar, and 2025/26, which has a full radar record.

Everything below came out of the shipped package, not out of a script written
to produce it. That distinction is the point of this document.

## The headline: findings/01 reproduced independently

Findings/01 was measured in an afternoon with a throwaway script, before any of
this existed. Running the finished pipeline over the same box and the same
season, with entirely different code:

| month    | findings/01, S2 alone | pipeline, S2 alone | findings/01 blind | pipeline blind |
|----------|----------------------:|-------------------:|------------------:|---------------:|
| Nov 2024 |                  4.05 |               4.03 |              0.0% |           0.0% |
| Dec 2024 |                  8.77 |               8.75 |              0.0% |           0.0% |
| Jan 2025 |                  3.08 |               3.08 |              0.2% |           0.2% |
| Feb 2025 |                  1.76 |               1.79 |             16.7% |          16.1% |
| Mar 2025 |                  2.08 |               2.13 |              9.4% |           9.3% |
| Apr 2025 |                  7.31 |               7.30 |              0.0% |           0.0% |

Agreement to within 0.05 clear looks in every month. Two independent
implementations landing on the same numbers is the strongest evidence either
one is right.

## The archive is not uniform, and now the pipeline says so

Scenes held over the box, by season, from the shipped inventory:

| season  | Sentinel-2 | Landsat 8 | Landsat 9 | Sentinel-1 | radar days |
|---------|-----------:|----------:|----------:|-----------:|-----------:|
| 2021/22 |         96 |        46 |        39 |         16 |          4 |
| 2022/23 |         72 |        44 |        44 |      **0** |          0 |
| 2023/24 |         73 |        46 |        46 |      **0** |          0 |
| 2024/25 |         84 |        46 |        46 |      **0** |          0 |
| 2025/26 |         99 |        44 |        44 |     **68** |         17 |

Optical coverage is steady at roughly 80 to 92 acquisition days a season
throughout. Radar vanishes for three seasons and returns for the fourth. That
is the whole architectural argument in one table: a pipeline with a hardcoded
sensor list is wrong for part of this archive, and wrong silently.

## What Landsat buys, measured twice

2024/25, no radar:

| month    | S2 alone | with L8 and L9 | gain  | blind S2 | blind both |
|----------|---------:|---------------:|------:|---------:|-----------:|
| Nov 2024 |     4.03 |           9.45 | +5.42 |     0.0% |       0.0% |
| Dec 2024 |     8.75 |          16.17 | +7.42 |     0.0% |       0.0% |
| Jan 2025 |     3.08 |           6.80 | +3.71 |     0.2% |       0.0% |
| Feb 2025 |     1.79 |           3.72 | +1.93 |    16.1% |       0.0% |
| Mar 2025 |     2.13 |           5.46 | +3.33 |     9.3% |       0.3% |
| Apr 2025 |     7.30 |          11.18 | +3.88 |     0.0% |       0.0% |

2025/26, a wetter January:

| month    | S2 alone | with L8 and L9 | gain  | blind S2 | blind both |
|----------|---------:|---------------:|------:|---------:|-----------:|
| Nov 2025 |     2.94 |           6.36 | +3.42 |     0.0% |       0.0% |
| Dec 2025 |     4.46 |           4.81 | +0.36 |     0.0% |       0.0% |
| Jan 2026 |     1.14 |           4.46 | +3.32 |    32.6% |       6.1% |
| Feb 2026 |     2.38 |           6.43 | +4.04 |     6.5% |       0.2% |
| Mar 2026 |     2.97 |           5.87 | +2.90 |     0.5% |       0.0% |
| Apr 2026 |     6.50 |           7.41 | +0.91 |     0.0% |       0.0% |

The worst month moved. In 2024/25 it was February with 16.1% of the district
unseen by Sentinel-2; in 2025/26 it was January with 32.6%. That matters more
than it looks: a pipeline tuned to fix February would have failed in 2026, and
the reason to solve the problem generally rather than seasonally is that you
cannot know in advance which month will be the bad one.

## What radar buys, in the season that has it

Sentinel-1, 2025/26, against the optical record for the same months:

| month    | radar passes | radar looks | radar blind | optical looks | optical blind |
|----------|-------------:|------------:|------------:|--------------:|--------------:|
| Nov 2025 |            8 |        2.00 |        0.1% |          6.36 |          0.0% |
| Dec 2025 |           12 |        2.99 |        0.1% |          4.81 |          0.0% |
| Jan 2026 |            8 |        2.00 |        0.1% |          4.46 |          6.1% |
| Feb 2026 |           12 |        3.00 |        0.1% |          6.43 |          0.2% |
| Mar 2026 |            8 |        2.00 |        0.1% |          5.87 |          0.0% |
| Apr 2026 |           20 |        4.99 |        0.1% |          7.41 |          0.0% |

Radar coverage is flat. Identically flat: 0.1% unseen in every month of the
season, including the January that took 6.1% of the district away from the
combined optical record. Radar has fewer looks than optical and it does not
matter, because the looks it has are unconditional.

That is the argument for the fourth sensor stated as a measurement rather than
as physics. Optical coverage is weather. Radar coverage is orbital mechanics.

## The season, read off the curve

Per-acquisition district means, smoothed with a Whittaker filter weighted by
observation coverage:

| | 2024/25 | 2025/26 |
|---|---|---|
| observations in the series | 112 | 97 |
| largest gap | 10 days | 20 days |
| start of season | 27 Nov 2024 | 3 Dec 2025 |
| peak | 23 Jan 2025 | 23 Jan 2026 |
| baseline NDVI | 0.313 | 0.335 |
| amplitude over baseline | 0.271 | 0.226 |
| integral | 27.3 | 23.4 |
| confidence | 1.00 | 1.00 |

Both peak on 23 January, which is when Zimbabwean maize tassels. The 2025/26
season started six days later and reached an amplitude 17% lower.

Neither season records an end of season, and that is the correct answer rather
than a gap. The season window closes on 30 April and district NDVI had not yet
fallen back through the 20% threshold by then, so there is no observed end
date. Inventing one from the edge of the window would be fabrication, which is
why the field is empty.

One methodological result is worth separating out. Running the same phenology
on monthly composites instead of on individual acquisitions gives a peak of
27 January 2026 with a confidence of 0.19, against 23 January with a confidence
of 1.00 from the per-acquisition series. Compositing to months before fitting
throws away most of the temporal information that the three-sensor
harmonisation was built to gather. The composites are for maps. The curve
should be fitted on the observations.

## Cross-sensor calibration, fitted over Zimbabwe

Fitted on 2024/25 from coincident acquisitions within 30 hours, Deming
regression, reference Sentinel-2, r-squared floor 0.70:

| sensor | band  | slope | intercept | r²    | OLS slope | pixel pairs |
|--------|-------|------:|----------:|------:|----------:|------------:|
| l8     | red   | 1.112 |   -0.0040 | 0.915 |     1.061 |     334,181 |
| l8     | nir   | 0.986 |   -0.0096 | 0.790 |     0.884 |     334,181 |
| l8     | swir1 | 0.996 |   +0.0208 | 0.891 |     0.942 |     334,181 |
| l9     | red   | 1.204 |   -0.0126 | 0.851 |     1.100 |     200,512 |
| l9     | swir1 | 1.116 |   -0.0140 | 0.784 |     0.985 |     200,512 |

Blue was refused for both sensors and NIR was refused for Landsat 9, all three
below the r-squared floor. Those bands run uncorrected and every composite
built from them says so in its caveats.

Refusing them is the finding, not a gap in it. Blue is where the two
atmospheric corrections disagree most, and on this data it fitted at r-squared
0.60 and returned slopes of 1.51 and 1.82 for two nominally identical
instruments. Those are not physical cross-sensor differences. An adjustment
that injects more error than it removes is worse than none, and worse
precisely because it looks like diligence.

## Calibration is a property of the instruments, not of the season

The 2025/26 season could not fit its own coefficients. It produced 66
coincident scene pairs but only 130,992 overlapping valid pixels against
2024/25's 334,181, because cloud fell differently, and every band failed the
r-squared floor. The season with the best sensor coverage in four years was
therefore about to run completely uncorrected.

That exposed a wrong assumption in the design rather than a problem with the
data. Fitting per season buys locality in space, which matters, and locality in
time, which does not: the difference between OLI and MSI is a property of two
spectral response functions and does not change between one growing season and
the next. So a fitted calibration is now saved and can be carried, band by
band, into a season that cannot fit its own.

Carrying the 2024/25 coefficients into 2025/26, measured on that season's own
coincident pairs:

| band  | residual, uncalibrated | residual, 2024/25 coefficients | change |
|-------|-----------------------:|-------------------------------:|--------|
| nir   |               -0.00795 |                       -0.00073 | 10.9x better |
| swir1 |               +0.02564 |                       +0.00761 | 3.4x better |
| red   |               +0.02024 |                       +0.01343 | 1.5x better |
| blue  |               +0.03792 |                       +0.03792 | unchanged |

Blue is unchanged because blue was refused in both seasons and so was never
corrected in either. That is the guard behaving correctly: it declined to
invent a coefficient, and the uncorrected offset stays visible in the output
instead of being hidden by a bad fit.

Every composite built from carried coefficients records
`calibration_fitted_on` in its provenance and carries a caveat saying part of
its calibration came from another season. The borrowing is defensible. Hiding
it would not be.

## What the phenology figure admits

The per-acquisition figures colour each point by the instrument that made it,
for one reason: if a reader can pick out which sensor produced which point, the
calibration has failed, and the figure says so without needing a statistic.

For 2025/26 the reader can. Both Landsat sensors sit systematically above
Sentinel-2 from February onward, by roughly 0.02 to 0.04 NDVI. The carried
calibration reduced the offset but did not remove it.

Three candidate explanations, none of them yet tested. The residual red offset
of +0.0134 propagates into NDVI and has the right sign. Blue is uncorrected
entirely, which does not enter NDVI but indicates the two atmospheric
corrections are further apart this season. And the 2024/25 coefficients were
fitted on a season whose aerosol conditions may simply differ.

This is written down rather than smoothed over because it is the most useful
open question the project currently has, and because a figure designed to
expose a failure has no value if the failure it exposes goes unmentioned.

## Four bugs, and what each one teaches

Every one of these produced output that looked entirely reasonable. That is
what makes them worth writing down.

**1. A reflectance floor of zero deleted most of the data.**
Surface reflectance cannot be negative, so zero is the obvious floor. It is
also wrong: Sen2Cor over-corrects aerosol over dark vegetation, and on a clear
Sentinel-2 scene over Harare the blue band has a median of -0.0092 with 62% of
clear pixels below zero. Because validity accumulates across bands, a floor of
zero discarded nearly two thirds of every scene where blue was requested.
Sentinel-2 alone dropped from 4.03 clear looks in November to 1.07. Caught by
comparing a pipeline run against the earlier hand measurement.
*The lesson: a physically motivated bound is not the same as an empirically
correct one, and the gap between them is where the data lives.*

**2. Fixing that broke NDVI, in the opposite direction.**
Allowing negative red is right for compositing and wrong for a ratio. Red of
-0.05 with NIR of 0.30 gives an NDVI of 1.40, which the range check then turns
into NaN, and the pixel disappears. February zone statistics collapsed from
16 of 16 reportable to 4 of 16, while the composite behind them reported 0.0%
blind. Two individually correct decisions produced a wrong answer where they
met.
*The lesson: the characteristic failure of a pipeline is not a broken stage, it
is two correct stages with incompatible assumptions at the boundary. The only
thing that catches it is an end-to-end check against a known answer.*

**3. The BOA offset was applied twice.**
Sentinel-2 red came out at -0.0079 where Landsat read +0.0792 over the same
ground on the same day: a 0.087 discrepancy, two orders of magnitude larger
than any real difference between OLI and MSI. The adapter read the processing
baseline and applied the -1000 offset, not knowing that DE Africa's archive had
already applied it and published `sentinel:boa_offset_applied: true` saying so.
The right property was there all along; the code asked the wrong question.
*The lesson: ask the archive what it did, rather than inferring it from the
instrument's processing history. And the cross-sensor comparison is what caught
this, which is an argument for building the comparison before you need it.*

**4. Asking for optical and radar together emptied the season.**
The compositor reduces the observations carrying every requested band, which is
the correct contract. Handing it reflectance and gamma-0 bands together
therefore selected the scenes carrying both, and no instrument carries both.
The 2025/26 run loaded 255 observations, 68 of them radar, and produced six
monthly composites that were 100% blind. Nothing failed. Nothing warned.
*The lesson: a correct component given inputs its contract excludes fails
silently and completely. The fix belongs at the layer that knows both families
exist, not inside the component.*

A fifth, statistical rather than a bug: ordinary least squares was the wrong
estimator. Both sides of a cross-sensor pair are measurements, so noise in the
predictor attenuates the OLS slope toward zero. It showed up as Landsat 9 NIR
fitting at 0.64 while Landsat 8 NIR fitted at 0.86 on the same season, which
two nominally identical instruments cannot do. Deming regression removes the
bias and moves Landsat 8 NIR from 0.884 to 0.986.

## What is not done

The calibration is fitted on one district and one season. Whether it holds
across Zimbabwe's agro-ecological zones is untested and testable.

The r-squared floor is a convention set at 0.70 after seeing where Deming
begins to diverge. It has not been validated against an independent standard.

There is no crop mask. Every district number here is over all land, urban
Harare included, which is why the peak district NDVI is 0.53 rather than the
0.75 a maize field reaches. `esa_worldcereal_maize_main` is in the catalogue
and is the obvious next input.

Phenology runs on the district mean. Per-pixel phenology is what the banded
smoother was written for and has not been run at scale.
