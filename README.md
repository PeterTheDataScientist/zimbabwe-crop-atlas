# Zimbabwe Crop Atlas

Multi-sensor optical harmonisation and crop phenology over Zimbabwe, built on
free and open satellite archives.

The project reads Sentinel-2, Landsat 8, Landsat 9 and Sentinel-1 from Digital
Earth Africa, puts them on one radiometric basis, composites them per pixel
rather than per scene, and extracts season metrics from the result. Every
number it produces travels with the evidence for it: which instruments
contributed, how many clear looks each pixel got, and how much of the district
was never seen at all.

## Why it is built this way

Three measurements shaped the design, and each is written up in `findings/`.

**The cloud problem is real and it is worse than the metadata suggests.**
Scene-level cloud cover overstates the problem for a single district, so the
honest measure is per pixel. In February 2025 over Harare, Sentinel-2 alone
gave a mean of 1.79 clear looks and left 16.1% of the district never seen in
the month that decides the maize yield.

**The obvious fix was not available.** The original plan was Sentinel-1 radar,
which sees through cloud. Digital Earth Africa holds no Sentinel-1 over the
Zimbabwean maize belt for the 2022/23, 2023/24 or 2024/25 growing seasons. That
was verified against the Copernicus Data Space Ecosystem and the Alaska
Satellite Facility, which agree: nothing was acquired, by anyone, for three
seasons. Sentinel-1B failed in December 2021 and a single-satellite
constellation rationed southern Africa out.

**The gap ended.** From November 2025 the same box has 17 radar acquisition
days across the season, from Sentinel-1C and Sentinel-1D, and Digital Earth
Africa has processed them to analysis-ready gamma-0.

So the archive is not uniform, and a pipeline with a hardcoded sensor list is
wrong for part of it, silently. That single fact drives the architecture:
sensors declare what they hold, the pipeline composites whatever answered, and
every output names the instruments behind it.

## What it measured

Harare and Goromonzi, 0.40 by 0.30 degrees, EPSG:32736 at 200 m.

Adding Landsat 8 and 9 to Sentinel-2, 2024/25 season:

| month    | S2 alone | with L8 and L9 | blind on S2 | blind on all |
|----------|---------:|---------------:|------------:|-------------:|
| Nov 2024 |     4.03 |           9.45 |        0.0% |         0.0% |
| Dec 2024 |     8.75 |          16.17 |        0.0% |         0.0% |
| Jan 2025 |     3.08 |           6.80 |        0.2% |         0.0% |
| Feb 2025 |     1.79 |           3.72 |       16.1% |         0.0% |
| Mar 2025 |     2.13 |           5.46 |        9.3% |         0.3% |
| Apr 2025 |     7.30 |          11.18 |        0.0% |         0.0% |

Radar coverage in the season that has it, 2025/26, is flat at 0.1% unseen in
every month including a January when the combined optical record still lost
6.1% of the district. Optical coverage is weather. Radar coverage is orbital
mechanics.

Season metrics from 112 and 97 acquisitions respectively:

| | 2024/25 | 2025/26 |
|---|---|---|
| start of season | 27 Nov 2024 | 3 Dec 2025 |
| peak | 23 Jan 2025 | 23 Jan 2026 |
| amplitude over baseline | 0.271 | 0.226 |
| confidence | 1.00 | 1.00 |

Full results, including the cross-sensor calibration coefficients fitted over
Zimbabwe, are in `findings/03-what-the-pipeline-measured.md`.

## Design decisions worth knowing about

**Sensors are discovered, not declared.** `Sensor` is a two-method protocol:
what do you hold for this box and period, and turn this scene into an
`Observation`. A sensor holding nothing answers with an empty list, which is a
valid answer. A sensor that cannot be reached raises, which is a different
answer, and the pipeline keeps the two apart: a network timeout must never be
published as a claim about satellite coverage.

**Cross-sensor calibration is measured, not cited.** Coefficients are fitted
from coincident Sentinel-2 and Landsat acquisitions within 30 hours over the
study area itself, rather than imported from published tables fitted on other
continents. The estimator is Deming regression rather than ordinary least
squares, because both sides of the pair are measurements and noise in the
predictor attenuates an OLS slope toward zero.

**A fit the data cannot support is refused.** Bands falling below an r-squared
floor of 0.70 are left uncorrected, and every composite built from them says so
in its caveats. On this data that refuses blue for both Landsat sensors, where
Deming returned slopes of 1.51 and 1.82 for two nominally identical
instruments. An adjustment that injects more error than it removes is worse
than none, and worse precisely because it looks like diligence.

**Calibration outlives the season that fitted it.** The difference between OLI
and MSI is a property of two spectral response functions, not of the weather.
A season that cannot fit its own coefficients carries them from one that could,
band by band, and records `calibration_fitted_on` in provenance.

**Coverage is never a separate panel.** A district statistic computed from
fewer than 60% of its pixels is withheld rather than reported, because February
cloud sits on the high ground and a mean over the visible pixels is biased with
a known sign rather than merely noisy. A blank cell that says why is worth more
than a plausible number nobody can audit.

**Phenology is fitted on acquisitions, not composites.** A Whittaker smoother
on a daily grid with zero weights in the gaps, rather than Savitzky-Golay,
which assumes even spacing that Zimbabwean wet-season observations do not have.
Fitting on monthly composites instead of individual acquisitions moved the
measured peak by four days and dropped the confidence from 1.00 to 0.19.

## Layout

```
src/cropatlas/
  observation.py   the unit every sensor produces and every stage consumes
  masks.py         per-instrument cloud masking as pure array functions
  harmonise.py     cross-sensor calibration, fitted and applied
  composite.py     per-pixel reduction with the clear count attached
  indices.py       NDVI, EVI, SAVI, NDMI, NBR, NDRE, CIRE
  phenology.py     Whittaker smoothing and season metrics with confidence
  district.py      zone roll-up that withholds rather than misleads
  pipeline.py      inventory, load, calibrate, composite, phenology
  figures.py       figures that carry their own caveats
  sensors/         the Sensor protocol and the Digital Earth Africa adapters
  io/              STAC search and windowed COG reads
findings/          measurements, including the ones that refuted earlier plans
scripts/           reproducible runs
tests/             167 tests, no network
```

## Running it

```bash
pip install -e ".[dev,figures]"
pytest                                   # 167 tests, no network needed
python scripts/run_phase1.py --seasons 2024/25 2025/26 --resolution 200
python scripts/make_figures.py
```

The archives are open and need no account. Reading them does need
`AWS_DEFAULT_REGION=af-south-1` alongside `AWS_NO_SIGN_REQUEST=YES`, which
`cropatlas.io.stac.apply_gdal_env()` sets; without the region GDAL fails with a
message that reads like a permissions problem and is not one.

Resolution has real cost. The test box at 10 m is tens of gigabytes for a
season across three sensors; at 200 m it is a few hundred megabytes and every
district statistic is unchanged. Choose fine resolution when field boundaries
are the subject, coarse when district phenology is.

## What is not done

No crop mask, so district numbers are over all land including urban Harare.
`esa_worldcereal_maize_main` is in the catalogue and is the obvious next input.

Calibration is fitted on one district. Whether it holds across Zimbabwe's
agro-ecological zones is untested and testable.

Phenology runs on the district mean. The banded smoother was written for
per-pixel work and has not been run at scale.

The 2025/26 phenology figure shows both Landsat sensors sitting systematically
above Sentinel-2 from February onward, by roughly 0.02 to 0.04 NDVI. The
carried calibration reduced that offset without removing it, and the cause is
not yet established.

## Data

Digital Earth Africa (`s2_l2a`, `ls8_sr`, `ls9_sr`, `s1_rtc`), the Copernicus
Data Space Ecosystem, and the Alaska Satellite Facility. All open, all free, no
account required for the catalogue queries used here.
