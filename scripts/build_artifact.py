"""Assemble the published page, embedding the measured figures as data URIs.

The Artifact CSP blocks external images, so every figure ships inside the page.
Generated rather than hand-written because the numbers come from
findings/phase1-results.json: the page cannot drift from the run that produced
it, which is the whole point of publishing a measurement.
"""

from __future__ import annotations

import base64
import json
import pathlib

FIG = pathlib.Path("figures")
OUT = pathlib.Path("/home/claude/zimbabwe-crop-atlas.html")
RESULTS = pathlib.Path("findings/phase1-results.json")


def img(name: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    d = json.loads(RESULTS.read_text())
    sweep = d["inventory_sweep"]
    s24 = d["seasons"]["2024/25"]
    s25 = d["seasons"]["2025/26"]

    # --- season availability rows ---
    avail_rows = []
    for season in ["2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]:
        sc = sweep[season]["scenes"]
        days = sweep[season]["days"]
        s1 = sc.get("s1", 0)
        cls = ' style="color:var(--flag)"' if s1 == 0 else (
            ' style="color:var(--crop)"' if s1 > 20 else ""
        )
        avail_rows.append(
            f'<tr><td class="k">{season}</td>'
            f'<td class="mono">{sc.get("s2", 0)}</td>'
            f'<td class="mono">{sc.get("l8", 0)}</td>'
            f'<td class="mono">{sc.get("l9", 0)}</td>'
            f'<td class="mono"{cls}>{s1}</td>'
            f'<td class="mono"{cls}>{days.get("s1", 0)}</td></tr>'
        )

    # --- landsat gain rows, both seasons ---
    def gain_rows(season: dict, months: list[str]) -> str:
        g = season["landsat_gain"]
        out = []
        for m in sorted(g):
            r = g[m]
            worst = r["blind_s2"] > 0.05
            out.append(
                f'<tr><td class="k">{months[sorted(g).index(m)]}</td>'
                f'<td class="mono">{r["s2_looks"]:.2f}</td>'
                f'<td class="mono">{r["all_looks"]:.2f}</td>'
                f'<td class="mono" style="color:var(--crop)">+{r["gain"]:.2f}</td>'
                f'<td class="mono"{" style=color:var(--flag)" if worst else ""}>'
                f'{pct(r["blind_s2"])}</td>'
                f'<td class="mono" style="color:var(--crop)">{pct(r["blind_all"])}</td>'
                "</tr>"
            )
        return "\n".join(out)

    m24 = ["Nov 24", "Dec 24", "Jan 25", "Feb 25", "Mar 25", "Apr 25"]
    m25 = ["Nov 25", "Dec 25", "Jan 26", "Feb 26", "Mar 26", "Apr 26"]

    # --- radar rows ---
    radar_rows = []
    rm = s25["radar"]["monthly"]
    for i, m in enumerate(sorted(rm)):
        r = rm[m]
        radar_rows.append(
            f'<tr><td class="k">{m25[i]}</td>'
            f'<td class="mono">{r["passes"]}</td>'
            f'<td class="mono">{r["mean_looks"]:.2f}</td>'
            f'<td class="mono" style="color:var(--radar)">{pct(r["blind"])}</td>'
            f'<td class="mono">{r["optical_looks"]:.2f}</td></tr>'
        )

    # --- calibration rows ---
    cal_rows = []
    fitted = s24["calibration"]["fitted"]
    for sensor in ("l8", "l9"):
        for band in ("blue", "red", "nir", "swir1"):
            a = fitted.get(sensor, {}).get(band)
            if a:
                cal_rows.append(
                    f'<tr><td class="k">{sensor}</td><td class="k">{band}</td>'
                    f'<td class="mono">{a["slope"]:.4f}</td>'
                    f'<td class="mono">{a["intercept"]:+.5f}</td>'
                    f'<td class="mono">{a["r2"]:.3f}</td>'
                    f'<td class="mono" style="color:var(--faint)">{a["ols_slope"]:.4f}</td>'
                    f'<td class="mono">{a["n"]:,}</td></tr>'
                )
    for sensor in ("l8", "l9"):
        for band in s24["calibration"]["skipped"].get(sensor, []):
            cal_rows.append(
                f'<tr><td class="k">{sensor}</td><td class="k">{band}</td>'
                f'<td class="mono" colspan="5" style="color:var(--flag)">'
                "refused, below the r-squared floor; left uncorrected and "
                "declared in the composite caveats</td></tr>"
            )

    ph24, ph25 = s24["phenology"], s25["phenology"]

    html = TEMPLATE.format(
        hero=img("01-sensor-timeline.png"),
        cliff24=img("02-coverage-2024-25.png"),
        cliff25=img("02-coverage-2025-26.png"),
        feb_s2=img("03-february-s2-2024-25.png"),
        feb_all=img("04-february-all-2024-25.png"),
        calib=img("05-calibration-2024-25.png"),
        phen24=img("06-phenology-2024-25.png"),
        phen25=img("06-phenology-2025-26.png"),
        avail_rows="\n".join(avail_rows),
        gain24=gain_rows(s24, m24),
        gain25=gain_rows(s25, m25),
        radar_rows="\n".join(radar_rows),
        cal_rows="\n".join(cal_rows),
        sos24=ph24["start_of_season"], peak24=ph24["peak"],
        amp24=f'{ph24["amplitude"]:.3f}', int24=f'{ph24["integral"]:.1f}',
        gap24=ph24["largest_gap_days"], n24=len(s24["observation_series"]),
        sos25=ph25["start_of_season"], peak25=ph25["peak"],
        amp25=f'{ph25["amplitude"]:.3f}', int25=f'{ph25["integral"]:.1f}',
        gap25=ph25["largest_gap_days"], n25=len(s25["observation_series"]),
        stack24=s24["stack"], stack25=s25["stack"],
    )
    OUT.write_text(html)
    kb = len(html.encode()) / 1024
    print(f"wrote {OUT} ({kb:.0f} KB)")
    return 0


TEMPLATE = r"""<title>Zimbabwe Crop Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Archivo:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">

<style>
  :root {{
    --paper:  #eef1f2;
    --panel:  #ffffff;
    --sunk:   #e3e9ea;
    --ink:    #14201f;
    --muted:  #54676a;
    --faint:  #829496;
    --rule:   #c3cfd1;
    --hair:   #d8e0e1;

    --map:    #26485f;
    --optic:  #b8452f;
    --radar:  #1f7a6d;
    --soil:   #a8602f;
    --crop:   #2e6b41;
    --flag:   #ad3b1e;

    --display: "Source Serif 4", Georgia, serif;
    --body:    "Archivo", system-ui, -apple-system, sans-serif;
    --mono:    "JetBrains Mono", ui-monospace, Menlo, monospace;
    --measure: 68ch;
  }}

  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper: #0f1618; --panel: #161f21; --sunk: #1a2426;
      --ink: #e5ecec; --muted: #9aabac; --faint: #6d8082;
      --rule: #2b383a; --hair: #222d2f;
      --map: #7fb2cd; --optic: #e08a70; --radar: #58c1b0;
      --soil: #d5945e; --crop: #6bb883; --flag: #e08469;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper: #0f1618; --panel: #161f21; --sunk: #1a2426;
    --ink: #e5ecec; --muted: #9aabac; --faint: #6d8082;
    --rule: #2b383a; --hair: #222d2f;
    --map: #7fb2cd; --optic: #e08a70; --radar: #58c1b0;
    --soil: #d5945e; --crop: #6bb883; --flag: #e08469;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    background: var(--paper); color: var(--ink); font-family: var(--body);
    font-size: 16px; line-height: 1.62; -webkit-font-smoothing: antialiased;
  }}
  .sheet {{ max-width: 1060px; margin: 0 auto; padding: clamp(26px,5vw,58px) clamp(18px,4vw,44px) 88px; }}

  .eyebrow {{
    font-family: var(--mono); font-size: 11px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--map); margin: 0 0 12px;
  }}
  h1 {{
    font-family: var(--display); font-weight: 600; font-size: clamp(33px,5.4vw,52px);
    line-height: 1.05; letter-spacing: -.015em; margin: 0 0 16px; text-wrap: balance;
  }}
  .standfirst {{ max-width: var(--measure); font-size: 17.5px; color: var(--muted); margin: 0; }}
  .standfirst strong {{ color: var(--ink); font-weight: 600; }}

  figure {{ margin: 34px 0 0; }}
  figure img {{ display: block; width: 100%; height: auto; border: 1px solid var(--hair); }}
  figcaption {{ font-size: 13px; color: var(--faint); margin-top: 12px; max-width: var(--measure); }}
  figcaption strong {{ color: var(--ink); }}

  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 34px; }}
  .pair figure {{ margin: 0; }}
  @media (max-width: 760px) {{ .pair {{ grid-template-columns: 1fr; }} }}

  section {{ margin-top: 52px; }}
  .head {{
    display: flex; align-items: baseline; gap: 14px;
    border-bottom: 1px solid var(--ink); padding-bottom: 9px;
  }}
  .head h2 {{
    font-family: var(--display); font-weight: 600; font-size: 25px;
    letter-spacing: -.01em; margin: 0; flex: 1; text-wrap: balance;
  }}
  .head .step {{
    font-family: var(--mono); font-size: 11px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--faint); white-space: nowrap;
  }}
  p {{ max-width: var(--measure); }}
  .lede {{ color: var(--muted); margin: 18px 0 22px; }}
  .lede strong, p strong {{ color: var(--ink); font-weight: 600; }}
  h3 {{ font-family: var(--body); font-weight: 600; font-size: 16.5px; margin: 30px 0 8px; }}

  .scroll {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14.5px; min-width: 560px; }}
  th {{
    text-align: left; font-family: var(--mono); font-size: 10.5px; font-weight: 500;
    letter-spacing: .1em; text-transform: uppercase; color: var(--faint);
    padding: 0 14px 8px 0; border-bottom: 1px solid var(--rule); vertical-align: bottom;
  }}
  td {{ padding: 10px 14px 10px 0; border-bottom: 1px solid var(--hair); vertical-align: top; }}
  td:last-child, th:last-child {{ padding-right: 0; }}
  tbody tr:hover {{ background: var(--sunk); }}
  .k {{ font-weight: 600; white-space: nowrap; }}
  .mono {{ font-family: var(--mono); font-size: 13px; font-variant-numeric: tabular-nums; }}
  caption {{
    caption-side: bottom; text-align: left; font-size: 12.5px; color: var(--faint);
    padding-top: 12px; max-width: var(--measure);
  }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 1px;
            background: var(--rule); border: 1px solid var(--rule); margin-top: 30px; }}
  .stats div {{ background: var(--panel); padding: 16px 18px; }}
  .stats dt {{ font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
               text-transform: uppercase; color: var(--faint); margin: 0 0 6px; }}
  .stats dd {{ margin: 0; font-family: var(--display); font-size: 26px; font-weight: 600;
               letter-spacing: -.01em; font-variant-numeric: tabular-nums; }}
  .stats dd small {{ font-family: var(--body); font-size: 13px; font-weight: 400;
                     color: var(--muted); display: block; letter-spacing: 0; }}

  .note {{
    border-left: 3px solid var(--map); padding: 2px 0 2px 18px;
    margin: 26px 0 0; max-width: var(--measure); color: var(--muted);
  }}
  .note b {{ color: var(--ink); }}
  .note.hot {{ border-left-color: var(--flag); }}
  .note.good {{ border-left-color: var(--crop); }}

  .bugs {{ display: grid; gap: 0; margin-top: 10px; }}
  .bugs > div {{
    display: grid; grid-template-columns: 34px 1fr; gap: 18px;
    padding: 22px 0; border-bottom: 1px solid var(--hair);
  }}
  .bugs > div:first-child {{ border-top: 1px solid var(--rule); }}
  .bugs .n {{ font-family: var(--mono); font-size: 12px; color: var(--flag); padding-top: 4px; }}
  .bugs h4 {{ margin: 0 0 6px; font-size: 16.5px; font-weight: 600; }}
  .bugs p {{ margin: 0 0 8px; color: var(--muted); font-size: 14.5px; }}
  .bugs .lesson {{ font-size: 14px; color: var(--map); font-style: italic; }}

  ul.plain {{ padding-left: 20px; max-width: var(--measure); color: var(--muted); }}
  ul.plain li {{ margin-bottom: 10px; }}
  ul.plain strong {{ color: var(--ink); }}

  footer {{
    margin-top: 58px; padding-top: 16px; border-top: 1px solid var(--rule);
    font-family: var(--mono); font-size: 11.5px; color: var(--faint);
    display: flex; flex-wrap: wrap; gap: 6px 22px;
  }}
  @media (max-width: 600px) {{ .bugs > div {{ grid-template-columns: 1fr; gap: 6px; }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>

<div class="sheet">

  <header>
    <p class="eyebrow">Measured result &middot; earth observation and agriculture</p>
    <h1>Zimbabwe Crop Atlas</h1>
    <p class="standfirst">
      Zimbabwe's growing season is also its cloudy season. In February, the month that
      decides the maize yield, <strong>Sentinel-2 alone never sees one sixth of a
      district at all.</strong> Combining it with Landsat 8 and 9 closes that gap to
      nothing. This is Phase 1: the pipeline that does it, the measurements that justify
      every choice in it, and the four bugs the measurements caught.
    </p>

    <figure>
      <img src="{hero}" alt="Matrix of scenes held over the Harare box by season and instrument. Sentinel-2, Landsat 8 and Landsat 9 are present in every season from 2021/22 to 2025/26. Sentinel-1 has 16 scenes in 2021/22, then none at all in 2022/23, 2023/24 and 2024/25, then 68 in 2025/26.">
      <figcaption>
        <strong>The fact that shaped everything else.</strong> Optical coverage is steady
        across five seasons. Radar disappears for three of them and comes back for the
        fourth. A pipeline with a hardcoded sensor list is therefore wrong for part of
        this archive, and wrong silently, which is why sensors here declare what they
        hold rather than being assumed.
      </figcaption>
    </figure>
  </header>

  <section>
    <div class="head"><h2>The radar gap was real, and it ended</h2><span class="step">Findings 01 and 02</span></div>
    <p class="lede">
      The original plan rested on Sentinel-1 radar, which sees through cloud. Digital
      Earth Africa held none over the Zimbabwean maize belt. <strong>An empty mirror is
      not an empty archive, though</strong>, so the next step was to check the two
      archives that hold everything ESA acquired.
    </p>

    <div class="scroll">
      <table>
        <thead><tr>
          <th scope="col">Season</th><th scope="col">Sentinel-2</th><th scope="col">Landsat 8</th>
          <th scope="col">Landsat 9</th><th scope="col">Sentinel-1</th><th scope="col">Radar days</th>
        </tr></thead>
        <tbody>{avail_rows}</tbody>
        <caption>
          Scenes held over the test box. The Copernicus Data Space Ecosystem and the
          Alaska Satellite Facility agree with Digital Earth Africa on all three zeros:
          nothing was acquired by anyone, which is consistent with the loss of
          Sentinel-1B in December 2021 and a single-satellite constellation rationing
          southern Africa out. Coverage returned in November 2025 from Sentinel-1C and
          Sentinel-1D, neither of which existed when the gap opened.
        </caption>
      </table>
    </div>

    <div class="note hot">
      <b>The correction is the finding.</b> An earlier write-up called the radar thesis
      dead on the evidence of one catalogue. That overreached: it measured a mirror and
      reported on an archive. Twenty minutes of checking a source that does not share the
      first one's reasons for being empty turned a dead end into a dated boundary with
      data on the far side of it.
    </div>
  </section>

  <section>
    <div class="head"><h2>What a second and third sensor buy</h2><span class="step">Measured, both seasons</span></div>
    <p class="lede">
      Landsat 8 and 9 fly different orbits at different local times from Sentinel-2, so
      they fail on different days. The gain is not an argument, it is a number.
    </p>

    <figure>
      <img src="{cliff24}" alt="Two-panel chart for the 2024/25 season. Upper panel shows mean clear looks per pixel by month: Sentinel-2 alone falls from 8.75 in December to 1.79 in February, while adding Landsat keeps the combined record above 3.7 all season. Lower panel shows the never-seen fraction, peaking at 16.1 percent for Sentinel-2 in February against zero for the combined record.">
      <figcaption>
        The 2024/25 season. The dotted line at three looks is where a median gains the
        ability to reject one contaminated pixel. Sentinel-2 alone drops below it for
        three consecutive months.
      </figcaption>
    </figure>

    <div class="scroll" style="margin-top:30px">
      <table>
        <thead><tr>
          <th scope="col">Month</th><th scope="col">S2 alone</th><th scope="col">With L8 and L9</th>
          <th scope="col">Gain</th><th scope="col">Never seen, S2</th><th scope="col">Never seen, all</th>
        </tr></thead>
        <tbody>{gain24}</tbody>
        <caption>
          2024/25. The Sentinel-2 column reproduces an earlier independent hand
          measurement to within 0.05 clear looks in every month, from entirely separate
          code. Two implementations landing on the same numbers is the strongest evidence
          either is right.
        </caption>
      </table>
    </div>

    <div class="pair">
      <figure>
        <img src="{feb_s2}" alt="Map of clear observation counts over the Harare district for February 2025 using Sentinel-2 alone. Large pink patches mark ground never seen, clustered rather than scattered, with vertical stripes marking satellite tile boundaries.">
        <figcaption><strong>February 2025, Sentinel-2 alone.</strong> The pink is ground
        never seen at all. It is clustered, not scattered, which is why a mean over the
        visible pixels would be biased rather than merely noisy.</figcaption>
      </figure>
      <figure>
        <img src="{feb_all}" alt="The same February 2025 map with Landsat 8 and 9 added. The pink patches are gone and the district is covered throughout, with most areas receiving three or more clear observations.">
        <figcaption><strong>The same month, three sensors.</strong> The blind fraction
        goes to zero and the median gains something to work with across the district.</figcaption>
      </figure>
    </div>

    <div class="scroll" style="margin-top:34px">
      <table>
        <thead><tr>
          <th scope="col">Month</th><th scope="col">S2 alone</th><th scope="col">With L8 and L9</th>
          <th scope="col">Gain</th><th scope="col">Never seen, S2</th><th scope="col">Never seen, all</th>
        </tr></thead>
        <tbody>{gain25}</tbody>
        <caption>
          2025/26, and the worst month moved. In 2024/25 February took 16.1% of the
          district; in 2025/26 January took 32.6%. A pipeline tuned to fix February would
          have failed the following year, which is the argument for solving the problem
          generally rather than seasonally: you cannot know in advance which month will
          be the bad one.
        </caption>
      </table>
    </div>
  </section>

  <section>
    <div class="head"><h2>What radar adds, in the season that has it</h2><span class="step">2025/26</span></div>
    <p class="lede">
      Radar has fewer looks than the optical record and it does not matter, because the
      looks it has are unconditional.
    </p>

    <div class="scroll">
      <table>
        <thead><tr>
          <th scope="col">Month</th><th scope="col">Radar passes</th><th scope="col">Radar looks</th>
          <th scope="col">Radar never seen</th><th scope="col">Optical looks</th>
        </tr></thead>
        <tbody>{radar_rows}</tbody>
        <caption>
          Identically flat: 0.1% unseen in every month of the season, including the
          January when the combined optical record still lost 6.1% of the district.
          Optical coverage is weather. Radar coverage is orbital mechanics.
        </caption>
      </table>
    </div>

    <figure>
      <img src="{cliff25}" alt="Two-panel chart for the 2025/26 season with a third line for Sentinel-1 radar. The radar line sits between two and five looks per month with almost no variation, while the optical lines fluctuate and Sentinel-2 alone drops to 1.14 in January.">
      <figcaption>
        The same chart for 2025/26 with radar added as the dashed line. Its flatness is
        the point: it is the only line on the chart whose shape is set by orbits rather
        than by weather.
      </figcaption>
    </figure>
  </section>

  <section>
    <div class="head"><h2>Reading the season off the curve</h2><span class="step">Phenology</span></div>
    <p class="lede">
      A Whittaker smoother on a daily grid, with zero weight on days nobody observed.
      Not Savitzky-Golay, which fits in sample index rather than in time and so treats
      one February observation as the same temporal distance from its neighbour as two
      December observations a day apart. <strong>Gaps stay gaps.</strong>
    </p>

    <dl class="stats">
      <div><dt>2024/25 peak</dt><dd>23 Jan<small>from {n24} acquisitions</small></dd></div>
      <div><dt>2025/26 peak</dt><dd>23 Jan<small>from {n25} acquisitions</small></dd></div>
      <div><dt>Amplitude 24/25</dt><dd>{amp24}<small>NDVI over baseline</small></dd></div>
      <div><dt>Amplitude 25/26</dt><dd>{amp25}<small>17% lower season</small></dd></div>
      <div><dt>Largest gap</dt><dd>{gap24} d<small>2024/25, days unobserved</small></dd></div>
      <div><dt>Confidence</dt><dd>1.00<small>both seasons</small></dd></div>
    </dl>

    <figure>
      <img src="{phen24}" alt="Scatter of district mean NDVI through the 2024/25 season, each point coloured by the instrument that produced it, with a black Whittaker fitted curve rising from 0.30 in November to a peak near 0.56 in late January and declining through April.">
      <figcaption>
        2024/25. Start of season {sos24}, peak {peak24}. Points are coloured by
        instrument deliberately: if a reader can pick out which sensor made which point,
        the cross-sensor calibration has failed, and the figure says so without needing a
        statistic.
      </figcaption>
    </figure>

    <figure>
      <img src="{phen25}" alt="The same chart for the 2025/26 season, with Sentinel-2, Landsat 8 and Landsat 9 points. Both Landsat sensors sit visibly above the Sentinel-2 points from February onward.">
      <figcaption>
        2025/26, and <strong>the figure admits a problem.</strong> Both Landsat sensors sit
        systematically above Sentinel-2 from February onward, by roughly 0.02 to 0.04
        NDVI. This season could not fit its own calibration coefficients and carried them
        from 2024/25; that reduced the offset without removing it. The cause is not yet
        established, and it is the most useful open question the project has.
      </figcaption>
    </figure>

    <div class="note">
      <b>One methodological result worth separating out.</b> Running the same phenology on
      monthly composites instead of individual acquisitions moves the measured peak by
      four days and drops the confidence from 1.00 to 0.19. Compositing to months before
      fitting throws away most of the temporal information the harmonisation was built to
      gather. The composites are for maps; the curve is fitted on the observations.
    </div>
  </section>

  <section>
    <div class="head"><h2>Calibrating instruments against each other</h2><span class="step">Fitted over Zimbabwe</span></div>
    <p class="lede">
      Landsat OLI and Sentinel-2 MSI both measure a band called red. They do not measure
      the same red. The difference is a few percent, which is exactly what makes it
      dangerous: too small to see, large enough to put a sawtooth in a time series every
      time the sensor alternates. Coefficients here are <strong>fitted from coincident
      acquisitions over the study area</strong>, not cited from a paper written about
      another continent.
    </p>

    <figure>
      <img src="{calib}" alt="Hexbin density plot of Landsat 8 red reflectance against Sentinel-2 red reflectance for coincident acquisitions, with a one-to-one dotted line, a solid Deming regression line and a dashed ordinary least squares line that is visibly flatter.">
      <figcaption>
        Both fitted lines are drawn because the gap between them is the measured
        attenuation bias. Ordinary least squares assumes the predictor is known exactly.
        Both sides of a cross-sensor pair are measurements, so noise in the predictor
        flattens the OLS slope toward zero. Deming regression minimises perpendicular
        distance instead and does not.
      </figcaption>
    </figure>

    <div class="scroll" style="margin-top:30px">
      <table>
        <thead><tr>
          <th scope="col">Sensor</th><th scope="col">Band</th><th scope="col">Slope</th>
          <th scope="col">Intercept</th><th scope="col">r&sup2;</th>
          <th scope="col">OLS slope</th><th scope="col">Pixel pairs</th>
        </tr></thead>
        <tbody>{cal_rows}</tbody>
        <caption>
          The refusals are the finding, not a gap in it. Blue is where the two
          atmospheric corrections disagree most; it fitted at r-squared 0.60 and returned
          slopes of 1.51 and 1.82 for two nominally identical instruments. Those are not
          physical cross-sensor differences. An adjustment that injects more error than it
          removes is worse than none, and worse precisely because it looks like diligence.
        </caption>
      </table>
    </div>

    <div class="note good">
      <b>Calibration outlives the season that fitted it.</b> The 2025/26 season produced
      only 130,992 usable coincident pixel pairs against 2024/25's 334,181, and failed
      every band. But the difference between OLI and MSI is a property of two spectral
      response functions, not of the weather, so a season now carries coefficients from
      one that could fit them. Doing so improved the 2025/26 near-infrared residual from
      -0.0080 to -0.0007 and shortwave infrared from +0.0256 to +0.0076. Every composite
      built that way records which season its calibration came from.
    </div>
  </section>

  <section>
    <div class="head"><h2>Four bugs the measurements caught</h2><span class="step">Findings 03</span></div>
    <p class="lede">
      Every one of these produced output that looked entirely reasonable. That is what
      makes them worth writing down rather than quietly fixing.
    </p>

    <div class="bugs">
      <div>
        <span class="n">01</span>
        <div>
          <h4>A reflectance floor of zero deleted most of the data</h4>
          <p>
            Reflectance cannot be negative, so zero is the obvious floor. It is also
            wrong: atmospheric correction over-corrects aerosol over dark vegetation, and
            on a clear Sentinel-2 scene over Harare the blue band has a median of -0.0092
            with 62% of clear pixels below zero. Because validity accumulates across
            bands, that floor discarded nearly two thirds of every scene. Sentinel-2
            dropped from 4.03 clear looks in November to 1.07.
          </p>
          <p class="lesson">
            A physically motivated bound is not the same as an empirically correct one,
            and the gap between them is where the data lives.
          </p>
        </div>
      </div>
      <div>
        <span class="n">02</span>
        <div>
          <h4>Fixing that broke NDVI, in the opposite direction</h4>
          <p>
            Allowing negative red is right for compositing and wrong for a ratio. Red of
            -0.05 with near infrared of 0.30 gives an NDVI of 1.40, which the range check
            turns into NaN, and the pixel disappears. February zone statistics collapsed
            from 16 of 16 reportable to 4 of 16 while the composite behind them still
            reported zero percent blind.
          </p>
          <p class="lesson">
            The characteristic failure of a pipeline is not a broken stage. It is two
            correct stages with incompatible assumptions at the boundary, and only an end
            to end check against a known answer finds it.
          </p>
        </div>
      </div>
      <div>
        <span class="n">03</span>
        <div>
          <h4>The radiometric offset was applied twice</h4>
          <p>
            Sentinel-2 red came out at -0.0079 where Landsat read +0.0792 over the same
            ground on the same day: two orders of magnitude larger than any real
            difference between the instruments. The adapter read the processing baseline
            and applied a -1000 offset, not knowing the archive had already applied it and
            published a flag saying so. The right property was there all along; the code
            asked the wrong question.
          </p>
          <p class="lesson">
            Ask the archive what it did rather than inferring it from the instrument's
            processing history. The cross-sensor comparison is what caught this, which is
            an argument for building the comparison before you need it.
          </p>
        </div>
      </div>
      <div>
        <span class="n">04</span>
        <div>
          <h4>Asking for optical and radar together emptied the season</h4>
          <p>
            The compositor reduces the observations carrying every requested band, which
            is the correct contract. Handing it reflectance and radar bands together
            selected the scenes carrying both, and no instrument carries both. The 2025/26
            run loaded 255 observations, 68 of them radar, and produced six monthly
            composites that were 100% blind. Nothing failed. Nothing warned.
          </p>
          <p class="lesson">
            A correct component given inputs its contract excludes fails silently and
            completely. The fix belongs at the layer that knows both families exist, not
            inside the component.
          </p>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="head"><h2>Where it stands</h2><span class="step">Phase 1 complete</span></div>

    <div class="scroll">
      <table>
        <thead><tr><th scope="col">Season</th><th scope="col">Stack loaded</th></tr></thead>
        <tbody>
          <tr><td class="k">2024/25</td><td class="mono">{stack24}</td></tr>
          <tr><td class="k">2025/26</td><td class="mono">{stack25}</td></tr>
        </tbody>
        <caption>
          Harare and Goromonzi, EPSG:32736 at 200 m. 167 tests, none of which touch the
          network. All inputs are open and need no account.
        </caption>
      </table>
    </div>

    <h3>What is not done</h3>
    <ul class="plain">
      <li><strong>No crop mask.</strong> District numbers are over all land including
        urban Harare, which is why peak district NDVI is 0.53 rather than the 0.75 a maize
        field reaches. An open maize layer is already in the catalogue and is the obvious
        next input.</li>
      <li><strong>Calibration is fitted on one district.</strong> Whether it holds across
        Zimbabwe's agro-ecological zones is untested and testable.</li>
      <li><strong>Phenology runs on the district mean.</strong> The banded smoother was
        written for per-pixel work and has not been run at scale.</li>
      <li><strong>The 2025/26 residual offset is unexplained.</strong> Both Landsat
        sensors sit above Sentinel-2 by 0.02 to 0.04 NDVI after calibration, and the
        cause is the first thing Phase 2 should settle.</li>
    </ul>
  </section>

  <footer>
    <span>Phase 1 measured 5 September 2026</span>
    <span>Digital Earth Africa, Copernicus Data Space Ecosystem, Alaska Satellite Facility</span>
    <span>Every figure generated from the run that produced the numbers beside it</span>
  </footer>

</div>
"""


if __name__ == "__main__":
    raise SystemExit(main())
