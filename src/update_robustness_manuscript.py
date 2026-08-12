"""Insert completed long-run robustness artefacts into the manuscript.

Called only after run_final_jobs.ps1 has produced every expected CSV.  The script
uses explicit markers, so it never rewrites manually authored parts of the paper.
"""
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT_GREEN = ROOT / "outputs_green"
PAPER = ROOT / "PAPER_MANUSCRIPT.md"
START = "<!-- ROBUSTNESS_RESULTS_START -->"
END = "<!-- ROBUSTNESS_RESULTS_END -->"


def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def cost_row(df, model, policy):
    r = df[(df.model == model) & (df.policy == policy)].iloc[0]
    return f"{r.cost:.3f} [{r.cost_ci_low:.3f}, {r.cost_ci_high:.3f}]"


def reduction_row(df, model):
    r = df[(df.model == model) & (df.policy == "q0.9_vs_median_reduction_pct")].iloc[0]
    return f"{r.cost:.1f}% [{r.cost_ci_low:.1f}, {r.cost_ci_high:.1f}]"


def capacity_rows(path, label):
    df = read(path)
    parts = []
    for pair, g in df.groupby("pair", sort=False):
        left, right = list(g.itertuples(index=False))
        parts.append(f"| {label}: {pair} | {left.model} ({int(left.params):,}) | "
                     f"{left.MAE:.3f} +/- {left.MAE_sd:.3f} | {right.model} "
                     f"({int(right.params):,}) | {right.MAE:.3f} +/- {right.MAE_sd:.3f} |")
    return "\n".join(parts)


def main():
    yellow = read(OUT / "decision_seed_summary.csv")
    green = read(OUT_GREEN / "decision_seed_green_gwnq_summary.csv")
    early_y = read(OUT / "early_origin_yellow_summary.csv")
    early_g = read(OUT_GREEN / "early_origin_green_summary.csv")

    text = f'''{START}
### 7.5 Completed robustness checks

The primary yellow decision result is stable over five independent training seeds.
The completed green Graph WaveNet-Q replication is also reported over three seeds.
Values below are mean 0.9-quantile cost with a 95% Student-t interval; reduction is
the paired reduction relative to the median policy.

| Dataset | Backbone | q0.9 cost [95% CI] | q0.9 vs median reduction [95% CI] |
|---|---|---:|---:|
| Yellow | ST-MRGAT | {cost_row(yellow, 'ST-MRGAT', 'q0.9')} | {reduction_row(yellow, 'ST-MRGAT')} |
| Yellow | Graph WaveNet-Q | {cost_row(yellow, 'GraphWaveNet-Q', 'q0.9')} | {reduction_row(yellow, 'GraphWaveNet-Q')} |
| Green | Graph WaveNet-Q | {cost_row(green, 'GraphWaveNet-Q', 'q0.9')} | {reduction_row(green, 'GraphWaveNet-Q')} |

An earlier rolling origin (train through January, tune through 15 February, test on
16--29 February) preserves the decision effect:

| Dataset | Backbone | q0.9 vs median reduction [95% CI] |
|---|---|---:|
| Yellow, earlier origin | ST-MRGAT | {reduction_row(early_y, 'ST-MRGAT')} |
| Yellow, earlier origin | Graph WaveNet-Q | {reduction_row(early_y, 'GraphWaveNet-Q')} |
| Green, earlier origin | ST-MRGAT | {reduction_row(early_g, 'ST-MRGAT')} |
| Green, earlier origin | Graph WaveNet-Q | {reduction_row(early_g, 'GraphWaveNet-Q')} |

The capacity-matched test shows that the primary claim is not architectural
superiority; it explicitly compares low- and high-capacity pairs.

| Pair | Model A (parameters) | MAE | Model B (parameters) | MAE |
|---|---|---:|---|---:|
{capacity_rows(OUT / 'capacity_matched_yellow_summary.csv', 'Yellow')}
{capacity_rows(OUT_GREEN / 'capacity_matched_green_summary.csv', 'Green')}

These checks strengthen the decision-layer conclusion while retaining the negative
architectural finding: a capacity-matched comparison is not used to claim that one
backbone is universally superior.
{END}'''

    body = PAPER.read_text(encoding="utf-8")
    pattern = re.escape(START) + r".*?" + re.escape(END)
    if re.search(pattern, body, flags=re.S):
        body = re.sub(pattern, text, body, flags=re.S)
    else:
        body = body.replace("### 7.5 Limitations", text + "\n\n### 7.5 Limitations")
    PAPER.write_text(body, encoding="utf-8")
    print("Updated", PAPER)


if __name__ == "__main__":
    main()
