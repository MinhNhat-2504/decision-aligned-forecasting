"""Day-block uncertainty intervals for the Amazon parcel-routing holdout."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    x = pd.read_csv(OUT / "amazon_parcel_routing_detail.csv")
    actual = x[x.policy == "Observed driver sequence"]
    nn = x[x.policy == "Nearest-neighbour travel time"]
    a = actual.groupby("date")[["travel_seconds", "route_seconds", "late_packages", "packages"]].sum()
    b = nn.groupby("date")[["travel_seconds", "route_seconds", "late_packages", "packages"]].sum()
    d = a.join(b, lsuffix="_actual", rsuffix="_nn")
    rng = np.random.default_rng(20260729); ids = rng.integers(0, len(d), (5000, len(d)))
    rows=[]
    for label, field in (("travel_time", "travel_seconds"), ("route_duration", "route_seconds")):
        av, bv = d[f"{field}_actual"].to_numpy(), d[f"{field}_nn"].to_numpy()
        draw = 100 * (bv[ids].mean(1) - av[ids].mean(1)) / bv[ids].mean(1)
        rows.append(dict(contrast=f"actual_vs_nearest_neighbour_{label}_reduction_pct",
                         estimate=100*(bv.mean()-av.mean())/bv.mean(),
                         ci95_low=np.quantile(draw,.025), ci95_high=np.quantile(draw,.975),
                         block="calendar_day", n_blocks=len(d), draws=len(ids)))
    late_a, late_b = d.late_packages_actual.to_numpy(), d.late_packages_nn.to_numpy()
    pack = d.packages_actual.to_numpy()
    draw = 100 * (late_b[ids].sum(1) - late_a[ids].sum(1)) / np.maximum(late_b[ids].sum(1), 1)
    rows.append(dict(contrast="actual_vs_nearest_neighbour_late_package_reduction_pct",
                     estimate=100*(late_b.sum()-late_a.sum())/late_b.sum(),
                     ci95_low=np.quantile(draw,.025), ci95_high=np.quantile(draw,.975),
                     block="calendar_day", n_blocks=len(d), draws=len(ids)))
    pd.DataFrame(rows).to_csv(OUT / "amazon_parcel_routing_bootstrap.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v:f"{v:.3f}"))

if __name__ == "__main__": main()
