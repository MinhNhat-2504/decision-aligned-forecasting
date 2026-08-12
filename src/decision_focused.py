"""Direct decision-focused Newsvendor baseline.

The baseline trains the same Graph WaveNet representation with a single pinball
head at tau=0.9.  It is deliberately compared with the modular three-quantile
head + CQR pipeline: both consume the same inputs and graph, but only the former
optimises the target operational functional directly.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, SEEDS
from dataset import STData, load_graphs
from operations import newsvendor
from train_eval import DEV, train_neural


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--start-seed', type=int, default=0)
    args = ap.parse_args()
    d, g = STData(), load_graphs(DEV)
    _, _, yr, _ = d.get("test")
    y = yr.numpy()[:, :, 0]
    path = os.path.join(OUT_DIR, "decision_focused.csv")
    rows = (pd.read_csv(path).to_dict('records') if os.path.exists(path) else [])
    done = {int(r['seed']) for r in rows}
    seeds = [s for s in SEEDS[args.start_seed:args.start_seed + args.seeds] if s not in done]
    for seed in seeds:
        m, _ = train_neural("GraphWaveNet-Q", d, g, seed=seed,
                            quantile_grid=[0.9])
        pred = np.maximum(m.pop("_pred")[:, :, 0, 0], 0)
        r = newsvendor(pred, y, .9)
        rows.append(dict(model="GraphWaveNet-direct-q0.9", seed=seed,
                         params=m["params"], forecast_mae=m["MAE"], cost=r['cost']))
        print(f"seed={seed} cost={r['cost']:.4f} MAE={m['MAE']:.3f}", flush=True)
        # Persist after every expensive run; a cancelled later seed cannot erase
        # completed decision-focused evidence.
        pd.DataFrame(rows).to_csv(path, index=False)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__": main()
