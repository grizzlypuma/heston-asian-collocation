"""Pool the Euler runs of 04b over the step sizes (inverse-variance weights) (Table 8).

Writes mc_euler_pooled.json with, per set: raw (no control variate), cv (geometric control
variate) and mlbgap (MLB plus the simulated gap, i.e. the bound as control variate), each as
[estimate, standard error].
"""
import _setup  # noqa: F401
import json
import numpy as np

runs = json.load(open('mc_euler.json'))
out = {}
for name, rr in runs.items():
    rec = {}
    for key, xs, ss in [('raw', [r['raw'] for r in rr], [r['raw_se'] for r in rr]),
                        ('cv', [r['cv'] for r in rr], [r['cv_se'] for r in rr]),
                        ('mlbgap', [r['mlb'] + r['gap'] for r in rr], [r['gap_se'] for r in rr])]:
        x, w = np.array(xs), 1 / np.array(ss) ** 2
        rec[key] = [float((w * x).sum() / w.sum()), float(np.sqrt(1 / w.sum()))]
    out[name] = rec
    steps = [r['nst'] for r in rr]; n = sum(r['n'] for r in rr)
    print(f"{name} ({n / 1e6:.0f}M paths, steps {steps}): "
          + " | ".join(f"{k} {v[0]:.5f}±{v[1]:.5f}" for k, v in rec.items()))
json.dump(out, open('mc_euler_pooled.json', 'w'), indent=1)
