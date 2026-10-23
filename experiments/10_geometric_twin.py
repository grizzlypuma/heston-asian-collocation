"""Geometric twin: validation, comparison with ADI, clustering at v = 0 (Section 5.1, 5.5)."""
import _setup  # noqa: F401
# Geometric twin: (A) validation against the exact price, (B) comparison with ADI, (C) sensitivity to av
import json, os, time, sys
import numpy as np
from geo_colloc import solve_geo
from geo_adi import solve_adi
from geo_chf import price_carr_madan
EX = {K: price_carr_madan(K) for K in (90., 100., 110.)}
OUT = 'geo_runs.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {json.dumps(r['job'], sort_keys=True) for r in res}
jobs = []
for Nz in [21, 31, 41, 61, 81]:
    for K in [90., 100., 110.]:
        jobs.append(dict(part='A', method='colloc', K=K, Nz=Nz, Nv=31, Nt=1600, hw=3.5, v_hi=4.0, av=0.15))
for (Nz, Nv, Nt) in [(21, 21, 800), (31, 21, 800), (41, 31, 800), (61, 31, 800)]:
    jobs.append(dict(part='B', method='colloc', K=100., Nz=Nz, Nv=Nv, Nt=Nt, hw=3.5, v_hi=4.0, av=0.15))
for (Nz, Nv, Nt) in [(81, 41, 800), (161, 61, 800), (241, 81, 800), (321, 101, 800)]:
    for av in [0.4, 0.02]:
        jobs.append(dict(part='B', method='adi', K=100., Nz=Nz, Nv=Nv, Nt=Nt, hw=3.5, v_hi=6.0, av=av))
for av in [0.4, 0.15, 0.05, 0.005]:
    jobs.append(dict(part='C', method='colloc', K=100., Nz=41, Nv=21, Nt=1600, hw=3.5, v_hi=4.0, av=av))
    jobs.append(dict(part='C', method='adi', K=100., Nz=161, Nv=61, Nt=1600, hw=3.5, v_hi=6.0, av=av))
for j in jobs:
    key = json.dumps(j, sort_keys=True)
    if key in done: continue
    t0 = time.time()
    if j['method'] == 'colloc':
        p = solve_geo(K=j['K'], Nz=j['Nz'], Nv=j['Nv'], Nt=j['Nt'], half_width=j['hw'], v_hi=j['v_hi'], av=j['av'])
    else:
        p = solve_adi(K=j['K'], Nz=j['Nz'], Nv=j['Nv'], Nt=j['Nt'], half_width=j['hw'], v_hi=j['v_hi'], av=j['av'])
    rec = dict(job=j, price=p, exact=EX[j['K']], err=p-EX[j['K']], sec=time.time()-t0)
    res.append(rec); json.dump(res, open(OUT, 'w'), indent=1)
    print(j['part'], j['method'], j['K'], j['Nz'], j['Nv'], j['Nt'], j['av'], f"err={rec['err']:+.3e}", f"{rec['sec']:.1f}s", flush=True)
