"""Accuracy against time, collocation vs MCS-ADI, arithmetic problem (Figure 4, Table 11)."""
import _setup  # noqa: F401
# Accuracy against computing time on the arithmetic problem (Corsaro set); all strikes from one run.
# Reference value: converged collocation (k=94, Nv correction, Richardson), consistent with the MC reference within ±3e-5.
import json, time, os
from arith_order import solve_multi
from arith_adi_multi import solve_arith_adi
Ks = (90.0, 100.0, 110.0)
REF = {90.0: 13.7602944808, 100.0: 7.0325229180, 110.0: 2.9674925504}
OUT = 'comp_arith.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {json.dumps(r['cfg'], sort_keys=True) for r in res}
def save(rec):
    res.append(rec); json.dump(res, open(OUT, 'w'), indent=1)
    e = {k: rec['prices'][str(k)] - REF[k] for k in Ks}
    print(rec['cfg'], f"t={rec['sec']:.2f}s", {k: f"{v:+.2e}" for k, v in e.items()}, flush=True)
cfgs = []
for (k, Nv, Nt) in [(2,21,200),(3,21,400),(4,31,400),(6,31,800),(8,41,1600),(11,41,3200),(16,61,6400),(23,61,12800)]:
    cfgs.append(dict(method='colloc', k=k, Nv=Nv, Nt=[Nt]))
for (k, Nv, Nts) in [(4,31,[200,400,800]),(6,31,[400,800,1600]),(8,41,[800,1600,3200]),(11,41,[800,1600,3200]),(16,61,[1600,3200,6400]),(23,61,[1600,3200,6400])]:
    cfgs.append(dict(method='colloc_rich', k=k, Nv=Nv, Nt=Nts))
for (Nx, Nv, Nt) in [(81,31,400),(121,41,400),(161,41,800),(241,61,1600),(321,61,1600),(481,81,1600),(641,101,1600),(961,121,1600)]:
    cfgs.append(dict(method='adi', Nx=Nx, Nv=Nv, Nt=Nt))
for cfg in cfgs:
    key = json.dumps(cfg, sort_keys=True)
    if key in done: continue
    t0 = time.time()
    if cfg['method'] == 'adi':
        p = solve_arith_adi(Nx=cfg['Nx'], Nv=cfg['Nv'], Nt=cfg['Nt'], Ks=Ks)
    else:
        P = []
        for Nt in cfg['Nt']:
            P.append(solve_multi(cfg['k'], Nv=cfg['Nv'], Nt=Nt, Ks=Ks)[0])
        if len(P) == 1:
            p = P[0]
        else:
            p = {}
            for K in Ks:
                a, b, c = (x[K] for x in P)
                r1a = (4*b - a)/3; r1b = (4*c - b)/3
                p[K] = (8*r1b - r1a)/7
    save(dict(cfg=cfg, sec=time.time()-t0, prices={str(k): v for k, v in p.items()}, Nx=(17*cfg['k']+1 if 'k' in cfg else cfg['Nx'])))
