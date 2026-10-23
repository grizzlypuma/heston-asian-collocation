"""Combine solver and reference values for Figure 3 / Table 10."""
import _setup  # noqa: F401
import json, os
v3 = json.load(open('sol_regimes_vhi3.json')); e8 = json.load(open('e8_regimes.json'))
lt = json.load(open('diag_longT_solver.json')); t5 = json.load(open('diag_T5_x.json'))
e1024 = json.load(open('diag_longT.json'))
e2048 = json.load(open('diag_longT_2048.json')) if os.path.exists('diag_longT_2048.json') else {}
Ks = ['90.0', '100.0', '110.0']
out = {}
order = ['rho=-0.9', 'rho=-0.7', 'rho=-0.3 (Corsaro)', 'rho=0.0', 'rho=0.5', 'rho=0.9', 'xi=0.3', 'xi=0.6', 'xi=1.5', 'T=0.25', 'T=0.5', 'T=3.0', 'T=5.0']
for n in order:
    if n.startswith('T=3') or n.startswith('T=5'):
        T = n[2:5]
        sol = (lt['T=3.0 x∈[-3,12]']['K'] if T == '3.0' else t5['[-4,16] k=32']['K'])
        ref = {}
        for K in Ks:
            vals = [(e1024[T][K]['E8'], e1024[T][K]['se']), (e2048[T][K]['E8'], e2048[T][K]['se'])]
            w = [1/s**2 for _, s in vals]; ref[K] = (sum(wi*v for wi, (v, _) in zip(w, vals))/sum(w), (1/sum(w))**0.5)
        note = f'solver: {"x∈[-3,12], v_hi=3" if T=="3.0" else "x∈[-4,16], v_hi=3, k=32"}; reference: pooled 1024+2048 steps'
    elif n == 'rho=0.9':
        sol = json.load(open('diag_rho09_solver.json'))['k=32']
        mc = json.load(open('diag_rho09_mc.json')); tag = [k for k in e8 if k.startswith(n + '|')][0]
        ref = {}
        for K in Ks:
            vals = [(mc['64'][K]['E8'], mc['64'][K]['se']), (e8[tag]['K'][K]['E8'], e8[tag]['K'][K]['gap_se']), (mc['1024'][K]['E8'], mc['1024'][K]['se'])]
            w = [1/s**2 for _, s in vals]; ref[K] = (sum(wi*v for wi, (v, _) in zip(w, vals))/sum(w), (1/sum(w))**0.5)
        note = 'solver: v_hi=3, k=32; reference: pooled 64+256+1024 steps'
    elif n.startswith('rho=-0.3'):
        sol = {K: v3[n]['K'][K]['R2'] for K in Ks}
        E = json.load(open('E8_pooled.json')); ref = {K: (E[K]['E8'], E[K]['se']) for K in Ks}
        note = 'solver: v_hi=3; reference: pooled 36M paths'
    else:
        sol = {K: v3[n]['K'][K]['R2'] for K in Ks}
        tag = [k for k in e8 if k.startswith(n + '|')][0]
        ref = {K: (e8[tag]['K'][K]['E8'], e8[tag]['K'][K]['gap_se']) for K in Ks}
        note = 'solver: v_hi=3; reference: 256 steps, 4M'
    out[n] = dict(note=note, K={K: dict(solver=sol[K], E8=ref[K][0], se=ref[K][1], z=(sol[K]-ref[K][0])/ref[K][1]) for K in Ks})
json.dump(out, open('regimes_final.json', 'w'), indent=1)
for n, v in out.items():
    print(f"{n:20s}", "  ".join(f"K{float(K):.0f}: {v['K'][K]['solver']:.6f}-{v['K'][K]['E8']:.6f}={v['K'][K]['solver']-v['K'][K]['E8']:+.1e} ({v['K'][K]['z']:+.2f}σ)" for K in Ks))
