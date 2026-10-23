"""Pool the reference runs and regress on the step size (Table 2)."""
import _setup  # noqa: F401
import json, numpy as np, glob
runs = []
for f in ['E8.json', 'E8_coarse.json', 'E8_fine.json']:
    try:
        d = json.load(open(f)); runs += d['runs']; mlb = d['mlb']
    except FileNotFoundError:
        pass
Ks = ['90.0', '100.0', '110.0']
out = {}
print("runs:", [(r['nst'], r['n']) for r in runs])
for K in Ks:
    x = np.array([1/r['nst'] for r in runs]); y = np.array([r['E8'][K] for r in runs]); s = np.array([r['se'][K] for r in runs])
    w = 1/s**2; W = w.sum(); xm = (w*x).sum()/W; ym = (w*y).sum()/W
    Sxx = (w*(x-xm)**2).sum(); b = (w*(x-xm)*(y-ym)).sum()/Sxx; sb = np.sqrt(1/Sxx)
    a = ym - b*xm; sa = np.sqrt(1/W + xm**2/Sxx)
    chi2 = (w*(y-ym)**2).sum()
    print(f"K={K}: MLB {mlb[K]:.7f} | reference pooled {ym:.6f}±{np.sqrt(1/W):.6f} (chi2={chi2:.2f}/{len(y)-1}) | slope in Δt {b:+.4f}±{sb:.4f} ({b/sb:+.2f}σ); a(Δt→0)={a:.6f}±{sa:.6f} | gap {ym-mlb[K]:.6f}")
    out[K] = dict(mlb=mlb[K], E8=ym, se=np.sqrt(1/W), chi2=chi2, dof=len(y)-1, slope=b, slope_se=sb, a=a, a_se=sa)
json.dump(out, open('E8_pooled.json', 'w'), indent=1)
