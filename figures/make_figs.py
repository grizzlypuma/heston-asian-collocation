# Figures for the paper (static, PDF + PNG).
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'          # categorical colours 1-3
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.6, 'lines.markersize': 5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(ROOT, 'results'))
OUT = os.path.join(ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
Ks = ['90.0', '100.0', '110.0']
REF = {'90.0': 13.7602944808, '100.0': 7.0325229180, '110.0': 2.9674925504}


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=200); plt.close(fig)


def fig_comparator():
    res = json.load(open('comp_arith.json'))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    spec = [('colloc', 'quintic collocation, CN', S1, 'o', '-'),
            ('colloc_rich', 'quintic collocation, CN + Richardson in time', S3, 's', '-'),
            ('adi', 'MCS-ADI, 2nd order', S2, '^', '--')]
    for m, lab, col, mk, ls in spec:
        rows = sorted([r for r in res if r['cfg']['method'] == m], key=lambda r: r['sec'])
        t = np.array([r['sec'] for r in rows])
        e = np.array([max(abs(r['prices'][k] - REF[k]) for k in Ks) for r in rows])
        ax.plot(t, e, ls, color=col, marker=mk, label=lab, markeredgecolor='white', markeredgewidth=0.8)
    ax.axhline(3e-5, color=INK2, lw=0.8, ls=':')
    ax.text(380, 3.6e-5, 'reference, ±3e-5', color=INK2, fontsize=7.5, ha='right')
    ax.axhline(3e-4, color=MUTED, lw=0.8, ls=':')
    ax.text(380, 3.6e-4, 'geometric CV only, ±3e-4', color=MUTED, fontsize=7.5, ha='right')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('wall-clock time per solve, all three strikes (s)')
    ax.set_ylabel('max error over K = 90, 100, 110')
    ax.set_xlim(0.08, 400)
    ax.legend(loc='lower left', fontsize=7.5)
    save(fig, 'fig_comparator')


def fig_order():
    c = json.load(open('C_orders.json'))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, lab, col, mk in [('0.25_100.0', 'smoothed terminal', S1, 'o'), ('0.0_100.0', 'call payoff (kink at x=0)', S2, 's')]:
        ks = np.array(c[key]['ks']); d = np.abs(np.array(c[key]['diffs']))
        Nx = 17 * ks[1:] + 1   # the difference P(k_{i+1})-P(k_i) is plotted at the finer level
        ax.plot(Nx, d, '-', color=col, marker=mk, label=lab, markeredgecolor='white', markeredgewidth=0.8)
    xr = np.array([180, 1600]); ax.plot(xr, 2e-4 * (xr / 180.0)**-4 * 0.02, color=MUTED, lw=1.0, ls='--', label='reference slope −4')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('number of x-nodes Nx (fixed sinh map)')
    ax.set_ylabel('|P(Nx) − P(previous Nx)|, K = 100')
    ax.legend(loc='upper right', fontsize=7.5)
    save(fig, 'fig_order')


def fig_strikes():
    sol = json.load(open('sol_strikes.json'))['base']['K']
    e8 = json.load(open('e8_strikes.json')); e8 = e8[list(e8)[0]]['K']
    Kv = sorted(float(k) for k in sol)
    d = np.array([(sol[str(k)]['R2'] - e8[str(k)]['E8']) * 1e5 for k in Kv])
    s = np.array([e8[str(k)]['gap_se'] * 1e5 for k in Kv])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhspan(-3, 3, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.errorbar(Kv, d, yerr=2 * s, fmt='o', color=S1, ecolor=S1, elinewidth=1.0, capsize=2.5,
                markeredgecolor='white', markeredgewidth=0.8)
    ax.set_title('shaded: ±3e-5, stated uncertainty of the reference; bars: ±2 s.e. of the Monte Carlo part', fontsize=7.5, loc='left', color=INK2)
    ax.set_xlabel('strike K (S₀ = 100)')
    ax.set_ylabel('collocation − reference (×1e-5)')
    save(fig, 'fig_strikes')


def fig_regimes():
    d = json.load(open('regimes_final.json'))
    names = list(d)
    labels = [n.replace('rho', 'ρ').replace('xi', 'ξ').replace(' (Corsaro)', ' (base)') for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(names) + 1.4))
    for j, (K, col, mk) in enumerate(zip(Ks, [S1, S2, S3], ['o', 's', 'D'])):
        ys = [i + (j - 1) * 0.22 for i in range(len(names))]
        zs = [d[n]['K'][K]['z'] for n in names]
        ax.plot(zs, ys, mk, color=col, markeredgecolor='white', markeredgewidth=0.6, label=f'K = {float(K):.0f}')
    ax.axvspan(-2, 2, color=GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.set_xlim(-4, 4)
    ax.set_xlabel('(collocation − reference) / s.e. of the reference')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=3)
    save(fig, 'fig_regimes')

if __name__ == '__main__':
    import sys
    names = sys.argv[1:] or ['fig_comparator', 'fig_order', 'fig_strikes', 'fig_regimes']
    for f in names:
        globals()[f]()
