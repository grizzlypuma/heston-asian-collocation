"""w_vv = 0 against w_v = 0 at v_max = 1.5, 3, 4 (Section 2.3)."""
import _setup  # noqa: F401
# Boundary condition at v_max: w_vv=0 against w_v=0 for v_max = 1.5 / 3 / 4 (k=16, Nv=61, Richardson)
import json, time
from arith_order import solve_multi
out = {}
for bc in ['vv', 'v']:
    for vh in [1.5, 3.0, 4.0]:
        t0 = time.time(); P = [solve_multi(16, Nv=61, Nt=Nt, v_hi=vh, bc_v=bc)[0] for Nt in [1600, 3200, 6400]]
        o = {}
        for K in (90., 100., 110.):
            a, b, c = (x[K] for x in P); r1a = (4*b-a)/3; r1b = (4*c-b)/3; o[str(K)] = (8*r1b-r1a)/7
        out[f'{bc}|{vh}'] = o; json.dump(out, open('bcv.json', 'w'), indent=1)
        print(bc, vh, {K: f'{v:.8f}' for K, v in o.items()}, f'{time.time()-t0:.0f}s', flush=True)
