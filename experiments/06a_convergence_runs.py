"""Time and space convergence runs (Section 5.3). Args: spec name, one of C1, C2, C4.
C1: time convergence at k=16; C2: x-levels k=4..47 plus Nv checks; C4: k=64, 94."""
import _setup  # noqa: F401
import json, os, sys
from arith_order import run_jobs

spec_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'specs')
spec = json.load(open(os.path.join(spec_dir, sys.argv[1] + '_spec.json')))
run_jobs(spec['jobs'], spec['out'])
