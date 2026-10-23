"""Common setup for experiment scripts: put src/ on the path and run inside results/."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
os.chdir(os.path.join(ROOT, 'results'))
