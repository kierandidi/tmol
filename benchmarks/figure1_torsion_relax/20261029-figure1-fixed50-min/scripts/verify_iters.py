import sys, os
sys.path.insert(0, '/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/opt')
import relaxlab as L, torch, benchmark_tmol as bt
from tmol.optimization import CartesianMinimizer, KinematicMinimizer
from tmol.relax import fast_relax
from tmol.relax._fast_relax import DEFAULT_RELAX_SCHEDULE
exec(open('/mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/src/torsion_fastrelax.py').read().split('class FixedStepMinimizer')[1].split('def tmol_fastrelax')[0].join(['class FixedStepMinimizer','']))
dev = torch.device('cuda')
for mode in ('torsion', 'cartesian'):
    pose, sfxn, ff, mm, pal = L.setup(bt.select('5uoi'), dev, 4, mode)
    mf = FixedStepMinimizer(KinematicMinimizer(False) if mode == 'torsion' else CartesianMinimizer(False), 50, mode == 'torsion')
    fast_relax(pose, sfxn, pal, mm, ff, num_repeats=1, schedule=[DEFAULT_RELAX_SCHEDULE[-1]], min_fn=mf, verbose=True)
