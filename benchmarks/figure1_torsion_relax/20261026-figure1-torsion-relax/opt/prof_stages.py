import sys, json, time
sys.path.insert(0, __import__('os').path.dirname(__file__))
import relaxlab as L, torch
from tmol.relax import fast_relax
from tmol.relax._fast_relax import DefaultKinematicMinimizer
import benchmark_tmol as bt
mode, ds, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, mode)
kw = {'min_fn': DefaultKinematicMinimizer(cuda_graph=True)} if mode == 'torsion' else {}
torch.manual_seed(20260906)
out, t = L.timed(lambda: fast_relax(pose, sfxn, pal, mm, ff, num_repeats=1, verbose=True, **kw))
print('TOTAL', mode, ds, batch, round(t, 2))
