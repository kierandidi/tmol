"""A/B timing of a full one-repeat fast_relax; prints one JSON line. Env OVERLAY picks the tmol tree."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import relaxlab as L, torch, tmol
from tmol.relax import fast_relax
from tmol.relax._fast_relax import DefaultKinematicMinimizer
import benchmark_tmol as bt
mode, ds, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]); extra = sys.argv[4] if len(sys.argv) > 4 else 'graph'
dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, mode)
kw = {}
if mode == 'torsion':
    kw['min_fn'] = DefaultKinematicMinimizer(cuda_graph=(extra == 'graph'))
init = sfxn.render_whole_pose_scoring_module(pose)(pose.coords).detach()
torch.manual_seed(20260906)
out, t = L.timed(lambda: fast_relax(pose, sfxn, pal, mm, ff, num_repeats=1, verbose=False, **kw))
fin = sfxn.render_whole_pose_scoring_module(out)(out.coords).detach().float()
print('AB', json.dumps(dict(tree=os.path.dirname(tmol.__file__).split('/')[-4], mode=mode, ds=ds, batch=batch, extra=extra,
      seconds=round(t, 3), per_struct=round(t / batch, 4), init=float(init.mean()), final_mean=float(fin.mean()), final_min=float(fin.min()), final_max=float(fin.max()))))
