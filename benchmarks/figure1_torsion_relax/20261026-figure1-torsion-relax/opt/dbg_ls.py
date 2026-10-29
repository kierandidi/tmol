import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import relaxlab as L, torch, benchmark_tmol as bt
import tmol.optimization._lbfgs_armijo as LA
from tmol.relax import fast_relax
from tmol.relax._fast_relax import DefaultKinematicMinimizer, DEFAULT_RELAX_SCHEDULE
mode, ds, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, mode)
orig = LA.armijo_linesearch_segmented
calls = []
def wrapped(func, derphi0, old_fval, alpha0, searching, **kw):
    log = []
    def f(alpha):
        phi = func(alpha); log.append((alpha.clone().cpu(), phi.clone().cpu())); return phi
    out = orig(f, derphi0, old_fval, alpha0, searching, **kw)
    n = out[2]
    if n >= 12:
        p_failed = [int(i) for i in (out[3] == 3).nonzero().flatten().tolist()]
        rec = dict(n_evals=n, failed=p_failed, derphi0={int(i): float(derphi0[i]) for i in p_failed}, phi0={int(i): float(old_fval[i]) for i in p_failed},
                   alpha0={int(i): float(alpha0[i]) for i in p_failed},
                   curve={int(i): [(float(a[i]), float(ph[i])) for a, ph in log] for i in p_failed[:3]})
        calls.append(rec)
    return out
LA.armijo_linesearch_segmented = wrapped
sched = [DEFAULT_RELAX_SCHEDULE[0]]
torch.manual_seed(20260906)
fast_relax(pose, sfxn, pal, mm, ff, num_repeats=1, schedule=sched, min_fn=DefaultKinematicMinimizer(cuda_graph=False), verbose=False)
print('n long searches', len(calls))
for rec in calls[:4]:
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}))
    for p, cv in rec['curve'].items():
        print(' pose', p, 'phi0', rec['phi0'][p], 'derphi0', rec['derphi0'][p])
        for a, ph in cv[:24]: print('   alpha %.3e  dphi %.4e  expected(linear) %.4e' % (a, ph - rec['phi0'][p], a * rec['derphi0'][p]))
