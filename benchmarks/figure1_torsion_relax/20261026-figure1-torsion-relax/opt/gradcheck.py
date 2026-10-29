import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import relaxlab as L, torch, benchmark_tmol as bt
from tmol.optimization import build_kinforest_network
from tmol.score import ScoreType
from tmol.relax import fast_relax
from tmol.relax._fast_relax import DefaultKinematicMinimizer, DEFAULT_RELAX_SCHEDULE
ds, batch = sys.argv[1], int(sys.argv[2]); dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, 'torsion')
torch.manual_seed(1)
# realistic state: pack at low fa_rep (stage 1 packing) via one pack-min stage, then test at min weights
sfxn.set_weight(ScoreType.fa_ljrep, 0.55 * 0.051)
net = build_kinforest_network(pose, sfxn, ff, mm)
x0 = net.masked_dofs.detach().clone(); seg = net.segment_ids
def E_at(x):
    with torch.no_grad():
        net.masked_dofs.copy_(x); return net().clone()
def grad_at(x):
    with torch.no_grad(): net.masked_dofs.copy_(x)
    net.zero_grad(); E = net(); E.sum().backward(); return net.masked_dofs.grad.detach().clone(), E.detach().clone()
g, E0 = grad_at(x0)
npz = batch
def per_pose(v): return torch.zeros(npz, device=dev).index_add_(0, seg, v)
for trial in range(3):
    d = torch.randn_like(x0)
    gd = per_pose(g * d)
    res = []
    for h in (1e-4, 1e-3, 1e-2):
        Ep = E_at(x0 + h * d); Em = E_at(x0 - h * d)
        fd = (Ep - Em) / (2 * h)
        res.append(fd)
    print('trial', trial)
    for p in range(min(npz, 6)):
        print('  pose', p, 'analytic', float(gd[p]), 'fd(1e-4,1e-3,1e-2)', [round(float(r[p]), 1) for r in res])
# per DOF type check using masks: restrict direction to subsets by dof type
dt = net.full_dofs  # shape [n_nodes, 9]
