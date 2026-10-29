import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import relaxlab as L, torch, benchmark_tmol as bt
from tmol.optimization import build_kinforest_network, CartesianSfxnNetwork
ds, batch = sys.argv[1], int(sys.argv[2]); dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, 'torsion')
# put the pose in a realistic stage-1 state: weights as in relax stage 1 minimisation
from tmol.score import ScoreType
sfxn.set_weight(ScoreType.fa_ljrep, 0.55 * 0.051)
for dt in (torch.float32, torch.float64):
    net = build_kinforest_network(pose, sfxn, ff, mm, kin_dtype=dt)
    with torch.no_grad():
        Es = torch.stack([net().clone() for _ in range(6)])
    print(dt, 'repeat-eval spread (max-min per pose, max over poses):', float((Es.max(0).values - Es.min(0).values).max()), ' |E| mean', float(Es.abs().mean()))
    # tiny perturbation of the DOFs
    base = net.masked_dofs.detach().clone()
    with torch.no_grad():
        E0 = net().clone()
        for eps in (1e-9, 1e-7, 1e-6, 1e-5):
            net.masked_dofs.copy_(base + eps * torch.randn_like(base)); E1 = net().clone()
            print('  perturb', eps, 'max |dE| over poses', float((E1 - E0).abs().max()))
        net.masked_dofs.copy_(base)
cn = CartesianSfxnNetwork(sfxn, pose, None)
with torch.no_grad():
    Es = torch.stack([cn().clone() for _ in range(6)])
print('cartesian repeat spread', float((Es.max(0).values - Es.min(0).values).max()))
