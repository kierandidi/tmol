import sys, os, time, cProfile, pstats
sys.path.insert(0, os.path.dirname(__file__))
import relaxlab as L, torch, benchmark_tmol as bt
from tmol.optimization import build_kinforest_network, CartesianSfxnNetwork
from tmol.kinematics import MinimizerMap, PoseStackKinematicsModule
ds, batch = sys.argv[1], int(sys.argv[2]); dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, 'torsion')
pr = cProfile.Profile(); pr.enable()
t = time.perf_counter(); kn = build_kinforest_network(pose, sfxn, ff, mm, verbose=True); torch.cuda.synchronize()
print('first build', time.perf_counter() - t); pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(45)
from tmol.optimization import LBFGS_Armijo
def closure(net):
    net.zero_grad(); E = net(); E.sum().backward()
def bench(net, label, n=30):
    for _ in range(3): closure(net)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): closure(net)
    torch.cuda.synchronize(); print(label, 'ms/closure', 1000 * (time.perf_counter() - t) / n)
bench(kn, 'torsion eager')
cn = CartesianSfxnNetwork(sfxn, pose, None); bench(cn, 'cartesian eager')
# decomposed torsion
def fwd_only(net, n=30):
    for _ in range(3):
        with torch.no_grad(): net()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): net()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t) / n
print('torsion fwd-only ms', fwd_only(kn), 'cart fwd-only ms', fwd_only(cn))
from torch.profiler import profile, ProfilerActivity
for label, net in (('torsion', kn), ('cartesian', cn)):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5): closure(net)
        torch.cuda.synchronize()
    print('#####', label)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=18, max_name_column_width=70))
