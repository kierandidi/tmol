import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import relaxlab as L, torch, benchmark_tmol as bt
from tmol.pack import pack_rotamers, PackerTask
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pack.rotamer._conjugated_groups import add_conjugated_group_sampler
from tmol.score import ScoreType
ds, batch = sys.argv[1], int(sys.argv[2]); dev = torch.device('cuda')
pose, sfxn, ff, mm, pal = L.setup(bt.select(ds), dev, batch, 'torsion')
samplers = [create_dunbrack_sampler_from_database(sfxn._param_db, dev), FixedAAChiSampler(), IncludeCurrentSampler()]
def make_task():
    t = PackerTask(pose, pal); t.restrict_to_repacking()
    for s in samplers: t.add_conformer_sampler(s)
    add_conjugated_group_sampler(t, pose); return t
sfxn.set_weight(ScoreType.fa_ljrep, 0.55 * 0.265)
for _ in range(2):  # warm
    L.timed(lambda: pack_rotamers(pose, sfxn, make_task(), False))
out, t = L.timed(lambda: pack_rotamers(pose, sfxn, make_task(), True)); print('pack wall', t)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    pack_rotamers(pose, sfxn, make_task(), False); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=90))
