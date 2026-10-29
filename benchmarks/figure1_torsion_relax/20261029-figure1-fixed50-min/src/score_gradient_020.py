"""Score + Cartesian-gradient throughput, tmol 0.2.0, same timing contract as the 0.1.58 Figure 1 harness
(5 warm-ups, 7 adaptive ~1 s samples, GPU synchronised, public forward_backward CUDA graph on GPU)."""
import argparse, json, math, os, sys, time, traceback
from pathlib import Path
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / 'src'))
import benchmark_tmol as bt
from common import SPEC, benchmark_callable, machine_metadata, sha256
from loader020 import replicated_pose_020

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True); p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
p.add_argument('--batch', type=int, required=True); p.add_argument('--threads', type=int, default=1)
p.add_argument('--protocol', choices=['score_gradient', 'score'], default='score_gradient')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
row = bt.select(a.dataset)
res = dict(engine='tmol', engine_version='0.2.0', protocol=a.protocol, device=a.device, batch_size=a.batch, modality=row['modality'],
           dataset_id=a.dataset, atoms=int(row['atoms']), residues=int(row['residues']), status='ok', error='',
           cpu_threads_requested=a.threads, machine=machine_metadata(), score_function='beta2016',
           started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
try:
    import torch, tmol
    from tmol.score import beta2016_score_function
    assert tmol.__version__.split('+')[0] == '0.2.0'
    dev = torch.device('cuda', torch.cuda.current_device()) if a.device == 'cuda' else torch.device('cpu')
    if a.device == 'cpu':
        torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    else:
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    gradient = a.protocol == 'score_gradient'
    t0 = time.perf_counter()
    pose, db = replicated_pose_020(row, dev, a.batch)
    sfxn = beta2016_score_function(dev, param_db=db)
    mode = ('forward_backward' if gradient else 'forward') if a.device == 'cuda' else False
    scorer = sfxn.render_whole_pose_scoring_module(pose, cuda_graph=mode)
    coords = pose.coords.detach().clone()
    if gradient:
        coords.requires_grad_(True)
        def op():
            torch.autograd.grad(scorer(coords).sum(), coords)
    else:
        def op():
            scorer(coords)
    if a.device == 'cuda':
        torch.cuda.synchronize()
    setup = time.perf_counter() - t0
    timings, iters = benchmark_callable(op, device=a.device)
    vc = pose.coords.detach().clone().requires_grad_(gradient)
    vs = scorer(vc)
    assert bool(torch.isfinite(vs).all()), 'non-finite score'
    if gradient:
        (g,) = torch.autograd.grad(vs.sum(), vc)
        assert bool(torch.isfinite(g).all()), 'non-finite gradient'
    res.update(prepared_atoms=int(pose.n_ats_per_block[0].sum()), seconds_per_batch_samples=timings,
               seconds_per_structure_samples=[t / a.batch for t in timings], iterations_per_sample=iters, setup_seconds=setup,
               cuda_graph_mode=mode or None, validation_score_mean=float(vs.detach().mean()),
               peak_memory_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None,
               torch_version=torch.__version__, tmol_import_version=tmol.__version__)
except Exception as e:
    res.update(status='failed', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
res['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
a.output.parent.mkdir(parents=True, exist_ok=True)
tmp = a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(res, indent=2) + '\n'); tmp.replace(a.output)
print(json.dumps({k: res.get(k) for k in ['dataset_id', 'device', 'batch_size', 'status', 'error']}))
raise SystemExit(0 if res['status'] == 'ok' else 2)
