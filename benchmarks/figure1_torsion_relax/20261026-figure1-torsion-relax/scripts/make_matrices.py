"""Build slurm matrices from the old Figure 1 FastRelax matrix (same systems / series)."""
import json, pathlib
run = pathlib.Path(__file__).resolve().parents[1]
old = json.loads(pathlib.Path('/mnt/home/kdidi/tmol-paper-benchmark/runs/20261005-fastrelax-f92d/matrix.json').read_text())
def tag(rows, mode): return [dict(r, mode=mode) for r in rows]
full = {'gpu': tag(old['gpu'], 'torsion'), 'cpu': tag(old['cpu'], 'torsion'), 'pyro': tag(old['pyro'], 'torsion'),
        'gpu_cart': tag(old['gpu'], 'cartesian'), 'cpu_cart': tag(old['cpu'], 'cartesian')}
(run/'matrix.json').write_text(json.dumps(full, indent=1)+'\n')
pil = lambda rows, ds, b=None: [r for r in rows if r['dataset'] in ds and (b is None or r['batch'] in b)]
pilot = {'gpu': pil(full['gpu'], ['5uoi','hsp90'], [1,10]), 'cpu': pil(full['cpu'], ['5uoi']),
         'pyro': pil(full['pyro'], ['5uoi','hsp90']), 'gpu_cart': pil(full['gpu_cart'], ['5uoi'], [1,10])}
(run/'matrix_pilot.json').write_text(json.dumps(pilot, indent=1)+'\n')
print({k: len(v) for k, v in full.items()}, {k: len(v) for k, v in pilot.items()})
