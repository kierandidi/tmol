"""Optimised (patched tmol) vs baseline (tmol 0.2.0 wheel) FastRelax timings.

Reads results/opt2_gpu (torsion), results/opt2_gpu_cart (Cartesian) and compares with
results/raw (torsion baseline) and results/raw_cart020 (Cartesian baseline); also divides
the PyRosetta torsion time by the optimised tmol time.
"""
import csv
import glob
import json
from pathlib import Path
from statistics import median

RUN = Path(__file__).resolve().parents[1]


def load(pattern):
    out = {}
    for f in glob.glob(str(RUN / pattern)):
        if f.endswith('.running.json'):
            continue
        r = json.loads(Path(f).read_text())
        out[(r['dataset_id'], r['batch_size'])] = r
    return out


base_t = load('results/raw/tmol-fastrelax-torsion-cuda-*.json')
base_c = load('results/raw_cart020/tmol-fastrelax-cartesian-cuda-*.json')
opt_t = load('results/opt2_gpu/tmol-fastrelax-torsion-cuda-*.json')
opt_c = load('results/opt2_gpu_cart/tmol-fastrelax-cartesian-cuda-*.json')
pyro = {k[0]: v for k, v in load('results/raw/pyrosetta-fastrelax-torsion-cpu-*.json').items()}


def sps(r):
    return r['seconds_per_structure_samples'][0] if r and r['status'] == 'ok' else None


rows = []
for key in sorted(set(opt_t) | set(opt_c), key=lambda k: (k[1], k[0])):
    ds, b = key
    bt, ot, bc, oc = sps(base_t.get(key)), sps(opt_t.get(key)), sps(base_c.get(key)), sps(opt_c.get(key))
    p = sps(pyro.get(ds))
    rows.append(dict(
        dataset_id=ds, batch=b, atoms=(opt_t.get(key) or opt_c.get(key))['atoms'],
        torsion_baseline_s=bt, torsion_optimised_s=ot, torsion_speedup=(bt / ot) if bt and ot else None,
        cartesian_baseline_s=bc, cartesian_optimised_s=oc, cartesian_speedup=(bc / oc) if bc and oc else None,
        torsion_over_cartesian_optimised=(ot / oc) if ot and oc else None,
        pyrosetta_torsion_s=p, speedup_vs_pyrosetta_torsion_baseline=(p / bt) if p and bt else None,
        speedup_vs_pyrosetta_torsion_optimised=(p / ot) if p and ot else None,
        final_torsion_baseline=(base_t.get(key) or {}).get('validation_score_mean'),
        final_torsion_optimised=(opt_t.get(key) or {}).get('validation_score_mean'),
        final_cartesian_baseline=(base_c.get(key) or {}).get('validation_score_mean'),
        final_cartesian_optimised=(opt_c.get(key) or {}).get('validation_score_mean'),
        status_t=(opt_t.get(key) or {}).get('status', 'missing'), status_c=(opt_c.get(key) or {}).get('status', 'missing')))
out = RUN / 'summary'
out.mkdir(exist_ok=True)
with (out / 'optimised_vs_baseline.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
f = lambda x, p=2: '–' if x is None else f'{x:.{p}f}'
md = ['| system | B | torsion base (s) | torsion opt (s) | speedup | Cart base (s) | Cart opt (s) | speedup | torsion/Cart (opt) | PyRos tors. / tmol-opt | final E tors base | tors opt |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    md.append(f"| {r['dataset_id']} | {r['batch']} | {f(r['torsion_baseline_s'],3)} | {f(r['torsion_optimised_s'],3)} | {f(r['torsion_speedup'])} | "
              f"{f(r['cartesian_baseline_s'],3)} | {f(r['cartesian_optimised_s'],3)} | {f(r['cartesian_speedup'])} | {f(r['torsion_over_cartesian_optimised'])} | "
              f"{f(r['speedup_vs_pyrosetta_torsion_optimised'],1)} | {f(r['final_torsion_baseline'],1)} | {f(r['final_torsion_optimised'],1)} |")
md += ['', '**Median speedup by batch size (baseline time / optimised time)**', '', '| B | n | torsion | Cartesian | torsion/Cart (opt) | PyRosetta torsion / tmol-opt |', '|---:|---:|---:|---:|---:|---:|']
for b in sorted({r['batch'] for r in rows}):
    sub = [r for r in rows if r['batch'] == b]
    g = lambda k: [r[k] for r in sub if r[k] is not None]
    med = lambda k, p=2: f(median(g(k)), p) if g(k) else '–'
    md.append(f"| {b} | {len(g('torsion_speedup'))} | {med('torsion_speedup')} | {med('cartesian_speedup')} | {med('torsion_over_cartesian_optimised')} | {med('speedup_vs_pyrosetta_torsion_optimised', 1)} |")
(out / 'optimised_vs_baseline.md').write_text('\n'.join(md) + '\n')
print(len(rows), 'rows;', sum(r['status_t'] != 'ok' for r in rows), 'torsion non-ok;', sum(r['status_c'] != 'ok' for r in rows), 'cart non-ok')
