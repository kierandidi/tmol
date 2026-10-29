"""Summaries for the fixed-50-step run (torsion + Cartesian, tmol and PyRosetta).

Writes summary/{torsion,cartesian}/timing_summary.csv (old fastrelax_latest schema),
summary/status.json and summary/comparison.{csv,md}: 50-step vs the 200-step runs
(runs/20261026-figure1-torsion-relax: torsion and tmol-0.2.0 Cartesian control;
runs/20261005-fastrelax-f92d: old Cartesian incl. PyRosetta).
"""
import csv
import glob
import json
from pathlib import Path
from statistics import median

RUN = Path(__file__).resolve().parents[1]
BASE = RUN.parent
FIELDS = ['engine', 'engine_version', 'protocol', 'dataset_id', 'modality', 'device', 'batch_size', 'atoms', 'status',
          'cpu_threads_requested', 'source_file', 'seconds_per_structure', 'structures_per_second', 'n_samples']
SERIES = [('pyrosetta', 'cpu', 1, 'PyRosetta CPU B1'), ('tmol', 'cpu', 1, 'tmol CPU B1 (32 thr)'), ('tmol', 'cuda', 1, 'tmol GPU B1'),
          ('tmol', 'cuda', 10, 'tmol GPU B10'), ('tmol', 'cuda', 100, 'tmol GPU B100'), ('tmol', 'cuda', 1000, 'tmol GPU B1000')]


def load(pattern, mode=None):
    out = {}
    for f in sorted(glob.glob(str(pattern))):
        if f.endswith('.running.json'):
            continue
        r = json.loads(Path(f).read_text())
        if mode and r.get('relax_mode', 'cartesian') != mode:
            continue
        s = r['seconds_per_structure_samples'][0] if r.get('seconds_per_structure_samples') else None
        ok = r['status'] == 'ok'
        out[(r['engine'], r['device'], int(r['batch_size']), r['dataset_id'])] = dict(
            engine=r['engine'], engine_version=r['engine_version'], protocol='fastrelax', dataset_id=r['dataset_id'],
            modality=r['modality'], device=r['device'], batch_size=int(r['batch_size']), atoms=r['atoms'], status=r['status'],
            cpu_threads_requested=r['cpu_threads_requested'], source_file=str(Path(f).relative_to(BASE)),
            seconds_per_structure=s if ok else '', structures_per_second=(1 / s) if ok and s else '', n_samples=1,
            _final=r.get('validation_score_mean'), _err=r.get('error', ''))
    return out


new = {m: load(RUN / f'results/raw/*-fastrelax-{m}-*.json', m) for m in ('torsion', 'cartesian')}
ref200 = {'torsion': load(BASE / '20261026-figure1-torsion-relax/results/raw/*.json', 'torsion'),
          'cartesian': {**load(BASE / '20261005-fastrelax-f92d/results/raw/pyrosetta-*.json'),
                        **load(BASE / '20261026-figure1-torsion-relax/results/raw_cart020/*.json', 'cartesian')}}
for m in new:
    d = RUN / 'summary' / m
    d.mkdir(parents=True, exist_ok=True)
    with (d / 'timing_summary.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(new[m].values(), key=lambda r: (r['engine'], r['device'], r['batch_size'], float(r['atoms']))))

matrix = json.loads((RUN / 'matrix.json').read_text())
extra = json.loads((RUN / 'matrix_extra.json').read_text())
for k in ('gpu', 'cpu', 'gpu_cart', 'cpu_cart'):
    matrix[k] = matrix[k] + extra[k]
status = {}
for m, keys in (('torsion', ('gpu', 'cpu', 'pyro')), ('cartesian', ('gpu_cart', 'cpu_cart', 'pyro_cart'))):
    exp = [(r['engine'], r['device'], r['batch'], r['dataset']) for k in keys for r in matrix[k]]
    bad = {str(k): new[m][k]['_err'] for k in exp if k in new[m] and new[m][k]['status'] != 'ok'}
    ok = [k for k in exp if k in new[m] and new[m][k]['status'] == 'ok']
    status[m] = dict(expected=len(exp), successful=len(ok), failed=bad, pending=[str(k) for k in exp if k not in new[m]], complete=len(ok) == len(exp))
status['minimization'] = '50 L-BFGS iterations per minimization (tmol: fixed_iterations; PyRosetta: max_iter=50, tolerance 1e-12)'
(RUN / 'summary/status.json').write_text(json.dumps(status, indent=1) + '\n')

v = lambda d, k: d[k]['seconds_per_structure'] if k in d and d[k]['status'] == 'ok' else None
systems = list(dict.fromkeys(r['dataset'] for r in matrix['gpu'] + matrix['cpu']))
rows = []
for m in ('torsion', 'cartesian'):
    for ds in systems:
        p50, p200 = v(new[m], ('pyrosetta', 'cpu', 1, ds)), v(ref200[m], ('pyrosetta', 'cpu', 1, ds))
        for e, dev, b, label in SERIES:
            k = (e, dev, b, ds)
            a, c = v(new[m], k), v(ref200[m], k)
            src = new[m].get(k) or ref200[m].get(k)
            rows.append(dict(mode=m, dataset_id=ds, modality=src['modality'] if src else '', atoms=src['atoms'] if src else '', series=label,
                             s_per_struct_50=a, s_per_struct_200=c, ratio_50_over_200=(a / c) if a and c else None,
                             speedup_vs_pyrosetta_50=(p50 / a) if a and p50 else None, speedup_vs_pyrosetta_200=(p200 / c) if c and p200 else None,
                             final_score_50=new[m][k]['_final'] if k in new[m] else None, final_score_200=ref200[m][k]['_final'] if k in ref200[m] else None))
with (RUN / 'summary/comparison.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
f = lambda x, p=2: '–' if x is None else (f'{x:.{p}f}' if abs(x) < 1000 else f'{x:,.0f}')
md = []
for m in ('torsion', 'cartesian'):
    md += [f'### {m}: median over systems (s/structure; speedup = PyRosetta CPU 1 thread / tmol, same mode and step budget)', '',
           '| series | n | 50-step (s) | 200-step (s) | 50/200 | PyRos speedup, 50-step | PyRos speedup, 200-step |', '|---|---:|---:|---:|---:|---:|---:|']
    for _, _, _, label in SERIES:
        sub = [r for r in rows if r['mode'] == m and r['series'] == label]
        g = lambda k: [r[k] for r in sub if r[k] is not None]
        md.append(f"| {label} | {len(g('s_per_struct_50'))} | {f(median(g('s_per_struct_50')) if g('s_per_struct_50') else None)} | "
                  f"{f(median(g('s_per_struct_200')) if g('s_per_struct_200') else None)} | {f(median(g('ratio_50_over_200')) if g('ratio_50_over_200') else None)} | "
                  f"{f(median(g('speedup_vs_pyrosetta_50')) if g('speedup_vs_pyrosetta_50') else None, 1)} | "
                  f"{f(median(g('speedup_vs_pyrosetta_200')) if g('speedup_vs_pyrosetta_200') else None, 1)} |")
    md.append('')
# torsion vs cartesian at 50 steps
md += ['### torsion / Cartesian time ratio at 50 steps (median over systems)', '', '| series | torsion/Cartesian |', '|---|---:|']
for _, _, _, label in SERIES:
    rs = []
    for ds in systems:
        t = next((r['s_per_struct_50'] for r in rows if r['mode'] == 'torsion' and r['series'] == label and r['dataset_id'] == ds), None)
        c = next((r['s_per_struct_50'] for r in rows if r['mode'] == 'cartesian' and r['series'] == label and r['dataset_id'] == ds), None)
        if t and c:
            rs.append(t / c)
    md.append(f"| {label} | {f(median(rs)) if rs else '–'} |")
(RUN / 'summary/comparison.md').write_text('\n'.join(md) + '\n')
print(json.dumps({m: {k: status[m][k] for k in ('expected', 'successful', 'complete')} for m in ('torsion', 'cartesian')}))
