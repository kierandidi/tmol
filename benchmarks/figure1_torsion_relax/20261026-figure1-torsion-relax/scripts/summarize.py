"""Summaries for the torsion-relax rerun.

Writes, under <run>/summary/:
  fastrelax_torsion/{timing_summary.csv,status.json,matrix.json}   (old fastrelax_latest schema)
  fastrelax_cartesian_tmol020_control/timing_summary.csv            (same schema)
  fastrelax_cartesian_old_0158/timing_summary.csv                   (re-summary of the old raw run)
  comparison_old_cartesian_vs_torsion.csv / .md
"""
import csv
import json
from pathlib import Path
from statistics import median

RUN = Path(__file__).resolve().parents[1]
OLD_RAW = Path('/mnt/home/kdidi/tmol-paper-benchmark/runs/20261005-fastrelax-f92d/results/raw')
OUT = RUN / 'summary'
FIELDS = ['engine', 'engine_version', 'protocol', 'dataset_id', 'modality', 'device', 'batch_size', 'atoms',
          'status', 'cpu_threads_requested', 'source_file', 'seconds_per_structure', 'structures_per_second',
          'n_samples']
SERIES = [('pyrosetta', 'cpu', 1, 'PyRosetta CPU B1'), ('tmol', 'cpu', 1, 'tmol CPU B1 (32 thr)'),
          ('tmol', 'cuda', 1, 'tmol GPU B1'), ('tmol', 'cuda', 10, 'tmol GPU B10'),
          ('tmol', 'cuda', 100, 'tmol GPU B100'), ('tmol', 'cuda', 1000, 'tmol GPU B1000')]


def load(raw_dir, rel_prefix):
    rows = {}
    for f in sorted(raw_dir.glob('*.json')):
        if f.name.endswith('.running.json'):
            continue
        r = json.loads(f.read_text())
        s = r['seconds_per_structure_samples'][0] if r.get('seconds_per_structure_samples') else None
        rows[(r['engine'], r['device'], int(r['batch_size']), r['dataset_id'])] = dict(
            engine=r['engine'], engine_version=r['engine_version'], protocol='fastrelax',
            dataset_id=r['dataset_id'], modality=r['modality'], device=r['device'],
            batch_size=int(r['batch_size']), atoms=r['atoms'], status=r['status'],
            cpu_threads_requested=r['cpu_threads_requested'], source_file=f'{rel_prefix}/{f.name}',
            seconds_per_structure=s if r['status'] == 'ok' else '',
            structures_per_second=(1 / s) if (r['status'] == 'ok' and s) else '', n_samples=1,
            _initial=r.get('initial_score_mean'), _final=r.get('validation_score_mean'),
            _final_min=r.get('validation_score_min'), _final_max=r.get('validation_score_max'),
            _error=r.get('error', ''), _score_function=r.get('score_function', ''),
            _mem=r.get('peak_memory_bytes'), _node=(r.get('machine') or {}).get('hostname'),
            _job=(r.get('machine') or {}).get('slurm_job_id'))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: (r['engine'], r['device'], r['batch_size'], float(r['atoms']))))


new = load(RUN / 'results/raw', 'results/raw')
ctrl = load(RUN / 'results/raw_cart020', 'results/raw_cart020')
old = load(OLD_RAW, '../20261005-fastrelax-f92d/results/raw')
write_csv(OUT / 'fastrelax_torsion/timing_summary.csv', new)
write_csv(OUT / 'fastrelax_cartesian_tmol020_control/timing_summary.csv', ctrl)
write_csv(OUT / 'fastrelax_cartesian_old_0158/timing_summary.csv', old)

matrix = json.loads((RUN / 'matrix.json').read_text())
expected = [(r['engine'], r['device'], r['batch'], r['dataset']) for k in ('gpu', 'cpu', 'pyro') for r in matrix[k]]
ok = [k for k in expected if k in new and new[k]['status'] == 'ok']
failed = [f'{k}: {new[k]["_error"]}' for k in expected if k in new and new[k]['status'] != 'ok']
pending = [k for k in expected if k not in new]
status = dict(expected=len(expected), successful=len(ok), failed=failed, pending=[str(k) for k in pending],
              complete=len(ok) == len(expected), relax_mode='torsion (atom-tree / kinematic)',
              tmol_version='0.2.0', cpu_threads=dict(tmol=32, pyrosetta=1), gpu='one H200 per job',
              timing='one seeded native FastRelax repeat per point; no timing uncertainty estimated')
(OUT / 'fastrelax_torsion/status.json').write_text(json.dumps(status, indent=1) + '\n')
(OUT / 'fastrelax_torsion/matrix.json').write_text(json.dumps({k: matrix[k] for k in ('gpu', 'cpu', 'pyro')}, indent=1) + '\n')

# ---- comparison table
def val(d, key, field='seconds_per_structure'):
    r = d.get(key)
    return r[field] if r and r['status'] == 'ok' else None


systems = []
for r in matrix['gpu'] + matrix['cpu']:
    if r['dataset'] not in [s[0] for s in systems]:
        systems.append((r['dataset'], None))
rows = []
for ds, _ in systems:
    any_row = next(v for k, v in list(new.items()) + list(old.items()) if k[3] == ds)
    pyro_old = val(old, ('pyrosetta', 'cpu', 1, ds)); pyro_new = val(new, ('pyrosetta', 'cpu', 1, ds))
    for engine, dev, b, label in SERIES:
        key = (engine, dev, b, ds)
        o, c, n = val(old, key), val(ctrl, key), val(new, key)
        if engine == 'pyrosetta':
            c = None
        rec = dict(dataset_id=ds, modality=any_row['modality'], atoms=any_row['atoms'], series=label,
                   old_cartesian_s_per_struct=o, cartesian_tmol020_control_s_per_struct=c,
                   torsion_s_per_struct=n,
                   torsion_over_old_cartesian=(n / o) if (n and o) else None,
                   torsion_over_cartesian_same_version=(n / c) if (n and c) else None,
                   speedup_vs_pyrosetta_old_cartesian=(pyro_old / o) if (pyro_old and o) else None,
                   speedup_vs_pyrosetta_torsion=(pyro_new / n) if (pyro_new and n) else None,
                   initial_score_torsion=new[key]['_initial'] if key in new else None,
                   final_score_torsion=new[key]['_final'] if key in new else None,
                   final_score_old_cartesian=old[key]['_final'] if key in old else None,
                   final_score_cartesian_tmol020=ctrl[key]['_final'] if key in ctrl else None,
                   score_function_torsion=new[key]['_score_function'] if key in new else '',
                   status_torsion=new[key]['status'] if key in new else 'missing',
                   status_old=old[key]['status'] if key in old else 'missing')
        rows.append(rec)
cols = list(rows[0])
with (OUT / 'comparison_old_cartesian_vs_torsion.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=cols); w.writeheader(); w.writerows(rows)

f = lambda x, p=2: '–' if x is None else (f'{x:.{p}f}' if abs(x) < 1000 else f'{x:,.0f}')
md = ['| system | atoms | series | old Cart. 0.1.58 (s) | Cart. 0.2.0 ctrl (s) | torsion (s) | torsion/old | torsion/Cart-0.2.0 | PyRos speedup old | PyRos speedup torsion |',
      '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    md.append(f"| {r['dataset_id']} | {r['atoms']} | {r['series']} | {f(r['old_cartesian_s_per_struct'])} | "
              f"{f(r['cartesian_tmol020_control_s_per_struct'])} | {f(r['torsion_s_per_struct'])} | "
              f"{f(r['torsion_over_old_cartesian'])} | {f(r['torsion_over_cartesian_same_version'])} | "
              f"{f(r['speedup_vs_pyrosetta_old_cartesian'], 1)} | {f(r['speedup_vs_pyrosetta_torsion'], 1)} |")
# medians per series
md += ['', '**Median over systems (s per structure; speedups = PyRosetta CPU 1 thread / tmol, matched per system)**', '',
       '| series | n | old Cart. | torsion | torsion/old | torsion/Cart-0.2.0 | PyRos speedup old Cart. | PyRos speedup torsion |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for _, _, _, label in SERIES:
    sub = [r for r in rows if r['series'] == label]
    g = lambda k: [r[k] for r in sub if r[k] is not None]
    n_ok = len(g('torsion_s_per_struct'))
    md.append(f"| {label} | {n_ok} | {f(median(g('old_cartesian_s_per_struct')) if g('old_cartesian_s_per_struct') else None)} | "
              f"{f(median(g('torsion_s_per_struct')) if g('torsion_s_per_struct') else None)} | "
              f"{f(median(g('torsion_over_old_cartesian')) if g('torsion_over_old_cartesian') else None)} | "
              f"{f(median(g('torsion_over_cartesian_same_version')) if g('torsion_over_cartesian_same_version') else None)} | "
              f"{f(median(g('speedup_vs_pyrosetta_old_cartesian')) if g('speedup_vs_pyrosetta_old_cartesian') else None, 1)} | "
              f"{f(median(g('speedup_vs_pyrosetta_torsion')) if g('speedup_vs_pyrosetta_torsion') else None, 1)} |")
(OUT / 'comparison_old_cartesian_vs_torsion.md').write_text('\n'.join(md) + '\n')
print(json.dumps({k: status[k] for k in ('expected', 'successful', 'complete')}), 'failed:', failed[:5], 'pending:', len(pending))
