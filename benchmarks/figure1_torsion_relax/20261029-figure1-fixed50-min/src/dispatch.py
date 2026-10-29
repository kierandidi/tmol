"""Scheduler-only dispatch; each task is a separate process (and separate licensed PyRosetta process)."""
from pathlib import Path
import json, os, subprocess, sys
run = Path(__file__).resolve().parents[1]
image = '/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif'
matrix_file, mode_key, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
row = json.loads((run/matrix_file).read_text())[mode_key][int(os.environ['SLURM_ARRAY_TASK_ID'])]
threads = row['threads']
env = dict(os.environ)
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    env[k] = str(threads)
python = str(run/'env/bin/python') if row['engine']=='tmol' else '/mnt/home/kdidi/.local/bin/python3.12'
cg = row.get('cuda_graph')
name = f"{row['engine']}-fastrelax-{row['mode']}{'-graph'+cg if cg else ''}-{row['device']}-b{row['batch']}-{row['dataset']}.json"
cmd = ['apptainer','exec'] + (['--nv'] if row['device']=='cuda' else []) + [image, python, str(run/'src/torsion_fastrelax.py')]
for k in ['engine','mode','dataset','device','batch','threads']:
    cmd += ['--'+k, str(row[k])]
if cg: cmd += ['--cuda-graph', cg]
cmd += ['--output', str(run/outdir/name)]
print(json.dumps(row), 'SLURM_JOB_ID', os.environ.get('SLURM_JOB_ID'), 'node', os.uname().nodename, flush=True)
raise SystemExit(subprocess.run(cmd, env=env).returncode)
