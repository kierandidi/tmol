from pathlib import Path
import json, os, subprocess, sys
run = Path(__file__).resolve().parents[1]
image = '/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif'
key, outdir = sys.argv[1], sys.argv[2]
row = json.loads((run / 'matrix_sg_hivrt.json').read_text())[key][int(os.environ['SLURM_ARRAY_TASK_ID'])]
env = dict(os.environ)
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    env[k] = str(row['threads'])
name = f"tmol-score_gradient-{row['device']}-t{row['threads']}-b{row['batch']}-{row['dataset']}.json"
cmd = ['apptainer', 'exec'] + (['--nv'] if row['device'] == 'cuda' else []) + [image, str(run / 'env/bin/python'), str(run / 'src/score_gradient_020.py'),
       '--dataset', row['dataset'], '--device', row['device'], '--batch', str(row['batch']), '--threads', str(row['threads']), '--output', str(run / outdir / name)]
print(json.dumps(row), 'job', os.environ.get('SLURM_JOB_ID'), flush=True)
raise SystemExit(subprocess.run(cmd, env=env).returncode)
