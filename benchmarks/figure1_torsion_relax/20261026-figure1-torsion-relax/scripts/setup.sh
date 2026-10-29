#!/bin/bash
#SBATCH --job-name=torsion-setup
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax/logs/setup-%j.log
set -euo pipefail
run=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261026-figure1-torsion-relax
image=/mnt/home/kdidi/apptainer-artifacts/latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif
# Official uw-ipd/tmol v0.2.0 release wheels; sha256 match the GitHub release asset digests.
(cd $run/wheels && sha256sum -c ../metadata/wheel-sha256.txt)
apptainer exec --nv "$image" python -m venv --system-site-packages "$run/env"
apptainer exec --nv "$image" "$run/env/bin/python" -m pip install \
  $run/wheels/tmol_ops_cu13-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl \
  $run/wheels/tmol-0.2.0-cp312-cp312-manylinux_2_28_x86_64.whl > "$run/logs/install.log" 2>&1
apptainer exec --nv "$image" "$run/env/bin/python" - <<'PY'
import torch, tmol, atomworks, importlib.metadata as m
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(tmol.__version__, tmol.__file__, m.version('atomworks'), atomworks.__file__)
from tmol.relax._fast_relax import DefaultKinematicMinimizer
from tmol.optimization import KinematicMinimizer
assert torch.cuda.is_available()
PY
apptainer exec --nv "$image" "$run/env/bin/python" -m pip freeze > "$run/metadata/pip-freeze.txt"
sha256sum "$image" > "$run/metadata/container-sha256.txt"
{ nvidia-smi --query-gpu=name,driver_version --format=csv; hostname; } > "$run/metadata/setup-gpu.txt"
