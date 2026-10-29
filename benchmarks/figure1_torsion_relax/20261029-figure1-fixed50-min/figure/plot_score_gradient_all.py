"""Figure 1 score+gradient panel with ALL 27 inputs: tmol 0.2.0 (GPU B1/10/100/1000, CPU 1 thread; results/sg) and
PyRosetta 2024.39 CPU 1 thread (measurements reused from the 0.1.58-era Figure 1 data, same PyRosetta)."""
import csv, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parent; RUN = ROOT.parent
OLD = RUN.parent / '20261005-score-gap-reproduction/figure/data/performance/throughput.csv'
C = json.loads((ROOT / 'style/theme.json').read_text())['colors']
font = Path('/System/Library/Fonts/Helvetica.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family': 'Helvetica' if font.exists() else 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none',
    'axes.labelcolor': C['ink'], 'axes.edgecolor': C['ink'], 'text.color': C['ink'], 'xtick.color': C['ink'], 'ytick.color': C['ink'],
    'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 5.5})
rows = []
for f in glob.glob(str(RUN / 'results/sg/*.json')):
    if f.endswith('.running.json'):
        continue
    r = json.loads(Path(f).read_text())
    if r['status'] != 'ok':
        continue
    sps = np.array([1 / s for s in r['seconds_per_structure_samples']])
    rows.append(dict(engine='tmol', device=r['device'], batch=int(r['batch_size']), ds=r['dataset_id'], mod=r['modality'], atoms=float(r['prepared_atoms']),
                     y=float(np.median(sps)), lo=float(np.quantile(sps, .25)), hi=float(np.quantile(sps, .75))))
for r in csv.DictReader(OLD.open()):
    if r['engine'] == 'pyrosetta' and r['protocol'] == 'score_gradient' and r['status'] == 'ok':
        rows.append(dict(engine='pyrosetta', device='cpu', batch=1, ds=r['dataset_id'], mod=r['modality'], atoms=float(r['prepared_atoms']),
                         y=float(r['structures_per_second']), lo=float(r['q25']), hi=float(r['q75'])))
with (RUN / 'summary/score_gradient_panel_data.csv').open('w', newline='') as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
SERIES = [('pyrosetta', 'cpu', 1, 'PyRosetta CPU', 'target_gray', 's', '--'), ('tmol', 'cpu', 1, 'tmol CPU', 'amber', 'D', '--'),
          ('tmol', 'cuda', 1, 'tmol GPU B1', 'blue', 'o', '-'), ('tmol', 'cuda', 10, 'tmol GPU B10', 'teal', '^', '-'),
          ('tmol', 'cuda', 100, 'tmol GPU B100', 'mint', 'v', '-'), ('tmol', 'cuda', 1000, 'tmol GPU B1000', 'purple', 'P', '-')]
fig = plt.figure(figsize=(91 / 25.4, 76 / 25.4), facecolor='white')
fig.legend(handles=[Line2D([], [], color=C[c], marker=m, linestyle=l, linewidth=.8, markersize=2.5, label=lab) for _, _, _, lab, c, m, l in SERIES],
           loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2, handletextpad=.5)
ax = fig.add_axes([14 / 91, 13 / 76, 72 / 91, 42 / 76])
for e, dev, b, lab, col, mk, ls in SERIES:
    d = sorted([r for r in rows if r['engine'] == e and r['device'] == dev and r['batch'] == b], key=lambda r: r['atoms'])
    if not d:
        continue
    x = np.array([r['atoms'] for r in d]); y = np.array([r['y'] for r in d]); lo = np.array([r['lo'] for r in d]); hi = np.array([r['hi'] for r in d])
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=C[col], marker=mk, linestyle=ls, linewidth=.8, markersize=2.5, markeredgewidth=.4,
                elinewidth=.5, capsize=1, zorder=3)
n = len({r['ds'] for r in rows if r['engine'] == 'tmol'})
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(600, 30000)
ax.set_xticks([1000, 10000]); ax.set_xticklabels(['1,000', '10,000'])
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5])); ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}' if v < 1000 else f'{v:,.0f}'))
ax.yaxis.set_minor_formatter(NullFormatter()); ax.tick_params(which='minor', length=1, width=.25)
ax.set_xlabel('Atoms per structure', labelpad=3); ax.set_ylabel('Speed (structures/s)', labelpad=3)
ax.set_title('Score + gradient', pad=5); ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', which='major', color=C['light_gray'], linewidth=.25, zorder=0)
fig.text(.5, 59 / 76, f'tmol 0.2.0 · CPU 1 thread · all inputs ({n} tmol systems, 27 PyRosetta)', ha='center', fontsize=5, color=C['target_gray'])
out = ROOT / 'score_gradient_all_inputs'
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(f'{out}.{ext}', dpi=300, **({'metadata': {'Date': None}} if ext != 'png' else {}))
print('wrote', out, 'rows', len(rows), 'tmol systems', n)
