"""Figure 1 FastRelax panel (speed vs input atoms) for the fixed-50-step runs: Cartesian | torsion.
Same series, axes and style as runs/20261005-score-gap-reproduction/figure/plot_performance.py."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent
C = json.loads((ROOT / 'style/theme.json').read_text())['colors']
font = Path('/System/Library/Fonts/Helvetica.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family': 'Helvetica' if font.exists() else 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none',
    'axes.labelcolor': C['ink'], 'axes.edgecolor': C['ink'], 'text.color': C['ink'], 'xtick.color': C['ink'], 'ytick.color': C['ink'],
    'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 5.5})
SERIES = [('pyrosetta', 'cpu', 1, 'PyRosetta CPU', 'target_gray', 's', '--'), ('tmol', 'cpu', 1, 'tmol CPU', 'amber', 'D', '--'),
          ('tmol', 'cuda', 1, 'tmol GPU B1', 'blue', 'o', '-'), ('tmol', 'cuda', 10, 'tmol GPU B10', 'teal', '^', '-'),
          ('tmol', 'cuda', 100, 'tmol GPU B100', 'mint', 'v', '-'), ('tmol', 'cuda', 1000, 'tmol GPU B1000', 'purple', 'P', '-')]
PANELS = [('cartesian', 'Cartesian relax, 50 steps'), ('torsion', 'Torsion (atom-tree) relax, 50 steps')]
fig = plt.figure(figsize=(183 / 25.4, 76 / 25.4), facecolor='white')
fig.legend(handles=[Line2D([], [], color=C[c], marker=m, linestyle=l, linewidth=.8, markersize=2.5, label=lab) for _, _, _, lab, c, m, l in SERIES],
           loc='upper center', bbox_to_anchor=(.5, .995), ncol=6, frameon=False, handlelength=2.0, columnspacing=1.4, handletextpad=.5)
for (mode, title), left in zip(PANELS, (14, 100)):
    rows = list(csv.DictReader((RUN / f'summary/{mode}/timing_summary.csv').open()))
    ax = fig.add_axes([left / 183, 13 / 76, 70 / 183, 46 / 76])
    n_tmol = len({r['dataset_id'] for r in rows if r['engine'] == 'tmol' and r['status'] == 'ok'})
    for engine, device, batch, label, color, marker, ls in SERIES:
        d = [r for r in rows if r['engine'] == engine and r['device'] == device and int(r['batch_size']) == batch and r['status'] == 'ok']
        d.sort(key=lambda r: float(r['atoms']))
        if not d:
            continue
        ax.plot([float(r['atoms']) for r in d], [float(r['structures_per_second']) for r in d], color=C[color], marker=marker,
                linestyle=ls, linewidth=.8, markersize=2.5, markeredgewidth=.4, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(600, 30000)
    ax.set_xticks([1000, 10000]); ax.set_xticklabels(['1,000', '10,000'])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}' if v < 1000 else f'{v:,.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter()); ax.tick_params(which='minor', length=1, width=.25)
    ax.set_xlabel('Input atoms per structure', labelpad=3); ax.set_ylabel('Speed (structures/s)', labelpad=3)
    ax.set_title(title, pad=5); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', which='major', color=C['light_gray'], linewidth=.25, zorder=0)
    fig.text((left + 35) / 183, 65 / 76, f'tmol 0.2.0 · CPU 32 threads · {n_tmol} tmol systems', ha='center', va='center', fontsize=5.5, color=C['target_gray'])
out = ROOT / 'fastrelax_50steps'
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(f'{out}.{ext}', dpi=300, **({'metadata': {'Date': None}} if ext != 'png' else {}))
print('wrote', out)
