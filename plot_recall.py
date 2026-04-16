import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SYSTEMS   = ['SAE LogReg', 'SAE MLP', 'SAE RF', 'Cluster Membership', 'Confidence', 'Random']
COLORS    = ['#1d4ed8', '#16a34a', '#ea580c', '#7c3aed', '#dc2626', '#9ca3af']
COVERAGES = [0.2, 0.3, 0.4]

TOTAL_ERRORS = {'1B': 55, '4B': 32}

CONFIGS = ['1B-L7', '1B-L17', '1B-L22', '4B-L9', '4B-L17', '4B-L29']
CONFIG_DIRS = ['1B7', '1B17', '1B22', '4B9', '4B17', '4B29']

def load_results(config_dir):
    with open(f'data/{config_dir}/phase3_results.json') as f:
        p3 = json.load(f)
    with open(f'data/{config_dir}/phase4_results.json') as f:
        p4 = json.load(f)
    return p3 + p4['deferral_results']

def compute_recall(results, total_errors):
    recall = {sys: [] for sys in SYSTEMS}
    for cov in COVERAGES:
        seen = {sys: False for sys in SYSTEMS}
        for entry in results:
            sys = entry['name']
            if entry['coverage'] == cov and sys in SYSTEMS and not seen[sys]:
                tp = entry['precision'] * entry['n_deferred']
                recall[sys].append(tp / total_errors)
                seen[sys] = True
    return recall

RECALL = {}
for cfg, cfg_dir in zip(CONFIGS, CONFIG_DIRS):
    model = '1B' if '1B' in cfg else '4B'
    results = load_results(cfg_dir)
    RECALL[cfg] = compute_recall(results, TOTAL_ERRORS[model])

model_colors = ['#1d4ed8'] * 3 + ['#16a34a'] * 3

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
fig.suptitle('Recall of Error Class across Coverage Levels',
             fontweight='bold', y=1.01, fontfamily='monospace')

for idx, (cfg, ax) in enumerate(zip(CONFIGS, axes.flat)):
    for sys, col in zip(SYSTEMS, COLORS):
        vals = RECALL[cfg].get(sys, [])
        if vals:
            ax.plot(COVERAGES, vals, marker='o', markersize=5,
                    color=col, linewidth=1.8, label=sys)
    ax.set_title(cfg, fontsize=13, fontweight='bold', color=model_colors[idx])
    ax.set_xticks(COVERAGES)
    ax.set_xticklabels(['20%', '30%', '40%'])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Coverage')
    if idx % 3 == 0:
        ax.set_ylabel('Recall')

handles = [mpatches.Patch(color=c, label=s) for s, c in zip(SYSTEMS, COLORS)]
fig.legend(handles=handles, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, -0.06), framealpha=0.9, edgecolor='#ddd')
plt.tight_layout()
plt.savefig('recall_figure.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()