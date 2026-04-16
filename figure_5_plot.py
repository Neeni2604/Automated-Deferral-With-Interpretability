import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('white')

models = {
    '1B': {
        'title': 'Gemma 3 1B · Layer 17',
        'title_color': '#5F5E5A',
        'title_fill': '#D3D1C7',
        'clusters': [
            {
                'label': 'C0', 'n': 83,
                'color': '#B5D4F4', 'border': '#185FA5',
                'features': [
                    'Contrastive language',
                    '(unlike, compared, despite)',
                    'Necessity / obligation',
                    '(need, must, can)',
                ]
            },
            {
                'label': 'C1', 'n': 113,
                'color': '#FAC775', 'border': '#BA7517',
                'features': [
                    'Exam / quiz format',
                    'Boolean language (true/false)',
                    'Multiple choice options',
                    'Markdown formatting tokens',
                    'Opening parentheses',
                ]
            },
            {
                'label': 'C2', 'n': 69,
                'color': '#C0DD97', 'border': '#3B6D11',
                'features': [
                    'Superlatives (most, greatest, best)',
                    'Foreign language prepositions',
                    'Contrastive language (shared)',
                    'Definite article variants',
                ]
            },
        ]
    },
    '4B': {
        'title': 'Gemma 3 4B · Layer 17',
        'title_color': '#5F5E5A',
        'title_fill': '#D3D1C7',
        'clusters': [
            {
                'label': 'C0', 'n': 66,
                'color': '#B5D4F4', 'border': '#185FA5',
                'features': [
                    'Requiring energy or help',
                    'Parsing and filtering data',
                    'Hedging / uncertainty language',
                    'Properties and significance',
                ]
            },
            {
                'label': 'C1', 'n': 78,
                'color': '#FAC775', 'border': '#BA7517',
                'features': [
                    'Avoidance / prevention language',
                    'Prohibition (forbid, unsuitable)',
                    'Words after "from"',
                    'Common phrase endings',
                    'Requiring energy or help (shared)',
                ]
            },
            {
                'label': 'C2', 'n': 16,
                'color': '#C0DD97', 'border': '#3B6D11',
                'features': [
                    'Positive / celebratory language',
                    'Improvement language',
                    'Multilingual texts (shared)',
                    'Compound word suffixes',
                ]
            },
        ]
    }
}

cluster_heights = [0.22, 0.27, 0.22]
cluster_top = 0.86
gap = 0.02

for ax, (model_key, model_data) in zip(axes, models.items()):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.add_patch(FancyBboxPatch((0.02, 0.90), 0.96, 0.075,
                                boxstyle='round,pad=0.01',
                                facecolor=model_data['title_fill'],
                                edgecolor=model_data['title_color'],
                                linewidth=1.0,
                                clip_on=False))
    ax.text(0.5, 0.940, model_data['title'],
            ha='center', va='center', fontsize=15, fontweight='bold',
            color='black', transform=ax.transAxes)

    ax.text(0.13, 0.879, 'Cluster', ha='center', va='center',
            fontsize=8.75, color='black', transform=ax.transAxes)
    ax.text(0.6, 0.879, 'Top feature descriptions', ha='center', va='center',
            fontsize=8.75, color='black', transform=ax.transAxes)

    y = cluster_top
    for i, cluster in enumerate(model_data['clusters']):
        h = cluster_heights[i]
        label_x, label_w = 0.02, 0.22
        feat_x, feat_w = 0.26, 0.72

        ax.add_patch(FancyBboxPatch((label_x, y - h), label_w, h,
                                    boxstyle='round,pad=0.01',
                                    facecolor=cluster['color'],
                                    edgecolor=cluster['border'],
                                    linewidth=1.0))
        ax.text(label_x + label_w / 2, y - h / 2 + 0.025,
                cluster['label'], ha='center', va='center',
                fontsize=15, fontweight='bold',
                color='black', transform=ax.transAxes)
        ax.text(label_x + label_w / 2, y - h / 2 - 0.025,
                f"n={cluster['n']}", ha='center', va='center',
                fontsize=11, color='black',
                alpha=0.75, transform=ax.transAxes)

        ax.add_patch(FancyBboxPatch((feat_x, y - h), feat_w, h,
                                    boxstyle='round,pad=0.01',
                                    facecolor=cluster['color'],
                                    edgecolor=cluster['border'],
                                    linewidth=1.0))
        n_feats = len(cluster['features'])
        for j, feat in enumerate(cluster['features']):
            fy = y - (j + 1) * h / (n_feats + 1)
            ax.text(feat_x + 0.03, fy, feat,
                    ha='left', va='center', fontsize=15,
                    color='black', transform=ax.transAxes)

        y -= h + gap

fig.text(0.5, 0.055,
         'Notable finding: 1B Cluster 1 activates QA-format features (exam, quiz, true/false, multiple choice),\n'
         'suggesting the model pattern-matches to a Q&A format rather than reasoning over legal text.\n'
         '4B Cluster 1 activates prohibition language — a semantically relevant failure mode for NDA clauses.',
         ha='center', va='center', fontsize=9.5,
         color='black',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#FAC775',
                   edgecolor='#BA7517', linewidth=0.8))

fig.suptitle('Failure Cluster Profiles — Layer 17 (top SAE feature descriptions per cluster)',
             fontsize=14, fontweight='bold', y=0.995, color='black')

plt.tight_layout(rect=[0, 0.12, 1, 0.99])
plt.savefig('cluster_profiles.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()