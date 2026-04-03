"""Generate model comparison chart from actual Kaggle results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

datasets = ['CIC-IDS-2017', 'UNSW-NB15', 'CSE-CIC-IDS-2018']
models = ['XGBoost', 'RandomForest', 'LightGBM', 'VotingEnsemble']

accuracy = {
    'CIC-IDS-2017': [0.9966, 0.9857, 0.9744, 0.9886],
    'UNSW-NB15': [0.8004, 0.7635, 0.7630, 0.7867],
    'CSE-CIC-IDS-2018': [0.9990, 0.9993, 0.9990, 0.9993],
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (dataset, ax) in enumerate(zip(datasets, axes)):
    x = np.arange(len(models))
    bars = ax.barh(x, accuracy[dataset], color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=11, fontweight='500')
    ax.set_xlim(0.7 if dataset == 'CSE-CIC-IDS-2018' else 0.5, 1.0)
    ax.set_xlabel('Accuracy', fontsize=10, fontweight='500')
    ax.set_title(dataset, fontsize=13, fontweight='700', pad=12)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#e0e0e0')
    ax.tick_params(axis='x', colors='#666')
    ax.tick_params(axis='y', length=0)
    for bar, val in zip(bars, accuracy[dataset]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left',
                fontsize=10, fontweight='600', color='#333')

plt.suptitle('Model Accuracy Comparison Across Datasets',
             fontsize=16, fontweight='700', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved outputs/figures/model_comparison.png')
