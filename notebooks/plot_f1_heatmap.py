import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Visual Style Configuration ---
sns.set_theme(style="white", context="paper", font_scale=1.2)

# --- Data Preparation ---
data = {
    'Condition': ['Baseline on Clean Data', 'Baseline on FGSM Attack', 'Baseline on PGD Attack',
                  'Defended on Clean Data', 'Defended on FGSM Attack', 'Defended on PGD Attack'],
    'dos':   [0.86, 0.16, 0.14, 0.89, 0.87, 0.87],
    'normal':[0.79, 0.21, 0.03, 0.80, 0.79, 0.80],
    'probe': [0.78, 0.10, 0.11, 0.76, 0.74, 0.79],
    'r2l':   [0.18, 0.46, 0.31, 0.22, 0.23, 0.19],
    'u2r':   [0.11, 0.00, 0.00, 0.10, 0.02, 0.01]
}
df = pd.DataFrame(data).set_index('Condition')

# --- Plotting ---
plt.figure(figsize=(12, 8))
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=1.5, linecolor='white', cbar_kws={'label': 'F1-Score'})
ax.set_title('Per-Class F1-Score Under Different Attack Scenarios', fontsize=16, pad=20)
ax.set_xlabel('Traffic Class', fontsize=12)
ax.set_ylabel('Model and Condition', fontsize=12)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/f1_score_heatmap.png', dpi=300)
plt.show()