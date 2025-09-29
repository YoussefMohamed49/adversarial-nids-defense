import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Visual Style Configuration ---
# Use a consistent, professional color palette and theme
BASELINE_COLOR = "royalblue"
DEFENDED_COLOR = "forestgreen"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- Data Preparation ---
# This data comes from your initial full experiment run.
data = {
    'Model': ['Baseline', 'Defended', 'Baseline', 'Defended', 'Baseline', 'Defended'],
    'Condition': ['Clean Data', 'Clean Data', 'FGSM Attack', 'FGSM Attack', 'PGD Attack', 'PGD Attack'],
    'Accuracy': [0.76, 0.78, 0.21, 0.77, 0.11, 0.78]
}
df = pd.DataFrame(data)

# --- Plotting ---
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Condition', y='Accuracy', hue='Model', data=df, palette=[BASELINE_COLOR, DEFENDED_COLOR])

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.title('Model Accuracy Comparison: Baseline vs. Defended', fontsize=16)
plt.xlabel('Test Data Condition', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)
plt.legend(title='Model Type')
plt.tight_layout()
plt.savefig('../results/model_performance_comparison.png', dpi=300)
plt.show()