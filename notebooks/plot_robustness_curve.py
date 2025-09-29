import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Visual Style Configuration ---
BASELINE_COLOR = "royalblue"
DEFENDED_COLOR = "forestgreen"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# --- Load Data ---
try:
    df = pd.read_csv('../results/robustness_results.csv')
except FileNotFoundError:
    print("Error: '../results/robustness_results.csv' not found.")
    exit()

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(df['Epsilon'], df['Baseline_Accuracy'], marker='o', linestyle='--', color=BASELINE_COLOR, label='Baseline Model')
plt.plot(df['Epsilon'], df['Defended_Accuracy'], marker='s', linestyle='-', color=DEFENDED_COLOR, label='Defended Model')

plt.title('Model Accuracy vs. PGD Attack Strength (Epsilon)', fontsize=16)
plt.xlabel('Attack Strength (Epsilon ε)', fontsize=12)
plt.ylabel('Model Accuracy', fontsize=12)
plt.xticks(df['Epsilon'])
plt.ylim(-0.05, 1.05)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('../results/robustness_curve.png', dpi=300)
plt.show()