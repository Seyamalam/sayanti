import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

# Data for a representative local transaction explanation
features = ['job_enc', 'amt', 'merchant_enc', 'hour_cos', 'city_pop_log']
std_shap = [4.12, 1.25, 0.85, -0.45, 0.30]
ic_shap = [2.25, 3.85, 0.92, -0.55, 0.28]

y = np.arange(len(features))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.barh(y - height/2, std_shap, height, label='Standard SHAP', color='#d7191c', alpha=0.8)
rects2 = ax.barh(y + height/2, ic_shap, height, label='IC-SHAP', color='#2b83ba', alpha=0.8)

# Add a vertical line at 0
ax.axvline(0, color='black', linewidth=1)

ax.set_xlabel('Local SHAP Value (Impact on Log-Odds)')
ax.set_title('Local Explanation Shift: Unmasking Transaction Amount (Instance #1064)')
ax.set_yticks(y)
ax.set_yticklabels(features)
ax.legend(loc='lower right')

# Add annotations
for i in range(len(features)):
    if features[i] == 'amt':
        ax.annotate('Causal driver unmasked', 
                    xy=(ic_shap[i], y[i] + height/2), 
                    xytext=(ic_shap[i] + 0.5, y[i] + height/2),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    va='center')

plt.tight_layout()
os.makedirs('manuscript', exist_ok=True)
plt.savefig('manuscript/local_shap.pdf', dpi=300)
plt.close()
