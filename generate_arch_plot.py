import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import os

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis('off')

# Set global aesthetics
plt.rcParams['font.family'] = 'serif'

def draw_box(ax, x, y, width, height, text, color, edgecolor='#333333', title=None):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05,rounding_size=0.15", 
                         fc=color, ec=edgecolor, lw=1.5, alpha=0.95)
    ax.add_patch(box)
    if title:
        ax.text(x + width/2, y + height - 0.35, title, ha='center', va='center', 
                fontsize=13, fontweight='bold', family='serif')
        ax.text(x + width/2, y + height/2 - 0.25, text, ha='center', va='center', 
                fontsize=11.5, family='serif', linespacing=1.6)
    else:
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=12, fontweight='bold', family='serif')

# Framework Background Box
framework_box = FancyBboxPatch((0.2, 0.2), 12.6, 4.0, boxstyle="round,pad=0.05,rounding_size=0.2", 
                               fc="#f8f9fa", ec="#adb5bd", lw=2, linestyle='--')
ax.add_patch(framework_box)
ax.text(6.5, 3.8, "Imbalance-Aware Explainable Fraud Detection (IAE-FD) Framework", 
        ha='center', va='center', fontsize=15, fontweight='bold', family='serif', color="#212529")

# Top Row Boxes (Data Pipeline)
draw_box(ax, 0.5, 5.2, 3.2, 1.3, "Transaction Data\n(0.386% Frauds)", "#e0f3db")
draw_box(ax, 4.9, 5.2, 3.2, 1.3, "ML Models\n(XGBoost, RF, LightGBM)", "#fee0d2")
draw_box(ax, 9.3, 5.2, 3.2, 1.3, "Predictions &\nLog-Odds Margins", "#fff2ae")

# Top Row Arrows
ax.annotate('', xy=(4.9, 5.85), xytext=(3.7, 5.85), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#495057", mutation_scale=20))
ax.annotate('', xy=(9.3, 5.85), xytext=(8.1, 5.85), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#495057", mutation_scale=20))

# Bottom Row Boxes (Framework Components)
draw_box(ax, 0.6, 0.5, 3.6, 2.7, "• SLSQP Continuous Optimization\n• L1 Sparsity Penalty\n• Structural Equality Bounds", "#e5f5e0", title="CC-Opt\n(Recourse Generator)")
draw_box(ax, 4.7, 0.5, 3.6, 2.7, "• 50/50 Bipartite Background\n• Calibrated Expectations\n• Resolves Demographic Bias", "#e5f5e0", title="IC-SHAP\n(Causal Attributions)")
draw_box(ax, 8.8, 0.5, 3.6, 2.7, "• Log-Odds to Prob Mapping\n• Random-Walk Correlation\n• Fidelity & Stability Auditing", "#e5f5e0", title="EQA\n(Quality Auditor)")

# Vertical/Diagonal Connections into Framework

# Data -> CC-Opt
ax.annotate('', xy=(1.5, 3.2), xytext=(1.5, 5.2), arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="dashed", color="#495057", mutation_scale=15))
ax.text(1.55, 4.3, 'Original Instance', rotation=90, va='center', fontsize=10, family='serif', color='#495057')

# Model -> IC-SHAP
ax.annotate('', xy=(6.5, 3.2), xytext=(6.5, 5.2), arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="dashed", color="#495057", mutation_scale=15))
ax.text(6.55, 4.3, 'Decision Boundaries', rotation=90, va='center', fontsize=10, family='serif', color='#495057')

# Preds -> EQA
ax.annotate('', xy=(10.6, 3.2), xytext=(10.6, 5.2), arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="dashed", color="#495057", mutation_scale=15))
ax.text(10.65, 4.3, 'Log-Odds Margins', rotation=90, va='center', fontsize=10, family='serif', color='#495057')

# Preds -> CC-Opt (Target adjustment)
ax.annotate('', xy=(3.5, 3.2), xytext=(9.7, 5.2), arrowprops=dict(arrowstyle="-|>", lw=2, linestyle="dashed", color="#495057", mutation_scale=15))
ax.text(6.6, 4.5, 'Adversarial Gradient Signal', rotation=-14, va='center', ha='center', fontsize=10, family='serif', color='#495057', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

plt.tight_layout()
os.makedirs('manuscript', exist_ok=True)
plt.savefig('manuscript/architecture.pdf', dpi=300, bbox_inches='tight')
plt.close()
