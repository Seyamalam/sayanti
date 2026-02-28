import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

# Read data
df = pd.read_csv('data/fraudTest.csv')

# Output some stats
print(f"Total rows: {len(df)}")
print(f"Fraud count: {df['is_fraud'].sum()}")
print(f"Fraud percentage: {df['is_fraud'].mean() * 100:.3f}%")

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Class Imbalance (Log Scale)
sns.countplot(data=df, x='is_fraud', ax=axes[0], palette=['#2c7bb6', '#d7191c'])
axes[0].set_yscale('log')
axes[0].set_title('A. Extreme Class Imbalance (Log Scale)')
axes[0].set_xlabel('Transaction Class (0 = Legitimate, 1 = Fraud)')
axes[0].set_ylabel('Count (Log Scale)')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Legitimate\n(99.6%)', 'Fraud\n(0.39%)'])

# Plot 2: Transaction Amount Distribution
df['log_amt'] = np.log1p(df['amt'])
sns.kdeplot(data=df[df['is_fraud']==0], x='log_amt', ax=axes[1], color='#2c7bb6', label='Legitimate', fill=True, alpha=0.3)
sns.kdeplot(data=df[df['is_fraud']==1], x='log_amt', ax=axes[1], color='#d7191c', label='Fraud', fill=True, alpha=0.3)
axes[1].set_title('B. Transaction Amount Density')
axes[1].set_xlabel('Log(Transaction Amount)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
os.makedirs('manuscript', exist_ok=True)
plt.savefig('manuscript/eda_plots.pdf', dpi=300)
plt.close()
