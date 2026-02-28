import pandas as pd
import numpy as np
import dice_ml
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('data/fraudTest.csv')

# Use a manageable subset for speed but keep class imbalance
# 20000 legit, 500 fraud
df_sample = pd.concat([
    df[df['is_fraud'] == 0].sample(20000, random_state=42),
    df[df['is_fraud'] == 1].sample(500, random_state=42)
])

# Engineer features same as xai_fraud_detection.py
df_sample['trans_date_trans_time'] = pd.to_datetime(df_sample['trans_date_trans_time'])
df_sample['hour'] = df_sample['trans_date_trans_time'].dt.hour
df_sample['day_of_week'] = df_sample['trans_date_trans_time'].dt.dayofweek
df_sample['month'] = df_sample['trans_date_trans_time'].dt.month
df_sample['is_weekend'] = (df_sample['day_of_week'] >= 5).astype(int)
df_sample['hour_sin'] = np.sin(2 * np.pi * df_sample['hour'] / 24)
df_sample['hour_cos'] = np.cos(2 * np.pi * df_sample['hour'] / 24)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df_sample['distance'] = haversine(df_sample['lat'], df_sample['long'], df_sample['merch_lat'], df_sample['merch_long'])
df_sample['amt_log'] = np.log1p(df_sample['amt'])

df_sample['dob'] = pd.to_datetime(df_sample['dob'], format='%d-%m-%Y', errors='coerce')
df_sample['age'] = ((df_sample['trans_date_trans_time'] - df_sample['dob']).dt.days // 365).clip(18, 100)
df_sample['city_pop_log'] = np.log1p(df_sample['city_pop'])
df_sample['gender_enc'] = (df_sample['gender'] == 'M').astype(int)

# Target encode merchant and job (using full sample means for simplicity)
merch_means = df_sample.groupby('merchant')['is_fraud'].mean()
job_means = df_sample.groupby('job')['is_fraud'].mean()
df_sample['merchant_enc'] = df_sample['merchant'].map(merch_means).fillna(0)
df_sample['job_enc'] = df_sample['job'].map(job_means).fillna(0)

features = [
    'amt', 'amt_log', 'hour', 'hour_sin', 'hour_cos',
    'day_of_week', 'is_weekend', 'month', 'distance',
    'age', 'city_pop_log', 'merchant_enc', 'job_enc', 'gender_enc'
]

X = df_sample[features].fillna(0)
y = df_sample['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest...")
scale = (y_train == 0).sum() / max(1, (y_train == 1).sum())
model = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight={0: 1, 1: scale}, random_state=42)
model.fit(X_train, y_train)

# Setup DiCE
print("Setting up DiCE...")
# DiCE requires a dataframe with the target variable for the Data object
train_dataset = X_train.copy()
train_dataset['is_fraud'] = y_train

d = dice_ml.Data(dataframe=train_dataset, continuous_features=features, outcome_name='is_fraud')
m = dice_ml.Model(model=model, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")

# Sample 50 fraud cases from test set
fraud_cases = X_test[y_test == 1].head(50)

print(f"Generating counterfactuals for {len(fraud_cases)} cases...")
success_count = 0
total_changes = []

for i, (_, row) in enumerate(fraud_cases.iterrows()):
    query_instance = pd.DataFrame([row])
    try:
        # Generate 1 counterfactual
        dice_exp = exp.generate_counterfactuals(
            query_instance, 
            total_CFs=1, 
            desired_class=0,
            features_to_vary=features
        )
        
        if dice_exp.cf_examples_list[0].final_cfs_df is not None:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df.iloc[0]
            # Count changes
            changes = 0
            for f in features:
                if abs(cf_df[f] - row[f]) > 1e-4:
                    changes += 1
            success_count += 1
            total_changes.append(changes)
    except Exception as e:
        # DiCE failed to find CF
        pass

success_rate = (success_count / len(fraud_cases)) * 100
avg_changes = np.mean(total_changes) if total_changes else 0

print(f"\nDiCE Baseline Results:")
print(f"Success Rate: {success_rate:.1f}%")
print(f"Average Feature Changes: {avg_changes:.2f}")

results = {
    'method': 'DiCE (Random)',
    'success_rate': float(success_rate),
    'avg_changes': float(avg_changes)
}

with open('dice_baseline.json', 'w') as f:
    json.dump(results, f)
