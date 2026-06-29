# ============================================================
# Tennessee Eastman Process - Fault Detection 
# ============================================================

import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import kagglehub

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import gc
import os
os.environ['OMP_NUM_THREADS'] = '4'

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Step 2: Load Data
# ============================================================
train_normal = pd.read_csv('/kaggle/input/datasets/afrniomelo/tep-csv/TEP_FaultFree_Training.csv',
                            dtype='float32')
test_normal  = pd.read_csv('/kaggle/input/datasets/afrniomelo/tep-csv/TEP_FaultFree_Testing.csv',
                            dtype='float32')
train_fault  = pd.read_csv('/kaggle/input/datasets/afrniomelo/tep-csv/TEP_Faulty_Training.csv',
                            dtype='float32')
test_fault   = pd.read_csv('/kaggle/input/datasets/afrniomelo/tep-csv/TEP_Faulty_Testing.csv',
                            dtype='float32')
print("Train Normal Shape:", train_normal.shape)
print("Train Fault Shape:", train_fault.shape)
print("Test Normal Shape:", test_normal.shape)
print("Test Fault Shape:", test_fault.shape)


# ============================================================
# Step 3: EDA
# ============================================================
print("\nFault types in training data:")
print(train_fault['faultNumber'].value_counts().sort_index())

plt.figure(figsize=(12,4))
train_fault['faultNumber'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Fault Type Distribution in Training Data')
plt.xlabel('Fault Number')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('fault_distribution.png')
plt.show()
# ============================================================
# Step 4: Subsample + Balance
# ============================================================
train_normal['faultNumber'] = 0
test_normal['faultNumber']  = 0

train_fault_sub = train_fault.groupby('faultNumber', group_keys=False).apply(
    lambda x: x.sample(n=5000, random_state=42)
).reset_index(drop=True)
train_normal_sub = train_normal.sample(n=100000, random_state=42)

test_fault_sub = test_fault.groupby('faultNumber', group_keys=False).apply(
    lambda x: x.sample(n=5000, random_state=42)
).reset_index(drop=True)
test_normal_sub = test_normal.sample(n=100000, random_state=42)

train_df = pd.concat([train_normal_sub, train_fault_sub], axis=0).reset_index(drop=True)
test_df  = pd.concat([test_normal_sub, test_fault_sub], axis=0).reset_index(drop=True)

del train_normal, train_fault, test_normal, test_fault
del train_normal_sub, train_fault_sub, test_normal_sub, test_fault_sub
gc.collect()

print("\nBalanced train shape:", train_df.shape)
print("Class distribution:\n", train_df['faultNumber'].value_counts().sort_index())

# ============================================================
# Step 5: FAST Rolling Features
# ============================================================
print("\n--- Adding rolling-window features ---")

meta_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [c for c in train_df.columns if c not in meta_cols]
print(f"Original features: {len(feature_cols)}")

def add_rolling_features_fast(df, feature_cols, window=5):
    df = df.copy()
    df = df.sort_values(['faultNumber', 'simulationRun', 'sample']).reset_index(drop=True)
    groups = df.groupby(['faultNumber', 'simulationRun']).ngroup()
    group_shift = groups != groups.shift(1)
    
    for col in feature_cols:
        df[f'{col}_rollmean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rollstd'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
        # Fix boundary contamination
        df.loc[group_shift, f'{col}_rollmean'] = df.loc[group_shift, col]
        df.loc[group_shift, f'{col}_rollstd'] = 0.0
    
    return df

train_df = add_rolling_features_fast(train_df, feature_cols, window=5)
test_df  = add_rolling_features_fast(test_df, feature_cols, window=5)

gc.collect()

drop_cols = [c for c in meta_cols if c in train_df.columns]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['faultNumber']
X_test  = test_df.drop(columns=drop_cols)
y_test  = test_df['faultNumber']

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')

print(f"Feature shape: {X_train.shape} (now includes rolling mean/std)")
print(f"Number of classes: {y_train.nunique()}")

del train_df, test_df
gc.collect()

# ============================================================
# Step 6: Feature Scaling
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

del X_train, X_test
gc.collect()


# ============================================================
# Step 7: Train Models
# ============================================================

# --- Random Forest ---
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    max_samples=0.7,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred, average='weighted')
print(f"Random Forest Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")

gc.collect()

# --- XGBoost ---
print("\n--- Training XGBoost ---")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    tree_method='hist'
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_f1   = f1_score(y_test, xgb_pred, average='weighted')
print(f"XGBoost Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f}")

gc.collect()

# ============================================================
# Step 8: Model Comparison
# ============================================================
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_acc, xgb_acc],
    'F1 Score': [rf_f1, xgb_f1]
})
print("\nModel Comparison:")
print(results)

plt.figure(figsize=(8,4))
x = np.arange(2)
width = 0.35
plt.bar(x - width/2, results['Accuracy'], width, label='Accuracy', color='steelblue')
plt.bar(x + width/2, results['F1 Score'], width, label='F1 Score', color='coral')
plt.xticks(x, results['Model'])
plt.ylim(0.6, 1.0)
plt.title('Model Comparison - TEP Fault Detection')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()


# ============================================================
# Step 9: Best Model Evaluation
# ============================================================
best_model = xgb if xgb_acc >= rf_acc else rf
best_pred  = xgb_pred if xgb_acc >= rf_acc else rf_pred
best_name  = "XGBoost" if xgb_acc >= rf_acc else "Random Forest"

print(f"\nBest Model: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

plt.figure(figsize=(14,10))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ============================================================
# Step 10: Observable Faults Only (exclude 3, 9, 15)
# ============================================================
UNOBSERVABLE = [3.0, 9.0, 15.0]

mask_test = ~y_test.isin(UNOBSERVABLE)
y_test_obs = y_test[mask_test]
obs_pred   = best_pred[mask_test.values]

obs_acc = accuracy_score(y_test_obs, obs_pred)
obs_f1  = f1_score(y_test_obs, obs_pred, average='weighted')

print("\n" + "="*50)
print("OBSERVABLE FAULTS ONLY (excluding 3, 9, 15)")
print("="*50)
print(f"Accuracy: {obs_acc:.4f}")
print(f"F1 Score: {obs_f1:.4f}")
print(f"Samples evaluated: {len(y_test_obs)}")
print("="*50)

print("\nPer-fault accuracy on observable faults:")
for fault in sorted(y_test_obs.unique()):
    fault_mask = y_test_obs == fault
    fault_acc = accuracy_score(y_test_obs[fault_mask], obs_pred[fault_mask])
    print(f"  Fault {int(fault):2d}: {fault_acc:.4f}")


# ============================================================
# Step 11: Feature Importance
# ============================================================
importances = best_model.feature_importances_
feature_names_saved = [f"f{i}" for i in range(X_train_scaled.shape[1])]

# Get actual feature names from before scaling
# We need to reconstruct - use the column names we saved
# Actually, let's just use indices
top_idx = np.argsort(importances)[-10:]

print("\nTop 10 Most Important Features:")
for idx in top_idx[::-1]:
    print(f"  Feature {idx}: {importances[idx]:.4f}")

# ============================================================
# Step 12: SHAP Explainability
# ============================================================
print("\n--- Computing SHAP Values ---")

X_test_sample = pd.DataFrame(X_test_scaled[:500], 
                              columns=[f"f{i}" for i in range(X_train_scaled.shape[1])])

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_sample)

plt.figure()
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Global")
plt.tight_layout()
plt.savefig('shap_global.png')
plt.show()

plt.figure()
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title("SHAP Beeswarm Plot")
plt.tight_layout()
plt.savefig('shap_beeswarm.png')
plt.show()

print("SHAP analysis complete.")

# ============================================================
# Step 13: Fault Detection Delay Analysis
# ============================================================
print("\n--- Fault Detection Delay Analysis ---")

FAULT_INTRO_SAMPLE = 160
delay_results = []

for fault_num in sorted(y_test.unique()):
    if fault_num in UNOBSERVABLE:
        delay_results.append({
            'Fault': fault_num,
            'Detected': False,
            'Detection Delay (samples)': 'Unobservable'
        })
        continue
    
    fault_mask = y_test == fault_num
    fault_preds = best_pred[fault_mask.values]
    
    # In our subsampled data, fault samples start after sample 160
    # Find first correct prediction
    detected = False
    detection_delay = None
    for i in range(len(fault_preds)):
        if fault_preds[i] == fault_num:
            # Approximate: samples before FAULT_INTRO are pre-fault
            detection_delay = i
            detected = True
            break
    
    delay_results.append({
        'Fault': fault_num,
        'Detected': detected,
        'Detection Delay (samples)': detection_delay if detected else 'Not Detected'
    })

delay_df = pd.DataFrame(delay_results)
print("\nFault Detection Delay Results:")


# ============================================================
# Graph: Fault Detection Delay
# ============================================================

# Filter out Normal (Fault 0) and Unobservable faults (3, 9, 15) for a clean graph
plot_df = delay_df[
    (delay_df['Fault'] != 0.0) & 
    (delay_df['Detected'] == True)
].copy()

# Ensure delay is integer for plotting
plot_df['Detection Delay (samples)'] = plot_df['Detection Delay (samples)'].astype(int)

plt.figure(figsize=(12, 5))
bars = plt.bar(
    plot_df['Fault'].astype(int).astype(str), 
    plot_df['Detection Delay (samples)'], 
    color='steelblue', 
    edgecolor='black'
)

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        yval + 0.5, 
        f'{int(yval)}', 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        fontweight='bold'
    )

plt.title('Fault Detection Delay by Fault Type', fontsize=14, fontweight='bold')
plt.xlabel('Fault Number', fontsize=12)
plt.ylabel('Detection Delay (samples)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('detection_delay.png', dpi=150)
plt.show()


print(delay_df.to_string(index=False))

# ============================================================
# Final Summary
# ============================================================
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"Dataset:            Tennessee Eastman Process (TEP)")
print(f"Fault Types:        21 total (18 observable)")
print(f"Training Samples:   {X_train_scaled.shape[0]}")
print(f"Test Samples:       {X_test_scaled.shape[0]}")
print(f"Features:           {X_train_scaled.shape[1]} (incl. rolling mean/std)")
print(f"Best Model:         {best_name}")
print(f"All-Fault Accuracy: {max(rf_acc, xgb_acc):.4f}")
print(f"Observable Accuracy:{obs_acc:.4f}")
print(f"Observable F1:      {obs_f1:.4f}")
print("="*50)

