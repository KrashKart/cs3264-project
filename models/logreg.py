import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    f1_score, balanced_accuracy_score  , recall_score          
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import joblib
# from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------

df = pd.read_csv('diabetes_prediction_dataset.csv')
aug = pd.read_csv('datasets\synthetic_diabetes_data_positive.csv')

df = pd.concat([df, aug], ignore_index=True)

print(f"Dataset shape: {df.shape}")
print("\nClass balance:\n", df['diabetes'].value_counts(normalize=True))

# ---------------------------------------------

df['gender_enc']  = LabelEncoder().fit_transform(df['gender'])
df['smoking_enc'] = LabelEncoder().fit_transform(df['smoking_history'])

# FEATURE ENGINEERING 

# Clinical threshold flags 
df['hba1c_diabetic']   = (df['HbA1c_level'] >= 6.5).astype(int)
df['hba1c_prediab']    = ((df['HbA1c_level'] >= 5.7) & (df['HbA1c_level'] < 6.5)).astype(int)
df['glucose_high']     = (df['blood_glucose_level'] >= 126).astype(int)
df['glucose_very_high'] = (df['blood_glucose_level'] >= 200).astype(int)

# both clinical thresholds exceeded simultaneously
df['both_markers_high'] = (
    (df['HbA1c_level'] >= 5.7) & (df['blood_glucose_level'] >= 126)
).astype(int)

# BMI categories (WHO classification)
df['bmi_obese']      = (df['bmi'] >= 30).astype(int)
df['bmi_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)

# Age risk groups
df['age_high_risk']   = (df['age'] >= 45).astype(int)
df['age_senior']      = (df['age'] >= 60).astype(int)

# Cumulative risk score: sum of all risk factors
df['risk_score'] = (
    df['hba1c_diabetic'] +
    df['hba1c_prediab'] +
    df['glucose_high'] +
    # df['bmi_overweight'] +
    # df['age_high_risk'] +
    df['hypertension'] +
    df['heart_disease']
)

FEATURES = [
    'gender_enc', 'age', 'hypertension', 'heart_disease',
    'smoking_enc', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'hba1c_prediab', #'hba1c_diabetic'
    # 'glucose_high',
    'both_markers_high',
    # 'bmi_obese', 'bmi_overweight',
    # 'age_high_risk', 'age_senior',
    # 'risk_score'
]

TARGET = 'diabetes'

X = df[FEATURES].values
y = df[TARGET].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}")


scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_sc = poly.fit_transform(X_train_sc)
X_test_sc  = poly.transform(X_test_sc)

# print(f"Features after polynomial expansion: {X_train_sc.shape[1]}")


TOTAL_EPOCHS = 50
LOG_INTERVAL = 1
BATCH_SIZE   = 256
CLASSES      = np.array([0, 1])

weights           = compute_class_weight('balanced', classes=CLASSES, y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}

model = SGDClassifier(
    loss='log_loss',
    class_weight=class_weight_dict,       
    learning_rate='optimal',
    penalty='l2',        
    alpha=0.001,
    random_state=42
)

train_f1, test_f1         = [], []
train_bacc, test_bacc     = [], []
train_losses, test_losses = [], []
log_epochs                = []
train_recall, test_recall = [], []
rng = np.random.default_rng(42)

print("\n-- Training Log -----------------------------------------------------------------------")
print(f"{'Epoch':>6}  {'Tr F1':>8}  {'Te F1':>8}  {'Tr Acc':>10}  {'Te Acc':>10}  {'Tr Loss':>8}  {'Te Loss':>8}")
print("-" * 76)

for epoch in range(1, TOTAL_EPOCHS + 1):
    idx = rng.permutation(len(X_train_sc))
    for start in range(0, len(X_train_sc), BATCH_SIZE):
        batch = idx[start:start + BATCH_SIZE]
        model.partial_fit(X_train_sc[batch], y_train[batch], classes=CLASSES)

    if epoch % LOG_INTERVAL == 0:
        tr_pred = model.predict(X_train_sc)
        te_pred = model.predict(X_test_sc)
        tr_prob = model.predict_proba(X_train_sc)
        te_prob = model.predict_proba(X_test_sc)

        tr_f1_  = f1_score(y_train, tr_pred)
        te_f1_  = f1_score(y_test,  te_pred)
        tr_rec  = recall_score(y_train, tr_pred)
        te_rec  = recall_score(y_test,  te_pred)
        
        tr_ba   = accuracy_score(y_train, tr_pred)
        te_ba   = accuracy_score(y_test,  te_pred)
        tr_loss = log_loss(y_train, tr_prob)
        te_loss = log_loss(y_test,  te_prob)

        train_f1.append(tr_f1_);    test_f1.append(te_f1_)
        train_bacc.append(tr_ba);   test_bacc.append(te_ba)
        train_losses.append(tr_loss); test_losses.append(te_loss)
        train_recall.append(tr_rec)
        test_recall.append(te_rec)
        log_epochs.append(epoch)
        print(f"{epoch:>6}  {tr_f1_:>8.4f}  {te_f1_:>8.4f}  {tr_ba:>10.4f}  {te_ba:>10.4f}  {tr_loss:>8.4f}  {te_loss:>8.4f}")


y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\n-- Final Metrics --------------------------------")
print(f"Test Accuracy          : {accuracy_score(y_test, y_pred):.4f}  <- misleading due to imbalance")
print(f"Test F1 (diabetic)     : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC                : {roc_auc_score(y_test, y_prob):.4f}")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))   # was (1, 3)
fig.suptitle('Logistic Regression (SGD) - Diabetes Prediction',
             fontsize=13, fontweight='bold')

# F1 Score
axes[0,0].plot(log_epochs, train_f1, marker='o', color='steelblue',
               linewidth=2, markersize=4, label='Train F1')
axes[0,0].plot(log_epochs, test_f1,  marker='s', color='tomato',
               linewidth=2, markersize=4, label='Test F1')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('F1 Score')
axes[0,0].set_title('F1 Score - Diabetic Class')
axes[0,0].set_xticks(range(10, TOTAL_EPOCHS + 1, 10))
axes[0,0].legend()
axes[0,0].grid(True, linestyle='--', alpha=0.5)

# Accuracy
axes[0,1].plot(log_epochs, train_bacc, marker='o', color='steelblue',
               linewidth=2, markersize=4, label='Train Accuracy')
axes[0,1].plot(log_epochs, test_bacc,  marker='s', color='tomato',
               linewidth=2, markersize=4, label='Test Accuracy')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].set_title('Accuracy over Epochs')
axes[0,1].set_xticks(range(10, TOTAL_EPOCHS + 1, 10))
axes[0,1].legend()
axes[0,1].grid(True, linestyle='--', alpha=0.5)

# Log Loss
axes[1,0].plot(log_epochs, train_losses, marker='o', color='steelblue',
               linewidth=2, markersize=4, label='Train Loss')
axes[1,0].plot(log_epochs, test_losses,  marker='s', color='tomato',
               linewidth=2, markersize=4, label='Test Loss')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Log Loss')
axes[1,0].set_title('Log Loss over Epochs')
axes[1,0].set_xticks(range(10, TOTAL_EPOCHS + 1, 10))
axes[1,0].legend()
axes[1,0].grid(True, linestyle='--', alpha=0.5)

# Recall on diabetic class
axes[1,1].plot(log_epochs, train_recall, marker='o', color='steelblue',
               linewidth=2, markersize=4, label='Train Recall')
axes[1,1].plot(log_epochs, test_recall,  marker='s', color='tomato',
               linewidth=2, markersize=4, label='Test Recall')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Recall')
axes[1,1].set_title('Diabetes Caught')
axes[1,1].set_xticks(range(10, TOTAL_EPOCHS + 1, 10))
axes[1,1].legend()
axes[1,1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved -> logistic_regression_results.png")

print(f"Test Accuracy          : {accuracy_score(y_test, y_pred):.4f}")   
print(f"Test F1 (diabetic)     : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC                : {roc_auc_score(y_test, y_prob):.4f}")  
print(f"Test Recall (diabetic)  : {recall_score(y_test, y_pred):.4f}")


feature_names = poly.get_feature_names_out(FEATURES)
coefs = pd.Series(model.coef_[0], index=feature_names)

base_coefs = coefs[[f for f in coefs.index if ' ' not in f]].sort_values(ascending=False)
print("\n-- Feature Contributions (base features) --------")
print(base_coefs)

fig, ax = plt.subplots(figsize=(8, 5))
norm   = mcolors.Normalize(vmin=base_coefs.min(), vmax=base_coefs.max())
colors = [cm.coolwarm(norm(c)) for c in base_coefs]
ax.barh(base_coefs.index, base_coefs.values, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature Coefficients')
ax.set_xlabel('Coefficient value')
plt.tight_layout()
plt.savefig('feature_contributions.png', dpi=150, bbox_inches='tight')
plt.show()

joblib.dump({
    'model': model,
    'scaler': scaler,
    'poly': poly,
    'features': FEATURES,
}, 'models\logreg_model.joblib')
