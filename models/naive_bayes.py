import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss,
    f1_score, recall_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('data/diabetes_prediction_dataset.csv')
aug = pd.read_csv('data/synthetic_diabetes_data_positive.csv')

df = pd.concat([df, aug], ignore_index=True)

print(f"Dataset shape: {df.shape}")
print("\nClass balance:\n", df['diabetes'].value_counts(normalize=True))


df['gender_enc']  = LabelEncoder().fit_transform(df['gender'])
df['smoking_enc'] = LabelEncoder().fit_transform(df['smoking_history'])

FEATURES = ['gender_enc', 'age', 'hypertension', 'heart_disease',
            'smoking_enc', 'bmi', 'HbA1c_level', 'blood_glucose_level']
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


CLASSES    = np.array([0, 1])
N_CHUNKS   = 100       
chunk_size = len(X_train_sc) // N_CHUNKS

# Naive Bayes uses prior probabilities instead of class weights.
# We set class_prior manually to handle imbalance.
diabetic_prior    = y_train.mean()
non_diabetic_prior = 1 - diabetic_prior
class_prior = [non_diabetic_prior, diabetic_prior]  
print(f"\nClass priors -> 0: {non_diabetic_prior:.3f}  |  1: {diabetic_prior:.3f}")

model = GaussianNB(var_smoothing=1e-9)

train_f1, test_f1         = [], []
train_bacc, test_bacc     = [], []
train_losses, test_losses = [], []
train_recall, test_recall = [], []
log_chunks                = []

rng = np.random.default_rng(42)
idx = rng.permutation(len(X_train_sc))   # shuffle once before chunking

print("\n-- Training Log -----------------------------------------------------------------------")
print(f"{'Chunk':>6}  {'Tr F1':>8}  {'Te F1':>8}  {'Tr Acc':>10}  {'Te Acc':>10}  {'Tr Loss':>8}  {'Te Loss':>8}")
print("-" * 76)

for chunk_num in range(1, N_CHUNKS + 1):
    start = (chunk_num - 1) * chunk_size
    end   = start + chunk_size
    batch = idx[start:end]

    model.partial_fit(X_train_sc[batch], y_train[batch], classes=CLASSES)

    if chunk_num % 10 == 0:
        tr_pred = model.predict(X_train_sc[:end])   # only seen data so far
        te_pred = model.predict(X_test_sc)
        tr_prob = model.predict_proba(X_train_sc[:end])
        te_prob = model.predict_proba(X_test_sc)

        tr_f1_  = f1_score(y_train[:end], tr_pred)
        te_f1_  = f1_score(y_test,        te_pred)
        tr_ba   = accuracy_score(y_train[:end], tr_pred)
        te_ba   = accuracy_score(y_test,        te_pred)
        tr_loss = log_loss(y_train[:end], tr_prob)
        te_loss = log_loss(y_test,        te_prob)
        tr_rec  = recall_score(y_train[:end], tr_pred)
        te_rec  = recall_score(y_test,        te_pred)

        train_f1.append(tr_f1_);      test_f1.append(te_f1_)
        train_bacc.append(tr_ba);     test_bacc.append(te_ba)
        train_losses.append(tr_loss); test_losses.append(te_loss)
        train_recall.append(tr_rec);  test_recall.append(te_rec)
        log_chunks.append(chunk_num)

        print(f"{chunk_num:>6}  {tr_f1_:>8.4f}  {te_f1_:>8.4f}  {tr_ba:>10.4f}  {te_ba:>10.4f}  {tr_loss:>8.4f}  {te_loss:>8.4f}")


y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\n-- Final Metrics --------------------------------")
print(f"Test Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 (diabetic)     : {f1_score(y_test, y_pred):.4f}")
print(f"Test Recall (diabetic) : {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC                : {roc_auc_score(y_test, y_prob):.4f}")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Naive Bayes - Diabetes Prediction', fontsize=13, fontweight='bold')

x_ticks = log_chunks

# Score
axes[0,0].plot(log_chunks, train_f1, marker='o', color='steelblue', linewidth=2, markersize=4, label='Train F1')
axes[0,0].plot(log_chunks, test_f1,  marker='s', color='tomato',    linewidth=2, markersize=4, label='Test F1')
axes[0,0].set_xlabel('Chunks seen')
axes[0,0].set_ylabel('F1 Score')
axes[0,0].set_title('F1 Score - Diabetic Class')
axes[0,0].set_xticks(x_ticks)
axes[0,0].legend()
axes[0,0].grid(True, linestyle='--', alpha=0.5)

# Accuracy
axes[0,1].plot(log_chunks, train_bacc, marker='o', color='steelblue', linewidth=2, markersize=4, label='Train Accuracy')
axes[0,1].plot(log_chunks, test_bacc,  marker='s', color='tomato',    linewidth=2, markersize=4, label='Test Accuracy')
axes[0,1].set_xlabel('Chunks seen')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].set_title('Accuracy over Chunks')
axes[0,1].set_xticks(x_ticks)
axes[0,1].legend()
axes[0,1].grid(True, linestyle='--', alpha=0.5)

# Log Loss
axes[1,0].plot(log_chunks, train_losses, marker='o', color='steelblue', linewidth=2, markersize=4, label='Train Loss')
axes[1,0].plot(log_chunks, test_losses,  marker='s', color='tomato',    linewidth=2, markersize=4, label='Test Loss')
axes[1,0].set_xlabel('Chunks seen')
axes[1,0].set_ylabel('Log Loss')
axes[1,0].set_title('Log Loss over Chunks')
axes[1,0].set_xticks(x_ticks)
axes[1,0].legend()
axes[1,0].grid(True, linestyle='--', alpha=0.5)

# Recall
axes[1,1].plot(log_chunks, train_recall, marker='o', color='steelblue', linewidth=2, markersize=4, label='Train Recall')
axes[1,1].plot(log_chunks, test_recall,  marker='s', color='tomato',    linewidth=2, markersize=4, label='Test Recall')
axes[1,1].set_xlabel('Chunks seen')
axes[1,1].set_ylabel('Recall')
axes[1,1].set_title('Recall - Diabetic Class')
axes[1,1].set_xticks(x_ticks)
axes[1,1].legend()
axes[1,1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('naive_bayes_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved -> naive_bayes_results.png")
