import pandas as pd
import numpy as np
import joblib
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

train_df = pd.read_csv('data/diabetes_prediction_dataset.csv')
aug = pd.read_csv('data/synthetic_diabetes_data_positive.csv')

train_df = pd.concat([train_df, aug], ignore_index=True)

TRAIN_GLUCOSE_SORTED = np.sort(train_df['blood_glucose_level'].values)

def quantile_map_glucose(pima_glucose_values):
    """Map PIMA OGTT glucose to training glucose distribution via quantile matching."""
    pima_arr = np.asarray(pima_glucose_values)
    ranks = pd.Series(pima_arr).rank(pct=True).values
    return np.quantile(TRAIN_GLUCOSE_SORTED, ranks)

def map_pima_to_training_schema(pima_df):
    df = pima_df.copy()
    for col in ['Glucose', 'BloodPressure', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
    df = df.dropna(subset=['Glucose', 'BloodPressure', 'BMI'])

    # Quantile-map glucose to training scale, to correct the theoretical formula with distribution of the dataset
    mapped_glucose = quantile_map_glucose(df['Glucose'].values)

    out = pd.DataFrame(index=df.index)
    out['gender']              = 'Female' #  this dataset only contains females
    out['age']                 = df['Age'].astype(float)
    out['hypertension']        = (df['BloodPressure'] >= 90).astype(int)  # tunable
    out['heart_disease']       = 0
    out['smoking_history']     = 'No Info'
    out['bmi']                 = df['BMI'].astype(float)
    out['blood_glucose_level'] = mapped_glucose
    out['HbA1c_level']         = (mapped_glucose + 46.7) / 28.7 # using some theoretical formula
    out['diabetes']            = df['Outcome'].astype(int)

    out = out.dropna()
    return out

try: # only run if using PIMA dataset to validate
    pima = pd.read_csv('data/diabetes_pima_dataset.csv')
except:
    print("No PIMA dataset!")
else:
    pima_mapped = map_pima_to_training_schema(pima).reset_index(drop=True)
    print(f"Rows after mapping: {len(pima_mapped)} / {len(pima)}")
    print(f"Class balance:\n{pima_mapped['diabetes'].value_counts(normalize=True)}\n")

    y_true = pima_mapped['diabetes'].values

    P_TRAIN = train_df['diabetes'].mean()   
    P_PIMA  = pima_mapped['diabetes'].mean()  

def prior_shift_correct(probs, p_train=P_TRAIN, p_test=P_PIMA):
    """Adjust probabilities for known prevalence shift between train and test."""
    r_pos = p_test / p_train
    r_neg = (1 - p_test) / (1 - p_train)
    return (probs * r_pos) / (probs * r_pos + (1 - probs) * r_neg)

# refit encoders on original training data
gender_le  = LabelEncoder().fit(train_df['gender'])
smoking_le = LabelEncoder().fit(train_df['smoking_history'])

def build_logreg_features(df):
    """Matches FEATURES list in the logreg training script."""
    f = pd.DataFrame()
    f['gender_enc']          = gender_le.transform(df['gender'])
    f['age']                 = df['age']
    f['hypertension']        = df['hypertension']
    f['heart_disease']       = df['heart_disease']
    f['smoking_enc']         = smoking_le.transform(df['smoking_history'])
    f['bmi']                 = df['bmi']
    f['HbA1c_level']         = df['HbA1c_level']
    f['blood_glucose_level'] = df['blood_glucose_level']
    f['hba1c_prediab']       = ((df['HbA1c_level'] >= 5.7) & (df['HbA1c_level'] < 6.5)).astype(int)
    f['both_markers_high']   = ((df['HbA1c_level'] >= 5.7) & (df['blood_glucose_level'] >= 126)).astype(int)
    return f.values

def build_svm_features(df):
    """Matches the SVM preprocessing function (gender_binary, smoking_history_catogorical)."""
    f = pd.DataFrame()
    f['age']                          = df['age']
    f['hypertension']                 = df['hypertension']
    f['heart_disease']                = df['heart_disease']
    f['bmi']                          = df['bmi']
    f['HbA1c_level']                  = df['HbA1c_level']
    f['blood_glucose_level']          = df['blood_glucose_level']
    f['gender_binary']                = df['gender'].map({'Male': 1, 'Female': 0})
    smoking_map = {a: i for i, a in enumerate(train_df['smoking_history'].unique())}
    f['smoking_history_catogorical']  = df['smoking_history'].map(smoking_map)
    return f

def build_base_features(df):
    """Generic 8-feature layout — assumed for RF and XGB. Adjust if needed."""
    f = pd.DataFrame()
    f['gender_enc']          = gender_le.transform(df['gender'])
    f['age']                 = df['age']
    f['hypertension']        = df['hypertension']
    f['heart_disease']       = df['heart_disease']
    f['smoking_enc']         = smoking_le.transform(df['smoking_history'])
    f['bmi']                 = df['bmi']
    f['HbA1c_level']         = df['HbA1c_level']
    f['blood_glucose_level'] = df['blood_glucose_level']
    return f

def predict_logreg(path, df):
    bundle = joblib.load(path)
    X = build_logreg_features(df)
    X = bundle['scaler'].transform(X)
    X = bundle['poly'].transform(X)
    return bundle['model'].predict_proba(X)[:, 1]

_svm_train_X = build_svm_features(train_df)
_SVM_INPUT_SCALER = StandardScaler().fit(_svm_train_X)

def predict_svm(path, df):
    try:
        obj = joblib.load(path)
    except Exception:
        with open(path, 'rb') as fh:
            obj = pickle.load(fh)
    pipeline = obj['pipeline'] if isinstance(obj, dict) and 'pipeline' in obj else obj

    X = build_svm_features(df)
    X = _SVM_INPUT_SCALER.transform(X)   # match training preprocessing

    if hasattr(pipeline, 'predict_proba'):
        return pipeline.predict_proba(X)[:, 1]
    scores = pipeline.decision_function(X)
    return 1 / (1 + np.exp(-scores))

def recategorize_smoking(s):
    if s in ['never', 'No Info']:
        return 'non-smoker'
    elif s == 'current':
        return 'current'
    elif s in ['ever', 'former', 'not current']:
        return 'past_smoker'

def process_for_rf(df):
    X = df.copy()
    X['msi']        = X['HbA1c_level'] * X['blood_glucose_level']
    X['age_bmi']    = X['age'] * X['bmi']
    X['HbA1c_cat']  = pd.cut(X['HbA1c_level'],
                             bins=[0, 5.7, 6.0, 6.2, 6.5, 100],
                             labels=[0, 1, 2, 3, 4])
    X['bmi_cat']    = pd.cut(X['bmi'], bins=[0, 23, 100], labels=[0, 1])
    X['smoking_history'] = X['smoking_history'].apply(recategorize_smoking)
    X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True)
    return X

# Build canonical column list from training data — used to align PIMA features
_train_features = train_df.drop_duplicates().drop(columns=['diabetes'])
RF_COLUMNS = process_for_rf(_train_features).columns.tolist()

def predict_rf(path, df):
    with open(path, 'rb') as fh:
        obj = pickle.load(fh)
    model = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj

    X = process_for_rf(df.drop(columns=['diabetes']))
    # PIMA lacks gender_Male/gender_Other and some smoking dummies — fill with 0
    X = X.reindex(columns=RF_COLUMNS, fill_value=0)
    return model.predict_proba(X)[:, 1]

def predict_xgb(path, df):
    obj = joblib.load(path)
    # Handle either a bare model or a bundle dict
    model = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    X = build_base_features(df).rename(columns={
        'gender_enc':  'gender',
        'smoking_enc': 'smoking_history',
    })

    # Sklearn API (XGBClassifier) vs native API (Booster) have different interfaces
    if hasattr(model, 'predict_proba'):
        # XGBClassifier — uses feature_names_in_ if available
        if hasattr(model, 'feature_names_in_'):
            X = X[list(model.feature_names_in_)]
        return model.predict_proba(X)[:, 1]
    else:
        # Native Booster
        X = X[model.feature_names]
        dmat = xgb.DMatrix(X.values, feature_names=model.feature_names)
        return model.predict(dmat)

def _print_progress(label, done, total, width=30):
    pct = done / total
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    print(f'\r  {label:<22} [{bar}] {pct*100:5.1f}% ({done}/{total})',
          end='', flush=True)

def evaluate(name, df, correct_prior=False):
    df = df[df['gender'].isin(['Male', 'Female'])].reset_index(drop=True)
    y = df['diabetes'].values

    print(f"\n=== {name} (n={len(df)}) ===")

    steps = [
        ('logreg', lambda: predict_logreg('models/logreg_model.joblib', df)),
        ('svm',    lambda: predict_svm   ('models/svm_model.joblib',       df)),
        ('rf',     lambda: predict_rf    ('models/rf_model.pkl',        df)),
        ('xgb',    lambda: predict_xgb   ('models/xgb_model.joblib',    df)),
    ]
    total = len(steps)
    probs = {}
    for i, (model_name, fn) in enumerate(steps):
        _print_progress(f'predicting {model_name}', i, total)
        probs[model_name] = fn()
    _print_progress('done', total, total)
    print()  

    if correct_prior:
        probs = {k: prior_shift_correct(v) for k, v in probs.items()}

    weights = {
        'xgb':    0.50,
        'rf':     0.10,
        'logreg': 0.20,
        'svm':    0.20,
    }
    probs['ensemble'] = np.average(
        [probs['logreg'], probs['svm'], probs['rf'], probs['xgb']],
        axis=0,
        weights=[weights['logreg'], weights['svm'], weights['rf'], weights['xgb']]
    )

    print(f"{'Model':<10} {'Accuracy':>10} {'F1':>10} {'Recall':>10}")
    print("-" * 44)
    for model_name, p in probs.items():
        pred = (p >= 0.5).astype(int)
        print(f"{model_name:<10} "
              f"{accuracy_score(y, pred):>10.4f} "
              f"{f1_score(y, pred, average='macro'):>10.4f} "
              f"{recall_score(y, pred, average='macro'):>10.4f}")

evaluate("Training set", train_df)

try: 
    pima = pd.read_csv('data/diabetes_pima_dataset.csv')
except:
    print("No PIMA dataset!")
else:
    evaluate("PIMA (raw)", pima_mapped)
    evaluate("PIMA (prior-corrected)", pima_mapped, correct_prior=True)
