import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
    
def process(df):
    df = df.drop_duplicates()
    X = df.iloc[:, :-1]
    X["msi"] = X["HbA1c_level"] * X["blood_glucose_level"]
    X["age_bmi"] = X["age"] * X["bmi"]
    X["HbA1c_cat"] = pd.cut(X['HbA1c_level'], bins=[0, 5.7, 6.0, 6.2, 6.5, 100], labels=[0, 1, 2, 3, 4])
    X["bmi_cat"] = pd.cut(X['bmi'], bins=[0, 23, 100], labels=[0, 1])
    X['smoking_history'] = X['smoking_history'].apply(recategorize_smoking)
    X = pd.get_dummies(X, columns=["gender", "smoking_history"], drop_first=True)
    y = df["diabetes"]
    return X, y

def main():
    # read file
    df = pd.read_csv('diabetes_prediction_dataset.csv')

    X, y = process(df)
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level','hypertension', 'heart_disease', 'msi', 'age_bmi'])],
                                     remainder="passthrough")
    oversampler = SMOTETomek(sampling_strategy=0.2, n_jobs=-1, random_state=42)
    undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=1, min_samples_split=5, n_jobs=-1)

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('over', oversampler),
                            ('under', undersampler),
                            ('classifier', rf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5-fold startified cv
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(clf, X_train, y_train, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'f1_macro'])
    for k, v in cv_results.items():
        print(f"{k}: {v} (Avg {sum(v)/len(v)})")

    # test set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    with open('rf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
if __name__ == "__main__":
    main()
