import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/diabetes_prediction_dataset.csv')

os.makedirs('figures', exist_ok=True)

def save_plot(filename, plot_fn):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_fn(ax)
    plt.tight_layout()
    fig.savefig(f'figures/{filename}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature vs Diabetes Overview', fontsize=16, fontweight='bold')

box_palette = ['steelblue', 'tomato']

sns.boxplot(x='diabetes', y='bmi',               data=df, palette=box_palette, ax=axes[0, 0])
axes[0, 0].set_title('BMI vs Diabetes')

sns.boxplot(x='diabetes', y='age',               data=df, palette=box_palette, ax=axes[0, 1])
axes[0, 1].set_title('Age vs Diabetes')

sns.countplot(x='gender', hue='diabetes',        data=df, palette=box_palette, ax=axes[0, 2])
axes[0, 2].set_title('Gender vs Diabetes')

sns.boxplot(x='diabetes', y='HbA1c_level',       data=df, palette=box_palette, ax=axes[1, 0])
axes[1, 0].set_title('HbA1c Level vs Diabetes')

sns.boxplot(x='diabetes', y='blood_glucose_level', data=df, palette=box_palette, ax=axes[1, 1])
axes[1, 1].set_title('Blood Glucose Level vs Diabetes')

axes[1, 2].set_visible(False)   

plt.tight_layout(pad=3.0, h_pad=4.0)
plt.show()

# Interaction plot: HbA1c vs Blood Glucose by Diabetes 
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x='HbA1c_level', y='blood_glucose_level', hue='diabetes',
                palette=box_palette, data=df, alpha=0.5, ax=ax)
ax.set_title('HbA1c vs Blood Glucose Level by Diabetes')
plt.tight_layout()
plt.show()

save_plot('bmi_vs_diabetes', lambda ax: (
    sns.boxplot(x='diabetes', y='bmi', data=df, palette=box_palette, ax=ax),
    ax.set_title('BMI vs Diabetes')
))

save_plot('age_vs_diabetes', lambda ax: (
    sns.boxplot(x='diabetes', y='age', data=df, palette=box_palette, ax=ax),
    ax.set_title('Age vs Diabetes')
))

save_plot('gender_vs_diabetes', lambda ax: (
    sns.countplot(x='gender', hue='diabetes', data=df, palette=box_palette, ax=ax),
    ax.set_title('Gender vs Diabetes')
))

save_plot('hba1c_vs_diabetes', lambda ax: (
    sns.boxplot(x='diabetes', y='HbA1c_level', data=df, palette=box_palette, ax=ax),
    ax.set_title('HbA1c Level vs Diabetes')
))

save_plot('blood_glucose_vs_diabetes', lambda ax: (
    sns.boxplot(x='diabetes', y='blood_glucose_level', data=df, palette=box_palette, ax=ax),
    ax.set_title('Blood Glucose Level vs Diabetes')
))

fig.savefig('figures/hba1c_glucose_interaction.png', dpi=150, bbox_inches='tight')
plt.close()
