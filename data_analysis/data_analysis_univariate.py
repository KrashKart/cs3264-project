import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import os

df = pd.read_csv('data/diabetes_prediction_dataset.csv')

def save_plot(filename, plot_fn):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_fn(ax)
    plt.tight_layout()
    fig.savefig(f'figures/{filename}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')

diab_palette = ['steelblue', 'tomato']


# Histogram for age
axes[0, 0].hist(df['age'], bins=30, edgecolor='black', color='steelblue')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')

# Bar plot for gender
sns.countplot(x='gender', hue='gender', data=df, palette=['hotpink', 'deepskyblue', 'mediumpurple'], ax=axes[0, 1])
axes[0, 1].set_title('Gender Distribution')

# Distribution plot for BMI
sns.histplot(df['bmi'], bins=30, kde=True, color='mediumpurple', ax=axes[0, 2])
axes[0, 2].set_title('BMI Distribution')

# Count plots for binary variables
for ax, col in zip(axes.flat[3:6], ['hypertension', 'heart_disease', 'diabetes']):
    sns.countplot(x=col, data=df, palette=diab_palette, ax=ax)
    ax.set_title(f'{col.replace("_", " ").title()} Distribution')

# Count plot for smoking history
sns.countplot(x='smoking_history', data=df, palette='tab10', ax=axes[2, 0])
axes[2, 0].set_title('Smoking History Distribution')
axes[2, 0].tick_params(axis='x', rotation=30)

# Hide unused subplots
for ax in axes.flat[7:]:
    ax.set_visible(False)

plt.tight_layout(pad=3.0, h_pad=4.0)
plt.show()

save_plot('age_distribution', lambda ax: (
    ax.hist(df['age'], bins=30, edgecolor='black', color='steelblue'),
    ax.set(title='Age Distribution', xlabel='Age', ylabel='Count')
))

save_plot('gender_distribution', lambda ax: (
    sns.countplot(x='gender', hue='gender', data=df, palette=['hotpink', 'deepskyblue', 'mediumpurple'], ax=ax),
    ax.set_title('Gender Distribution')
))

save_plot('bmi_distribution', lambda ax: (
    sns.histplot(df['bmi'], bins=30, kde=True, color='mediumpurple', ax=ax),
    ax.set_title('BMI Distribution')
))

for col in ['hypertension', 'heart_disease', 'diabetes']:
    save_plot(col, lambda ax, c=col: (
        sns.countplot(x=c, data=df, palette=diab_palette, ax=ax),
        ax.set_title(f'{c.replace("_", " ").title()} Distribution')
    ))

save_plot('smoking_history', lambda ax: (
    sns.countplot(x='smoking_history', data=df, palette='tab10', ax=ax),
    ax.set_title('Smoking History Distribution'),
    ax.tick_params(axis='x', rotation=30)
))
