import pandas as pd
import numpy as np
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from scipy.spatial.distance import jensenshannon

LOAD = True
print("Loading Data")
real_data = pd.read_csv('diabetes_prediction_dataset.csv')

if not LOAD:
    metadata = Metadata.detect_from_dataframe(data=real_data)
    # print(metadata)

    print("Training")
    synthesizer = TVAESynthesizer(metadata, batch_size=1000, epochs=300, verbose=1)
    synthesizer.fit(real_data)

    print("Generating")
    diabetes_only_condition = Condition(num_rows=20000, column_values={'diabetes': 1})

    # synthetic_data = synthesizer.sample(num_rows=30_000)
    synthetic_data = synthesizer.sample_from_conditions(conditions=[diabetes_only_condition])

    print("Saving")
    synthetic_data.to_csv('synthetic_diabetes_data_positive.csv', index=False)
else:
    synthetic_data = pd.read_csv('synthetic_diabetes_data_positive.csv')
    positive_data = real_data[real_data['diabetes'] == 1]

print("Evaluating")
def jensen_shannon_distance(p, q, bins=20):
    """Calculates JS Distance (0 to 1). Lower is better."""
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    # Add small epsilon to avoid division by zero
    return jensenshannon(p_hist + 1e-10, q_hist + 1e-10)

# --- Metric 1: Mean and Standard Deviation ---
stats_comparison = pd.DataFrame({
    'Real Mean': positive_data.mean(numeric_only=True),
    'Syn Mean': synthetic_data.mean(numeric_only=True),
    'Real Std': positive_data.std(numeric_only=True),
    'Syn Std': synthetic_data.std(numeric_only=True)
})

# --- Metric 2: Jensen-Shannon Distance ---
# Measures similarity of distributions for each column
js_results = {col: jensen_shannon_distance(positive_data[col], synthetic_data[col]) 
              for col in real_data.select_dtypes(include=[np.number]).columns}

# --- Metric 3: Correlation Fidelity ---
# Calculates how much the correlation structure has drifted
real_corr = positive_data.corr(numeric_only=True)
syn_corr = synthetic_data.corr(numeric_only=True)
corr_fidelity_mae = np.abs(real_corr - syn_corr).mean().mean()

# --- Output Results ---
print("--- Mean & Std Comparison ---")
print(stats_comparison)

print("\n--- Jensen-Shannon Distances (Close to 0 is identical) ---")
for col, dist in js_results.items():
    print(f"{col}: {dist:.4f}")

cat_cols = real_data.select_dtypes(include=['object', 'category']).columns

print("\n--- Categorical Columns Analysis ---")
for col in cat_cols:
    # Get counts and percentages for Real data
    real_counts = positive_data[col].value_counts()
    real_pct = positive_data[col].value_counts(normalize=True) * 100
    
    # Get counts and percentages for Synthetic data
    syn_counts = synthetic_data[col].value_counts()
    syn_pct = synthetic_data[col].value_counts(normalize=True) * 100
    
    # Combine into a single comparison DataFrame
    comparison_df = pd.DataFrame({
        'Real Count': real_counts,
        'Real %': real_pct,
        'Syn Count': syn_counts,
        'Syn %': syn_pct
    }).fillna(0) # Fill NaN with 0 if a category exists in one but not the other
    
    # Calculate the difference in percentage points
    comparison_df['Diff %'] = comparison_df['Real %'] - comparison_df['Syn %']
    
    print(f"\nDistribution for: {col}")
    print(comparison_df.round(2))
