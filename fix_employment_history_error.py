"""
Fix for KeyError: 'Employment History (Years)' in neurodiversity_analysis.ipynb

This script demonstrates how to fix the error by:
1. Filtering out non-existent columns from the feature list
2. OR ensuring all columns exist in your dataframe
"""

# Run this in your notebook to fix the issue
# Option 1: Remove the problematic column from your feature list
all_features = [feature for feature in all_features if feature != 'Employment History (Years)']
print(f"Updated features list (removed 'Employment History (Years)'): {len(all_features)} features")

# Option 2: Modify the classify_condition function to handle missing columns
def classify_condition(condition_column, feature_list, use_pca=True):
    # Your existing code...
    
    # Create copy of df with only the features that exist in the dataframe
    # This ensures no KeyError for missing columns
    existing_features = [feat for feat in feature_list if feat in df.columns]
    X = df[existing_features]
    
    # Continue with your existing code...
    # For numeric_cols, also filter out missing columns:
    numeric_cols = [col for col in numeric_cols if col in X.columns]
    
    # Now fill NAs safely
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # Continue with the rest of your function... 