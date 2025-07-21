#!/usr/bin/env python3
# Credit Card Fraud Detection - SMOTE Implementation
# This script applies SMOTE to the training data only and saves it to a separate CSV

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

print("="*80)
print("CREDIT CARD FRAUD DETECTION - SMOTE IMPLEMENTATION")
print("="*80)

# Check if the cleaned dataset exists
if not os.path.exists('cleaned_credit_card_data.csv'):
    print("Error: cleaned_credit_card_data.csv file not found!")
    print("Please run the data cleaning script first.")
    exit(1)

# Load the cleaned dataset
print("Loading cleaned dataset...")
cleaned_df = pd.read_csv('cleaned_credit_card_data.csv')

# Split the data into features and target
print("\nSplitting data into features and target...")
X = cleaned_df.drop(['Class'], axis=1)
y = cleaned_df['Class']

# Perform train-test-validation split (same as in the cleaning script)
print("Performing stratified train-test-validation split...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Display class distribution before SMOTE
print("\nClass distribution before SMOTE:")
print(f"- Training set: {X_train.shape[0]} samples")
print(f"  - Legitimate (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.4f}%)")
print(f"  - Fraudulent (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.4f}%)")
print(f"- Validation set: {X_val.shape[0]} samples, fraud rate: {y_val.mean():.6f}")
print(f"- Test set: {X_test.shape[0]} samples, fraud rate: {y_test.mean():.6f}")

# Apply SMOTE to the training data only
print("\nApplying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Display class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(f"- Training set (with SMOTE): {X_train_smote.shape[0]} samples")
print(f"  - Legitimate (0): {(y_train_smote == 0).sum()} ({(y_train_smote == 0).sum() / len(y_train_smote) * 100:.4f}%)")
print(f"  - Fraudulent (1): {(y_train_smote == 1).sum()} ({(y_train_smote == 1).sum() / len(y_train_smote) * 100:.4f}%)")

# Create a DataFrame with the SMOTE-augmented training data
print("\nCreating DataFrame with SMOTE-augmented training data...")
train_smote_df = X_train_smote.copy()
train_smote_df['Class'] = y_train_smote

# Save the SMOTE-augmented training data to a separate CSV
print("Saving SMOTE-augmented training data to smote_train.csv...")
train_smote_df.to_csv('smote_train.csv', index=False)

# Also save the original train, validation, and test sets for reference
print("Saving original train, validation, and test sets...")
train_df = X_train.copy()
train_df['Class'] = y_train
train_df.to_csv('original_train.csv', index=False)

val_df = X_val.copy()
val_df['Class'] = y_val
val_df.to_csv('original_val.csv', index=False)

test_df = X_test.copy()
test_df['Class'] = y_test
test_df.to_csv('original_test.csv', index=False)

print("\nSMOTE implementation completed successfully!")
print("Files created:")
print("- smote_train.csv: Training data with SMOTE applied")
print("- original_train.csv: Original training data (without SMOTE)")
print("- original_val.csv: Original validation data")
print("- original_test.csv: Original test data")
print("\nIMPORTANT: Use smote_train.csv for training models with balanced classes")
print("          Use original_val.csv and original_test.csv for validation and testing")
