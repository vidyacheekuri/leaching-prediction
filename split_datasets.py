#!/usr/bin/env python3
"""
Split the processed data into training and testing datasets.

This script creates the same train/test split that was used during model training
and saves them as separate CSV files for analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DataProcessor
from src.utils import load_config


def main():
    """Split data into training and testing sets and save them."""
    print("📊 Creating Train/Test Dataset Split")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Load processed data
    processed_data_path = 'data/processed/consolidated_leaching_data_FINAL.csv'
    if os.path.exists(processed_data_path):
        print("📂 Loading processed data...")
        df = pd.read_csv(processed_data_path)
    else:
        print("📂 Processing raw data...")
        excel_file = config.get('paths', {}).get('data_file', 'LXS-Monolithe-21.xlsx')
        processor = DataProcessor(excel_file)
        df = processor.load_and_consolidate_data()
        df, feature_columns, label_encoders = processor.create_features(df)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv(processed_data_path, index=False)
    
    print(f"📈 Total dataset: {len(df)} samples")
    print(f"📈 Features: {len([col for col in df.columns if col not in ['Material', 'Cement_Type', 'Form_Type', 'Stat_Measure', 'Cumulative_Release_mg_m2']])}")
    
    # Get feature columns (exclude target and categorical columns that are encoded)
    feature_columns = [col for col in df.columns if col not in [
        'Material', 'Cement_Type', 'Form_Type', 'Stat_Measure', 
        'Cumulative_Release_mg_m2', 'Material_Condition', 'log_Release'
    ]]
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Cumulative_Release_mg_m2']
    
    # Create the same train/test split as used in training (80/20 split, random_state=42)
    test_size = config.get('data', {}).get('test_size', 0.2)
    random_state = config.get('data', {}).get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=df['Material']
    )
    
    # Get the corresponding indices for the full dataframe
    train_indices = X_train.index
    test_indices = X_test.index
    
    # Create train and test dataframes
    df_train = df.loc[train_indices].copy()
    df_test = df.loc[test_indices].copy()
    
    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Save datasets
    train_file = 'data/processed/train_dataset.csv'
    test_file = 'data/processed/test_dataset.csv'
    
    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
    
    print(f"✅ Training set saved: {train_file}")
    print(f"   Samples: {len(df_train)}")
    print(f"   Materials: {df_train['Material'].nunique()}")
    print(f"   pH range: {df_train['pH'].min():.2f} - {df_train['pH'].max():.2f}")
    print(f"   Time range: {df_train['Time_days'].min():.2f} - {df_train['Time_days'].max():.2f} days")
    print(f"   Target range: {df_train['Cumulative_Release_mg_m2'].min():.2f} - {df_train['Cumulative_Release_mg_m2'].max():.2f} mg/m²")
    
    print(f"\n✅ Test set saved: {test_file}")
    print(f"   Samples: {len(df_test)}")
    print(f"   Materials: {df_test['Material'].nunique()}")
    print(f"   pH range: {df_test['pH'].min():.2f} - {df_test['pH'].max():.2f}")
    print(f"   Time range: {df_test['Time_days'].min():.2f} - {df_test['Time_days'].max():.2f} days")
    print(f"   Target range: {df_test['Cumulative_Release_mg_m2'].min():.2f} - {df_test['Cumulative_Release_mg_m2'].max():.2f} mg/m²")
    
    # Show material distribution
    print(f"\n📊 Material Distribution:")
    print("Training set:")
    train_materials = df_train['Material'].value_counts().sort_index()
    for material, count in train_materials.items():
        print(f"   {material}: {count}")
    
    print("\nTest set:")
    test_materials = df_test['Material'].value_counts().sort_index()
    for material, count in test_materials.items():
        print(f"   {material}: {count}")
    
    # Calculate split statistics
    train_ratio = len(df_train) / len(df)
    test_ratio = len(df_test) / len(df)
    
    print(f"\n📈 Split Statistics:")
    print(f"   Training: {len(df_train)} samples ({train_ratio:.1%})")
    print(f"   Testing: {len(df_test)} samples ({test_ratio:.1%})")
    print(f"   Total: {len(df)} samples")
    
    print(f"\n🎉 Dataset split completed successfully!")
    print(f"📁 Files created:")
    print(f"   - {train_file}")
    print(f"   - {test_file}")


if __name__ == "__main__":
    main()
