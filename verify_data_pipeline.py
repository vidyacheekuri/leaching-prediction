"""
Comprehensive data verification script.
Verifies original Excel data matches processed CSV, checks data quality, and identifies issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from src.data_processing import DataProcessor

def verify_excel_processing():
    """Verify that Excel data is correctly processed."""
    print("=" * 80)
    print("DATA VERIFICATION PIPELINE")
    print("=" * 80)
    
    excel_file = "LXS-Monolithe-21.xlsx"
    processed_file = "data/processed/consolidated_leaching_data_FINAL.csv"
    
    # Step 1: Load original Excel and reprocess
    print("\n📊 Step 1: Loading and reprocessing Excel file...")
    processor = DataProcessor(excel_file)
    df_new = processor.load_and_consolidate_data()
    df_new, feature_columns, label_encoders = processor.create_features(df_new)
    
    print(f"   ✅ Reprocessed {len(df_new)} records")
    print(f"   Materials: {df_new['Material'].nunique()}")
    
    # Step 2: Load existing processed CSV
    print("\n📊 Step 2: Loading existing processed CSV...")
    if Path(processed_file).exists():
        df_existing = pd.read_csv(processed_file)
        print(f"   ✅ Loaded {len(df_existing)} records from CSV")
        print(f"   Materials: {df_existing['Material'].nunique()}")
    else:
        print(f"   ⚠️  CSV file not found: {processed_file}")
        df_existing = None
    
    # Step 3: Compare records
    if df_existing is not None:
        print("\n🔍 Step 3: Comparing records...")
        
        # Check for specific test cases
        test_cases = [
            {'Material': 'Cd', 'pH': 11.88, 'Time_days': 9.0, 'Stat_Measure': 'LPx'},
            {'Material': 'Ba', 'pH': 12.08, 'Time_days': 0.08, 'Stat_Measure': 'CL_Minus'},
            {'Material': 'Al', 'pH': 12.08, 'Time_days': 0.08, 'Stat_Measure': 'CL_Minus'},
        ]
        
        print("\n   Test Case Comparisons:")
        print("   " + "-" * 76)
        for test in test_cases:
            print(f"\n   Material: {test['Material']}, pH: {test['pH']}, Time: {test['Time_days']}, Stat: {test['Stat_Measure']}")
            
            # Find in new data
            new_match = df_new[
                (df_new['Material'] == test['Material']) &
                (np.isclose(df_new['pH'], test['pH'], rtol=1e-3)) &
                (np.isclose(df_new['Time_days'], test['Time_days'], rtol=1e-3)) &
                (df_new['Stat_Measure'] == test['Stat_Measure'])
            ]
            
            # Find in existing data
            existing_match = df_existing[
                (df_existing['Material'] == test['Material']) &
                (np.isclose(df_existing['pH'], test['pH'], rtol=1e-3)) &
                (np.isclose(df_existing['Time_days'], test['Time_days'], rtol=1e-3)) &
                (df_existing['Stat_Measure'] == test['Stat_Measure'])
            ]
            
            if len(new_match) > 0:
                new_val = new_match.iloc[0]['Cumulative_Release_mg_m2']
                print(f"      New processing: {new_val:.6f} mg/m²")
            else:
                print(f"      New processing: ❌ NOT FOUND")
            
            if len(existing_match) > 0:
                existing_val = existing_match.iloc[0]['Cumulative_Release_mg_m2']
                print(f"      Existing CSV:   {existing_val:.6f} mg/m²")
            else:
                print(f"      Existing CSV:   ❌ NOT FOUND")
            
            if len(new_match) > 0 and len(existing_match) > 0:
                diff = abs(new_val - existing_val)
                if diff < 1e-6:
                    print(f"      ✅ Match!")
                else:
                    print(f"      ⚠️  Mismatch! Difference: {diff:.6f}")
    
    # Step 4: Data quality checks
    print("\n🔍 Step 4: Data Quality Checks...")
    
    # Check for negative values
    negative = df_new[df_new['Cumulative_Release_mg_m2'] < 0]
    if len(negative) > 0:
        print(f"   ⚠️  Found {len(negative)} negative release values")
    else:
        print(f"   ✅ No negative release values")
    
    # Check pH range
    ph_min, ph_max = df_new['pH'].min(), df_new['pH'].max()
    print(f"   pH range: {ph_min:.2f} - {ph_max:.2f}")
    if ph_min < 0 or ph_max > 14:
        print(f"   ⚠️  pH values outside expected range [0, 14]")
    else:
        print(f"   ✅ pH values in valid range")
    
    # Check time range
    time_min, time_max = df_new['Time_days'].min(), df_new['Time_days'].max()
    print(f"   Time range: {time_min:.2f} - {time_max:.2f} days")
    if time_max > 64:
        print(f"   ⚠️  Time values exceed 64 days (max: {time_max:.2f})")
    else:
        print(f"   ✅ Time values within 64 days")
    
    # Check for missing values
    missing = df_new[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].isnull().sum()
    if missing.sum() > 0:
        print(f"   ⚠️  Missing values found:")
        print(missing[missing > 0])
    else:
        print(f"   ✅ No missing values in critical columns")
    
    # Step 5: Material-specific checks
    print("\n🔍 Step 5: Material-Specific Analysis...")
    material_stats = df_new.groupby('Material').agg({
        'Cumulative_Release_mg_m2': ['count', 'min', 'max', 'mean', 'median'],
        'pH': ['min', 'max'],
        'Time_days': ['min', 'max']
    }).round(6)
    
    print("\n   Material Statistics (first 10):")
    print(material_stats.head(10))
    
    # Step 6: Check for pH/Time swap issues
    print("\n🔍 Step 6: Checking for pH/Time Swap Issues...")
    suspicious = df_new[
        (df_new['pH'] > 64) |  # pH shouldn't be > 14
        (df_new['Time_days'] > 64) |  # Time shouldn't exceed 64
        (df_new['pH'] < 0) |  # pH shouldn't be negative
        (df_new['Time_days'] < 0)  # Time shouldn't be negative
    ]
    
    if len(suspicious) > 0:
        print(f"   ⚠️  Found {len(suspicious)} suspicious records:")
        print(suspicious[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].head(10))
    else:
        print(f"   ✅ No obvious pH/Time swap issues")
    
    # Step 7: Save verification report
    print("\n💾 Step 7: Saving verification report...")
    output_file = "data/processed/consolidated_leaching_data_VERIFIED.csv"
    df_new.to_csv(output_file, index=False)
    print(f"   ✅ Saved verified data to: {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total records: {len(df_new)}")
    print(f"Materials: {df_new['Material'].nunique()}")
    print(f"pH range: {ph_min:.2f} - {ph_max:.2f}")
    print(f"Time range: {time_min:.2f} - {time_max:.2f} days")
    print(f"Release range: {df_new['Cumulative_Release_mg_m2'].min():.6f} - {df_new['Cumulative_Release_mg_m2'].max():.6f} mg/m²")
    
    return df_new, feature_columns, label_encoders

if __name__ == "__main__":
    df_verified, features, encoders = verify_excel_processing()

