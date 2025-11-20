#!/usr/bin/env python3
"""
Script to reprocess data with the fixed data processing code.
This will regenerate the consolidated CSV with correct column mapping.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from src.data_processing import DataProcessor

print("=" * 60)
print("Reprocessing Data with Fixed Column Detection")
print("=" * 60)
print()

# Initialize processor
excel_file = 'data/LXS-Monolithe-21.xlsx'
if not os.path.exists(excel_file):
    print(f"❌ Error: {excel_file} not found!")
    sys.exit(1)

processor = DataProcessor(excel_file)

print("📊 Loading and processing data...")
print()

try:
    # Process data
    df = processor.load_and_consolidate_data()
    
    print()
    print("✅ Data processing complete!")
    print(f"📈 Total records: {len(df)}")
    print()
    
    # Verify some materials
    print("Verification - Sample records:")
    print()
    
    # Check Aluminum (should be correct)
    print("Aluminum (Al):")
    al = df[df['Material'] == 'Al'].head(3)
    print(al[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].to_string())
    print()
    
    # Check Barium (should now be fixed)
    print("Barium (Ba) - Should now be fixed:")
    ba = df[df['Material'] == 'Ba'].head(3)
    if len(ba) > 0:
        print(ba[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].to_string())
        # Verify pH values are reasonable (should be ~11-12, not 1-9)
        if ba['pH'].min() > 10:
            print("✅ Barium pH values look correct (pH > 10)")
        else:
            print("⚠️  Barium pH values still look wrong (pH < 10)")
    else:
        print("⚠️  No Barium records found")
    print()
    
    # Check a few more materials
    for material in ['As', 'Cr']:
        mat_data = df[df['Material'] == material].head(1)
        if len(mat_data) > 0:
            print(f"{material}:")
            print(f"  pH: {mat_data.iloc[0]['pH']:.2f}, Time: {mat_data.iloc[0]['Time_days']:.2f}")
            if mat_data.iloc[0]['pH'] > 10:
                print(f"  ✅ pH looks reasonable")
            else:
                print(f"  ⚠️  pH looks wrong (might still be swapped)")
    print()
    
    # Save to new file for comparison
    output_file = 'data/processed/consolidated_leaching_data_FIXED.csv'
    df.to_csv(output_file, index=False)
    print(f"💾 Saved reprocessed data to: {output_file}")
    print()
    print("⚠️  IMPORTANT: Compare this with the original Excel file to verify correctness!")
    print("   Then replace the old CSV and retrain the model.")
    
except Exception as e:
    print(f"❌ Error during processing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

