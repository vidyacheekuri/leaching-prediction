#!/usr/bin/env python3
"""
Simple test script to verify data processing fix.
Tests the column detection without requiring full ML pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, os.path.dirname(__file__))

# Import directly to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("data_processing", "src/data_processing.py")
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)
DataProcessor = data_processing.DataProcessor

print("=" * 60)
print("Testing Data Processing Fix")
print("=" * 60)
print()

excel_file = 'data/LXS-Monolithe-21.xlsx'

if not os.path.exists(excel_file):
    print(f"❌ Error: {excel_file} not found!")
    sys.exit(1)

try:
    processor = DataProcessor(excel_file)
    
    print("📊 Processing data with fixed column detection...")
    print()
    
    # Process data
    df = processor.load_and_consolidate_data()
    
    print()
    print("✅ Data processing complete!")
    print(f"📈 Total records: {len(df)}")
    print()
    
    # Verify materials
    print("=" * 60)
    print("VERIFICATION - Checking Material Data")
    print("=" * 60)
    print()
    
    # Check Aluminum (should be correct)
    print("1. Aluminum (Al) - Reference material:")
    al = df[df['Material'] == 'Al'].head(3)
    if len(al) > 0:
        print(al[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].to_string())
        al_ph_range = (al['pH'].min(), al['pH'].max())
        if 10 <= al_ph_range[0] <= 14:
            print("   ✅ pH values look correct (10-14 range)")
        else:
            print(f"   ⚠️  pH range {al_ph_range} might be wrong")
    print()
    
    # Check Barium (should now be fixed)
    print("2. Barium (Ba) - Should be FIXED now:")
    ba = df[df['Material'] == 'Ba'].head(5)
    if len(ba) > 0:
        print(ba[['Material', 'pH', 'Time_days', 'Cumulative_Release_mg_m2']].to_string())
        ba_ph_range = (ba['pH'].min(), ba['pH'].max())
        ba_time_range = (ba['Time_days'].min(), ba['Time_days'].max())
        
        if 10 <= ba_ph_range[0] <= 14:
            print(f"   ✅ pH values look correct: {ba_ph_range[0]:.2f} to {ba_ph_range[1]:.2f}")
        else:
            print(f"   ❌ pH values still wrong: {ba_ph_range[0]:.2f} to {ba_ph_range[1]:.2f}")
            print("      (Should be ~11-12, not 1-9)")
        
        if 0.01 <= ba_time_range[0] <= 1.0:
            print(f"   ✅ Time values look correct: {ba_time_range[0]:.2f} to {ba_time_range[1]:.2f}")
        else:
            print(f"   ❌ Time values still wrong: {ba_time_range[0]:.2f} to {ba_time_range[1]:.2f}")
            print("      (Should be 0.08-64, not 10-12)")
    else:
        print("   ⚠️  No Barium records found")
    print()
    
    # Check other affected materials
    print("3. Other materials (As, Cr, Mo) - Should be fixed:")
    for material in ['As', 'Cr', 'Mo']:
        mat_data = df[df['Material'] == material]
        if len(mat_data) > 0:
            ph_range = (mat_data['pH'].min(), mat_data['pH'].max())
            time_range = (mat_data['Time_days'].min(), mat_data['Time_days'].max())
            print(f"   {material}: pH {ph_range[0]:.2f}-{ph_range[1]:.2f}, Time {time_range[0]:.2f}-{time_range[1]:.2f}")
            if 10 <= ph_range[0] <= 14:
                print(f"      ✅ {material} pH looks correct")
            else:
                print(f"      ⚠️  {material} pH might still be wrong")
    print()
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Materials: {df['Material'].nunique()}")
    print(f"pH range (all): {df['pH'].min():.2f} to {df['pH'].max():.2f}")
    print(f"Time range (all): {df['Time_days'].min():.2f} to {df['Time_days'].max():.2f}")
    print()
    
    # Check for suspicious values
    suspicious_ph = df[df['pH'] < 3]  # pH < 3 is very unusual for cement
    if len(suspicious_ph) > 0:
        print(f"⚠️  WARNING: {len(suspicious_ph)} records have pH < 3 (unusual for cement)")
        print(f"   Materials affected: {suspicious_ph['Material'].unique().tolist()}")
    else:
        print("✅ No records with suspiciously low pH values")
    print()
    
    # Save to file
    output_file = 'data/processed/consolidated_leaching_data_FIXED.csv'
    df.to_csv(output_file, index=False)
    print(f"💾 Saved reprocessed data to: {output_file}")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. ✅ Review the output above")
    print("2. ⏳ Compare a few records with Excel file manually")
    print("3. ⏳ If data looks correct, replace old CSV:")
    print("   mv data/processed/consolidated_leaching_data_FIXED.csv \\")
    print("      data/processed/consolidated_leaching_data_FINAL.csv")
    print("4. ⏳ Retrain model: python main.py")
    print("5. ⏳ Deploy updated model")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

