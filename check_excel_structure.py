#!/usr/bin/env python3
"""
Script to check Excel file structure and compare with processed CSV.
This helps identify if pH and Time columns are swapped.
"""

import pandas as pd
import sys
import os

excel_file = 'data/LXS-Monolithe-21.xlsx'

if not os.path.exists(excel_file):
    print(f"Error: {excel_file} not found")
    sys.exit(1)

try:
    xl_file = pd.ExcelFile(excel_file)
    material_sheets = [s for s in xl_file.sheet_names if 'Monolithic' in s]
    
    print(f"Found {len(material_sheets)} material sheets\n")
    
    # Check a few different materials
    materials_to_check = ['Al', 'Ba', 'As']  # Check Aluminum, Barium, Arsenic
    
    for material_name in materials_to_check:
        # Find sheet for this material
        sheet_name = None
        for s in material_sheets:
            if material_name in s or material_name.lower() in s.lower():
                sheet_name = s
                break
        
        if not sheet_name:
            print(f"⚠️  Sheet not found for {material_name}, skipping...")
            continue
        
        print(f"{'='*60}")
        print(f"Material: {material_name}")
        print(f"Sheet: {sheet_name}")
        print(f"{'='*60}\n")
        
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        
        # Find header rows
        header_rows = []
        for idx, row in df.iterrows():
            row_values = [str(cell) if pd.notna(cell) else '' for cell in row]
            row_text = ' '.join(row_values)
            
            if 'pH' in row_text and 'Cumulative release' in row_text:
                header_rows.append((idx, row_values))
        
        if not header_rows:
            print(f"⚠️  No header row found for {material_name}")
            continue
        
        # Check each header row
        for header_idx, header_values in header_rows[:2]:  # Check first 2 headers
            print(f"Header row at index {header_idx}:")
            print(f"  Column indices and values:")
            for col_idx, val in enumerate(header_values[:10]):
                if val.strip():
                    print(f"    Column {col_idx}: '{val}'")
            print()
            
            # Check if Time column exists
            has_time = 'Time' in ' '.join(header_values)
            has_fraction = 'Fraction' in ' '.join(header_values)
            
            print(f"  Has 'Time' column: {has_time}")
            print(f"  Has 'Fraction' column: {has_fraction}")
            print()
            
            # Find column indices
            ph_col = None
            time_col = None
            fraction_col = None
            release_col = None
            
            for col_idx, val in enumerate(header_values):
                val_lower = str(val).lower()
                if 'ph' in val_lower and ph_col is None:
                    ph_col = col_idx
                if 'time' in val_lower and time_col is None:
                    time_col = col_idx
                if 'fraction' in val_lower and fraction_col is None:
                    fraction_col = col_idx
                if 'cumulative' in val_lower and 'release' in val_lower and release_col is None:
                    release_col = col_idx
            
            print(f"  Column positions:")
            print(f"    pH column: {ph_col}")
            print(f"    Time column: {time_col}")
            print(f"    Fraction column: {fraction_col}")
            print(f"    Cumulative Release column: {release_col}")
            print()
            
            # Show first few data rows
            print(f"  First 3 data rows:")
            for i in range(1, 4):
                if header_idx + i < len(df):
                    data_row = df.iloc[header_idx + i]
                    row_data = {}
                    if ph_col is not None:
                        row_data['pH'] = data_row.iloc[ph_col] if ph_col < len(data_row) else 'N/A'
                    if time_col is not None:
                        row_data['Time'] = data_row.iloc[time_col] if time_col < len(data_row) else 'N/A'
                    if fraction_col is not None:
                        row_data['Fraction'] = data_row.iloc[fraction_col] if fraction_col < len(data_row) else 'N/A'
                    if release_col is not None:
                        row_data['Release'] = data_row.iloc[release_col] if release_col < len(data_row) else 'N/A'
                    
                    print(f"    Row {header_idx + i}: {row_data}")
            print()
            
            # Compare with processed CSV
            print(f"  Comparison with processed CSV:")
            try:
                csv_df = pd.read_csv('data/processed/consolidated_leaching_data_FINAL.csv')
                csv_material = csv_df[csv_df['Material'] == material_name].head(3)
                if len(csv_material) > 0:
                    print(f"    CSV data (first 3 rows):")
                    for _, row in csv_material.iterrows():
                        print(f"      pH={row['pH']}, Time={row['Time_days']}, Release={row['Cumulative_Release_mg_m2']}")
                else:
                    print(f"    ⚠️  No CSV data found for {material_name}")
            except Exception as e:
                print(f"    ⚠️  Error reading CSV: {e}")
            print()
            print()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

