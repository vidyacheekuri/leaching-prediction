#!/usr/bin/env python3
"""
Investigate the statistical anomaly where mean/median values are lower than CL-.

This script analyzes the data to understand why statistical measures (mean/median)
might be lower than CL- (confidence limit minus), which is statistically surprising.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def investigate_anomaly():
    """Investigate the mean/median < CL- statistical anomaly."""
    print("=" * 70)
    print("🔍 INVESTIGATING STATISTICAL ANOMALY")
    print("Mean/Median < CL- Analysis")
    print("=" * 70)
    print()
    
    # Load data
    df = pd.read_csv('data/processed/consolidated_leaching_data_FINAL.csv')
    print(f"📊 Loaded {len(df)} records")
    print()
    
    # Group by Material and Stat_Measure
    print("=" * 70)
    print("📈 Analysis by Material and Statistical Measure")
    print("=" * 70)
    print()
    
    results = []
    
    for material in df['Material'].unique():
        material_df = df[df['Material'] == material]
        
        # Get different statistical measures
        stat_measures = material_df['Stat_Measure'].unique()
        
        if 'CL_Minus' not in stat_measures:
            continue
        
        # Get CL- values
        cl_minus_df = material_df[material_df['Stat_Measure'] == 'CL_Minus']
        
        # Check other statistical measures
        for stat in stat_measures:
            if stat == 'CL_Minus':
                continue
            
            stat_df = material_df[material_df['Stat_Measure'] == stat]
            
            if len(stat_df) == 0 or len(cl_minus_df) == 0:
                continue
            
            # Compare at same time points (64 days for standard comparison)
            cl_minus_64 = cl_minus_df[cl_minus_df['Time_days'] == 64]['Cumulative_Release_mg_m2']
            stat_64 = stat_df[stat_df['Time_days'] == 64]['Cumulative_Release_mg_m2']
            
            if len(cl_minus_64) > 0 and len(stat_64) > 0:
                cl_minus_val = cl_minus_64.values[0]
                stat_mean = stat_64.mean()
                stat_median = stat_64.median()
                
                anomaly = {
                    'Material': material,
                    'Stat_Measure': stat,
                    'CL_Minus_Value': cl_minus_val,
                    'Stat_Mean': stat_mean,
                    'Stat_Median': stat_median,
                    'Mean_Lower_Than_CL': stat_mean < cl_minus_val,
                    'Median_Lower_Than_CL': stat_median < cl_minus_val,
                    'Mean_Diff_%': ((stat_mean - cl_minus_val) / cl_minus_val * 100) if cl_minus_val > 0 else np.nan,
                    'Median_Diff_%': ((stat_median - cl_minus_val) / cl_minus_val * 100) if cl_minus_val > 0 else np.nan
                }
                
                results.append(anomaly)
                
                if stat_mean < cl_minus_val or stat_median < cl_minus_val:
                    print(f"⚠️  {material:3s} | {stat:10s} | CL-={cl_minus_val:10.2f} | "
                          f"Mean={stat_mean:10.2f} | Median={stat_median:10.2f}")
    
    print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        results_df.to_csv(results_dir / 'statistical_anomaly_analysis.csv', index=False)
        
        # Summary
        print("=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Total comparisons: {len(results_df)}")
        print(f"Cases where Mean < CL-: {results_df['Mean_Lower_Than_CL'].sum()}")
        print(f"Cases where Median < CL-: {results_df['Median_Lower_Than_CL'].sum()}")
        print()
        
        if results_df['Mean_Lower_Than_CL'].sum() > 0 or results_df['Median_Lower_Than_CL'].sum() > 0:
            print("⚠️  ANOMALIES FOUND:")
            print("-" * 70)
            anomalies = results_df[
                (results_df['Mean_Lower_Than_CL']) | 
                (results_df['Median_Lower_Than_CL'])
            ]
            print(anomalies[['Material', 'Stat_Measure', 'CL_Minus_Value', 
                            'Stat_Mean', 'Stat_Median', 'Mean_Diff_%', 'Median_Diff_%']].to_string(index=False))
            print()
            print("💡 Possible explanations:")
            print("   1. CL- might be calculated differently than mean/median")
            print("   2. Different sample sizes for different statistical measures")
            print("   3. CL- might include uncertainty bounds")
            print("   4. Data processing or extraction differences")
        
        print()
        print("💾 Results saved to: results/statistical_anomaly_analysis.csv")
    else:
        print("✅ No anomalies found - all means/medians are >= CL-")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    investigate_anomaly()

