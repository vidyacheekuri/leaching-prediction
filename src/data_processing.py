"""
Data processing module for cement leaching prediction.

This module handles Excel file loading, data extraction, feature engineering,
and data preprocessing for the cement leaching prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional


class DataProcessor:
    """Handles data loading, processing, and feature engineering."""
    
    def __init__(self, excel_file_path: str):
        """
        Initialize the data processor.
        
        Args:
            excel_file_path: Path to the Excel file containing leaching data
        """
        self.excel_file_path = excel_file_path
        self.label_encoders = {}
        
    def extract_leaching_data_from_sheet(self, sheet_df: pd.DataFrame, material_name: str) -> List[Dict]:
        """
        Extract leaching data from a single material sheet.
        
        Handles both standard format and special Bromide format correctly.
        
        Args:
            sheet_df: DataFrame containing the sheet data
            material_name: Name of the material being processed
            
        Returns:
            List of dictionaries containing extracted data points
        """
        consolidated_data = []
        current_material_condition = None

        for idx, row in sheet_df.iterrows():
            row_values = [str(cell) if pd.notna(cell) else '' for cell in row]
            row_text = ' '.join(row_values)

            # Check if this is a material condition header
            if any(keyword in row_text for keyword in ['Cement concrete', 'Cement mortar', 'Mortar', 'Sewage sludge']):
                for cell in row_values:
                    if len(cell) > 20 and any(keyword in cell for keyword in ['Cement', 'Mortar', 'Sewage']):
                        current_material_condition = cell
                        break

            # Check if this is a data header row (contains pH and Cumulative release)
            if 'pH' in row_text and 'Cumulative release' in row_text:
                # Determine column structure based on whether "Time" is present
                has_time_column = 'Time' in row_text

                # Look for data in subsequent rows
                data_idx = idx + 1

                while data_idx < len(sheet_df):
                    data_row = sheet_df.iloc[data_idx]

                    # Stop if we hit another header or empty section
                    if len(data_row) < 5 or (pd.isna(data_row.iloc[2]) and pd.isna(data_row.iloc[3])):
                        break

                    try:
                        if has_time_column:
                            # Standard format: pH, Time, Cumulative Release in columns 2, 3, 4
                            ph_val = pd.to_numeric(data_row.iloc[2], errors='coerce')
                            time_val = pd.to_numeric(data_row.iloc[3], errors='coerce')
                            cumulative_release = pd.to_numeric(data_row.iloc[4], errors='coerce')
                        else:
                            # Special Bromide format: Fraction, pH, Cumulative Release in columns 2, 3, 4
                            fraction_val = pd.to_numeric(data_row.iloc[2], errors='coerce')
                            ph_val = pd.to_numeric(data_row.iloc[3], errors='coerce')
                            cumulative_release = pd.to_numeric(data_row.iloc[4], errors='coerce')

                            # Map fraction to time values (based on standard leaching test time points)
                            time_mapping = {1: 0.08, 2: 1, 3: 2.25, 4: 4, 5: 9, 6: 16, 7: 28, 8: 36, 9: 64}
                            time_val = time_mapping.get(fraction_val, np.nan)

                        if pd.notna(ph_val) and pd.notna(cumulative_release) and pd.notna(time_val):
                            consolidated_data.append({
                                'Material': material_name,
                                'Material_Condition': current_material_condition,
                                'pH': ph_val,
                                'Time_days': time_val,
                                'Cumulative_Release_mg_m2': cumulative_release
                            })
                    except:
                        pass

                    data_idx += 1

        return consolidated_data

    def clean_material_condition(self, condition_str: str) -> Tuple[str, str, str]:
        """
        Clean and extract features from material condition strings.
        
        Args:
            condition_str: Raw material condition string
            
        Returns:
            Tuple of (cement_type, form_type, stat_measure)
        """
        if pd.isna(condition_str):
            return 'Unknown', 'Unknown', 'Unknown'

        condition_str = str(condition_str).strip()

        # Extract cement type
        cement_type = 'Unknown'
        if 'CEM I' in condition_str and 'CEM II' not in condition_str:
            cement_type = 'CEM_I'
        elif 'CEM II' in condition_str and 'CEM III' not in condition_str:
            cement_type = 'CEM_II'
        elif 'CEM III' in condition_str:
            cement_type = 'CEM_III'
        elif 'CEM V' in condition_str:
            cement_type = 'CEM_V'

        # Extract form type
        form_type = 'Unknown'
        if 'Cement concrete' in condition_str:
            form_type = 'Concrete'
        elif 'Cement mortar' in condition_str or 'Mortar' in condition_str:
            form_type = 'Mortar'
        elif 'Sewage sludge' in condition_str:
            form_type = 'Sewage_Sludge'

        # Extract statistical measure
        stat_measure = 'Unknown'
        if 'Monolith-[CL-]' in condition_str:
            stat_measure = 'CL_Minus'
        elif 'Monolith-[CL+]' in condition_str or 'Monolith-CL' in condition_str:
            stat_measure = 'CL_Plus'
        elif 'Monolith-[LPx]' in condition_str:
            stat_measure = 'LPx'
        elif 'Monolith-[Med]' in condition_str:
            stat_measure = 'Med'
        elif 'Monolith-[UPx]' in condition_str:
            stat_measure = 'UPx'
        elif 'Monolith-[X]' in condition_str:
            stat_measure = 'X'

        return cement_type, form_type, stat_measure

    def load_and_consolidate_data(self) -> pd.DataFrame:
        """
        Load and consolidate data from Excel file.
        
        Returns:
            Consolidated DataFrame with all leaching data
        """
        print(f"📊 Loading data from {self.excel_file_path}...")
        
        # Load Excel file
        df_sheets = pd.read_excel(self.excel_file_path, sheet_name=None)

        # Get material sheets
        material_sheets = [name for name in df_sheets.keys() if 'Monolithic-' in name]
        all_data = []

        for sheet_name in material_sheets:
            material_name = sheet_name.replace('Monolithic-', '').strip()
            sheet_data = self.extract_leaching_data_from_sheet(df_sheets[sheet_name], material_name)
            all_data.extend(sheet_data)
            print(f"  {material_name}: {len(sheet_data)} data points")

        df = pd.DataFrame(all_data)
        print(f"\n✅ Consolidated {len(df)} total data points from {df['Material'].nunique()} materials")

        return df

    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Create comprehensive feature set for machine learning.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            Tuple of (processed_df, feature_columns, label_encoders)
        """
        print("🔧 Creating advanced features...")

        # Extract features from material conditions
        material_features = df['Material_Condition'].apply(self.clean_material_condition)
        df['Cement_Type'] = [x[0] for x in material_features]
        df['Form_Type'] = [x[1] for x in material_features]
        df['Stat_Measure'] = [x[2] for x in material_features]

        # Encode categorical variables
        categorical_columns = ['Material', 'Cement_Type', 'Form_Type', 'Stat_Measure']

        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            print(f"  {col}: {df[col].nunique()} categories")

        # Create additional engineered features
        print("  Creating mathematical transformations...")
        df['log_Time'] = np.log1p(df['Time_days'])
        df['log_pH'] = np.log(df['pH'])
        df['sqrt_Time'] = np.sqrt(df['Time_days'])
        df['sqrt_pH'] = np.sqrt(df['pH'])
        df['pH_squared'] = df['pH'] ** 2
        df['pH_cubed'] = df['pH'] ** 3
        df['Time_squared'] = df['Time_days'] ** 2

        # Interaction features
        print("  Creating interaction features...")
        df['Time_pH_interaction'] = df['Time_days'] * df['pH']
        df['log_Time_pH'] = df['log_Time'] * df['pH']
        df['Material_pH_interaction'] = df['Material_encoded'] * df['pH']
        df['Material_Time_interaction'] = df['Material_encoded'] * df['Time_days']

        # Normalized features
        df['pH_normalized'] = (df['pH'] - df['pH'].mean()) / df['pH'].std()
        df['Time_normalized'] = (df['Time_days'] - df['Time_days'].mean()) / df['Time_days'].std()

        # Domain knowledge features
        print("  Creating domain-specific features...")
        df['Alkalinity_index'] = df['pH'] - 7  # Distance from neutral
        df['Reactivity_score'] = df['pH'] * np.log1p(df['Time_days'])
        df['Leaching_potential'] = (df['pH'] ** 2) * np.sqrt(df['Time_days'])

        # Material grouping by chemical similarity
        material_groups = {
            'Al': 0, 'Fe': 0, 'Si': 0,  # Light metals/metalloids
            'As': 1, 'Cr': 1, 'Mo': 1,  # Toxic heavy elements
            'Ba': 2, 'P': 2,             # Alkaline earth/phosphorus
            'Br': 3, 'F': 3, 'Cl': 3,   # Halogens
            'Ca': 4, 'K': 4, 'Mg': 4, 'Na': 4,  # Alkali/alkaline earth
            'Cd': 5, 'Cu': 5, 'Pb': 5, 'Zn': 5,  # Heavy metals
            'SO4': 6                     # Sulfates
        }
        df['Material_group'] = df['Material'].map(material_groups)

        # Target transformation
        df['log_Release'] = np.log1p(df['Cumulative_Release_mg_m2'])

        # Define comprehensive feature set
        feature_columns = [
            'Material_encoded', 'Cement_Type_encoded', 'Form_Type_encoded', 'Stat_Measure_encoded',
            'pH', 'Time_days', 'Cement_Content', 'Additives_Count',
            'log_Time', 'log_pH', 'sqrt_Time', 'sqrt_pH',
            'pH_squared', 'pH_cubed', 'Time_squared',
            'Time_pH_interaction', 'log_Time_pH', 'Material_pH_interaction', 'Material_Time_interaction',
            'pH_normalized', 'Time_normalized',
            'Alkalinity_index', 'Reactivity_score', 'Leaching_potential', 'Material_group'
        ]

        # Add default values for missing columns
        if 'Cement_Content' not in df.columns:
            df['Cement_Content'] = 80  # Default value
        if 'Additives_Count' not in df.columns:
            df['Additives_Count'] = 1  # Default value

        print(f"✅ Created {len(feature_columns)} features")

        return df, feature_columns, self.label_encoders

    def save_processed_data(self, df: pd.DataFrame, output_path: str = 'data/processed/consolidated_leaching_data_FINAL.csv'):
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path
        """
        df.to_csv(output_path, index=False)
        print(f"✅ Data saved to '{output_path}'")

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_samples': len(df),
            'materials': sorted(df['Material'].unique()),
            'num_materials': df['Material'].nunique(),
            'ph_range': (df['pH'].min(), df['pH'].max()),
            'time_range': (df['Time_days'].min(), df['Time_days'].max()),
            'leaching_range': (df['Cumulative_Release_mg_m2'].min(), df['Cumulative_Release_mg_m2'].max()),
            'material_distribution': df['Material'].value_counts().sort_index().to_dict()
        }
        return summary
