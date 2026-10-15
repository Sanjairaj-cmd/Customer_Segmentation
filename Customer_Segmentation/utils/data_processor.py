import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_or_path):
    """
    Load CSV dataset from file path or uploaded Streamlit file buffer.
    Returns (DataFrame, error_message).
    """
    try:
        if isinstance(file_or_path, str):
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
            
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        if df.empty:
            return None, "The uploaded CSV file is empty."
            
        return df, None
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}"

def auto_detect_columns(df):
    """
    Intelligently suggest default columns for Income, Spending Score, Age, Gender, and ID.
    Returns a dictionary of suggested column names.
    """
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    
    mapping = {
        'id': None,
        'gender': None,
        'age': None,
        'income': None,
        'spending': None,
        'numeric_cols': []
    }
    
    # Identify all numeric columns
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    mapping['numeric_cols'] = numeric_cols
    
    # Regex / keyword matching
    for orig, low in zip(cols, cols_lower):
        if 'id' in low and mapping['id'] is None:
            mapping['id'] = orig
        elif 'gender' in low or 'sex' in low:
            mapping['gender'] = orig
        elif 'age' in low:
            mapping['age'] = orig
        elif ('income' in low or 'salary' in low or 'earning' in low or 'revenue' in low) and mapping['income'] is None:
            mapping['income'] = orig
        elif ('spending' in low or 'score' in low or 'expenditure' in low or 'purchase' in low) and mapping['spending'] is None:
            mapping['spending'] = orig

    # Fallbacks if keyword matching didn't catch specific numeric columns
    if not mapping['income'] and len(numeric_cols) > 0:
        mapping['income'] = numeric_cols[0]
    if not mapping['spending'] and len(numeric_cols) > 1:
        mapping['spending'] = numeric_cols[1]
    elif not mapping['spending'] and len(numeric_cols) > 0:
        mapping['spending'] = numeric_cols[0]
        
    return mapping

def clean_data(df, selected_features, handle_missing="impute"):
    """
    Clean dataset: missing value imputation/removal and type safety.
    Returns (cleaned_df, report_dict).
    """
    cleaned_df = df.copy()
    report = {
        'initial_rows': len(df),
        'missing_values_imputed': 0,
        'rows_dropped': 0,
        'duplicates_removed': 0
    }
    
    # Check for duplicates
    initial_len = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    report['duplicates_removed'] = initial_len - len(cleaned_df)
    
    # Check missing values in selected features
    missing_counts = cleaned_df[selected_features].isnull().sum().sum()
    
    if missing_counts > 0:
        if handle_missing == "drop":
            cleaned_df = cleaned_df.dropna(subset=selected_features)
            report['rows_dropped'] = initial_len - len(cleaned_df)
        else:
            # Impute numerical features with median
            for col in selected_features:
                if np.issubdtype(cleaned_df[col].dtype, np.number):
                    median_val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)
            report['missing_values_imputed'] = missing_counts
            
    report['final_rows'] = len(cleaned_df)
    return cleaned_df, report

def scale_features(df, features, method="StandardScaler"):
    """
    Scale numeric features using StandardScaler or MinMaxScaler.
    Returns (scaled_array, scaler_instance).
    """
    feature_data = df[features].values
    
    if method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    scaled_array = scaler.fit_transform(feature_data)
    return scaled_array, scaler
