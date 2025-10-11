from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def universal_scaler(df_data, df_labels, feature_columns=None, label_columns=None):
    """
    Universal Scaler Afterburner.
    
    Scales the given feature columns and label columns together by using a combined scaler.
    It fits a scaler using all values from both feature and label columns together (flattened), 
    then transforms them separately.
    
    Args:
    - df_data (pd.DataFrame): DataFrame of features.
    - df_labels (pd.DataFrame): DataFrame of labels.
    - feature_columns (list): List of feature column names to scale (e.g., ['feature1', 'feature2']).
    - label_columns (list): List of label column names to scale (e.g., ['label1', 'label2']).
    
    Returns:
    - pd.DataFrame: The transformed DataFrame with scaled features and labels.
    """
    
    # Ensure that the necessary columns are present
    if feature_columns is None:
        raise ValueError("Feature columns must be specified for scaling.")
    if label_columns is None:
        label_columns = [col for col in df_labels.columns if col.startswith("linePrice")]
    
    # Ensure the required feature columns are in the df_data DataFrame
    for feature in feature_columns:
        if feature not in df_data:
            raise ValueError(f"Feature column '{feature}' not found in the DataFrame.")
    
    # Ensure the required label columns are in the df_labels DataFrame
    for label in label_columns:
        if label not in df_labels:
            raise ValueError(f"Label column '{label}' not found in the DataFrame.")
    
    # Step 1: Extract values from df_data and df_labels and flatten them
    all_feature_values = df_data[feature_columns].values.flatten()  # Flatten feature columns into a 1D array
    all_label_values = df_labels[label_columns].values.flatten()    # Flatten label columns into a 1D array

    # Step 2: Combine feature and label values into a single 1D array for scaling
    combined_values = np.concatenate([all_feature_values, all_label_values])

    # Step 3: Fit the scaler on the combined values
    scaler = StandardScaler()
    scaler.fit(combined_values.reshape(-1, 1))  # Reshape to 2D for fitting

    # Step 4: Store the scaler in the info dict
    info = {"universal_scaler": scaler}

    # Step 5: Apply the same scaler to df_data and df_labels independently
    # Reshape back and transform feature columns
    # CORRECTED CODE
    # Reshape, transform, and then reshape back for features
    feature_values = df_data[feature_columns].values
    df_data[feature_columns] = scaler.transform(feature_values.reshape(-1, 1)).reshape(feature_values.shape)

    # Do the same for labels
    label_values = df_labels[label_columns].values
    df_labels[label_columns] = scaler.transform(label_values.reshape(-1, 1)).reshape(label_values.shape)
        
    # Step 6: Return the scaled dataframes
    return df_data, df_labels, info
