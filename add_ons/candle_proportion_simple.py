import numpy as np
import pandas as pd

def add_candle_shape_features(df, **kwargs):
    """
    Computes candlestick shape features with optional multi-dict support.
    
    Arguments:
        df: pd.DataFrame with OHLC columns.
        kwargs:
            ohlc_cols: tuple of (open, high, low, close) column names.
            seperatable: "no" (default) -> add to main only
                         "complete" -> add to new dict only
                         "both" -> add to both main and new dict
            dict_name: optional, name for new dict if seperatable != "no"
    
    Returns:
        - If seperatable="no": (df_out, invalid_indices)
        - If seperatable!="no": (df_out, invalid_indices, dict_out)
    """
    df_out = df.copy()
    invalid_indices = set()
    dict_name = kwargs.get("dict_name", "add_candle_shape_features")
    seperatable = kwargs.get("seperatable", "no")
    
    open_col, high_col, low_col, close_col = kwargs.get(
        "ohlc_cols", ("open", "high", "low", "close")
    )
    
    max_oc = df_out[[open_col, close_col]].max(axis=1)
    min_oc = df_out[[open_col, close_col]].min(axis=1)
    
    max_oc_safe = max_oc.replace(0, np.nan)
    min_oc_safe = min_oc.replace(0, np.nan)
    
    # Compute features
    upper_shadow = (df_out[high_col] - max_oc) / max_oc_safe
    invalid_indices.update(upper_shadow[upper_shadow.isna()].index)
    
    lower_shadow = (min_oc - df_out[low_col]) / min_oc_safe
    invalid_indices.update(lower_shadow[lower_shadow.isna()].index)
    
    body = (max_oc - min_oc) / max_oc_safe
    invalid_indices.update(body[body.isna()].index)
    
    color = np.where(
        df_out[close_col] > df_out[open_col], 0.7,
        np.where(df_out[close_col] < df_out[open_col], 0.3, 0.0)
    )
    
    # Prepare feature DataFrame
    features_df = pd.DataFrame({
        "upper_shadow": upper_shadow.fillna(0.0).astype(np.float32),
        "lower_shadow": lower_shadow.fillna(0.0).astype(np.float32),
        "body": body.fillna(0.0).astype(np.float32),
        "color": color.astype(np.float32)
    }, index=df_out.index)
    
    if seperatable == "no":
        # Just merge into main
        df_out = pd.concat([df_out, features_df], axis=1)
        return df_out, list(invalid_indices)
    
    elif seperatable == "complete":
        # Keep main untouched, return new dict only
        dict_out = {dict_name: features_df}
        return df_out, list(invalid_indices), dict_out
    
    elif seperatable == "both":
        # Merge into main AND return new dict
        df_out = pd.concat([df_out, features_df], axis=1)
        dict_out = {dict_name: features_df.copy()}
        return df_out, list(invalid_indices), dict_out
    
    else:
        raise ValueError(f"Invalid seperatable value: {seperatable}")
