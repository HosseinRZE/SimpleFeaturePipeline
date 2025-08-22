import pandas as pd
from utils.zigzag_bandf import ZigZag

def add_zigzag(
    df,
    window_size=3,
    dev_threshold=1,
    max_pivots=10,
    stationary=False,
    include_last_candle_as_pivot=True,
    include_distances=True,
    shadow_mode=False,
    separatable="no"  # "no", "complete", "both"
):
    """
    Append zigzag features (pivots + distances) to dataframe.
    
    Parameters:
        separatable : str
            - "no"       : default, merge features into main df only
            - "complete" : features are not merged, returned in dict only
            - "both"     : features are merged into df and also returned in dict
    """
    if separatable not in ["no", "complete", "both"]:
        raise ValueError("separatable must be 'no', 'complete', or 'both'")

    zz = ZigZag(
        window_size=window_size,
        dev_threshold=dev_threshold,
        max_pivots=max_pivots,
        stationary=stationary,
        include_last_candle_as_pivot=include_last_candle_as_pivot,
        include_distances=include_distances,
        shadow_mode=shadow_mode,
    )

    df = df.copy()
    feature_dicts = []

    for idx, row in df.iterrows():
        zz.update(idx, row["high"], row["low"], row["close"])
        features = zz.get_features(current_price=row["close"], current_index=idx)
        feature_dicts.append(features)

    features_df = pd.DataFrame(feature_dicts, index=df.index)

    if separatable == "complete":
        return df, {"zigzag": features_df}
    elif separatable == "both":
        df = pd.concat([df, features_df], axis=1)
        return df, {"zigzag": features_df}
    else:  # "no"
        df = pd.concat([df, features_df], axis=1)
        return df