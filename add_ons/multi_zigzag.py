
def add_zigzag(df, window_size=3, dev_threshold=1, max_pivots=10, prefix=None):
    """
    Append zigzag features (swing highs/lows) to dataframe.
    """
    zz = ZigZag(window_size=window_size, dev_threshold=dev_threshold, max_pivots=max_pivots)

    df = df.copy()
    features_list = []

    for idx, row in df.iterrows():
        zz.update(idx, row["close"], high=row["high"], low=row["low"])
        feats = zz.get_features(current_price=row["close"], current_index=idx)

        # prefix feature names if provided
        if prefix:
            feats = {f"{prefix}_{k}": v for k, v in feats.items()}

        features_list.append(feats)

    features_df = pd.DataFrame(features_list, index=df.index)
    return pd.concat([df, features_df], axis=1)
