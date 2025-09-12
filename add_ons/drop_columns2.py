def drop_columns(data, cols_to_drop, dict_name="main"):
    """
    Remove specified columns from a DataFrame or a dict of DataFrames.
    Returns (df_out, bad_indices) or (df_out, bad_indices, dicts).
    """
    if isinstance(data, tuple):
        df, dicts = data

        if dict_name == "main":
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        else:
            if dict_name in dicts:
                dicts[dict_name] = dicts[dict_name].drop(
                    columns=[c for c in cols_to_drop if c in dicts[dict_name].columns]
                )
            else:
                raise KeyError(f"Dict '{dict_name}' not found in pipeline extras")

        return df, [], dicts  # ✅ always 3-tuple

    else:  # plain DataFrame
        df_out = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
        return df_out, []  # ✅ always 2-tuple
