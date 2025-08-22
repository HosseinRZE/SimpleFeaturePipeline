def drop_columns(data, cols_to_drop, dict_name="main"):
    """
    Remove specified columns from a DataFrame or a dict of DataFrames.

    Args:
        data (pd.DataFrame or (pd.DataFrame, dict)): Input DataFrame or (df, dict_of_dfs).
        cols_to_drop (list[str]): Columns to remove.
        dict_name (str): Which dict to drop columns from ("main" by default).

    Returns:
        pd.DataFrame or (pd.DataFrame, dict): Updated structure.
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

        return df, dicts

    else:  # plain DataFrame
        return data.drop(columns=[c for c in cols_to_drop if c in data.columns])
