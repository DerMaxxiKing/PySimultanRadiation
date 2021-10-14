from pandas.io import sql


def df_interpolate_at(df, ts, method='linear', axis='index'):
    return df.reindex(df.index.union(ts)).sort_index().interpolate(method=method, axis=axis).loc[ts]


def write_df_in_empty_table(df, tablename, engine, if_exists='append', index=True):
    """

    :param df: dataframe to write
    :param tablename: name of the table
    :param engine: engine
    :param if_exists: default: 'append'
    :param index: default: True
    """
    sql.execute(f'DROP TABLE IF EXISTS {tablename}', engine)
    if df.shape.__len__() > 1:
        if df.shape[1] > 1600:
            raise ValueError(f'Dataframe has to many columns: {df.shape[1]}. Tables can have at most 1600 columns')
    df.to_sql(tablename, engine, if_exists=if_exists, index=index)
