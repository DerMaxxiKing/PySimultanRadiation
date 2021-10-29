import numpy as np
import datetime
import pandas as pd
from pandas.io import sql
from sqlalchemy import Table, Column, Integer, String, MetaData, Boolean, DateTime
import psycopg2
import sys
import pickle

from sqlalchemy import Table, Column, MetaData, LargeBinary
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

def df_interpolate_at(df, ts, method='linear', axis='index'):
    return df.reindex(df.index.union(ts)).sort_index().interpolate(method=method, axis=axis).loc[ts]

def write_df_in_empty_table(df, tablename, engine, if_exists='append', index=True, dtype={}):
    """

    :param df: dataframe to write
    :param tablename: name of the table
    :param engine: engine
    :param if_exists: default: 'append'
    :param index: default: True
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    sql.execute(f'DROP TABLE IF EXISTS {tablename}', engine)
    if df.shape.__len__() > 1:
        if df.shape[1] > 1600:
            raise ValueError(f'Dataframe has to many columns: {df.shape[1]}. Tables can have at most 1600 columns')

    df.to_sql(tablename, engine, if_exists=if_exists, index=index, dtype=dtype)


def write_dataframe_meta(tablename, columns, engine):

    metadata = MetaData(engine)
    metadata.reflect()

    if 'dfs' not in metadata.tables.keys():  # If table don't exist, Create.

        # Create a table with the appropriate Columns
        df_columns_table = Table('dfs', metadata,
                                 Column('id', String, primary_key=True, nullable=False),
                                 Column('df_columns', postgresql.ARRAY(String)),
                                 Column('created', DateTime, default=datetime.datetime.utcnow),
                                 Column('type', String)
                                 )
        metadata.create_all(tables=[df_columns_table], checkfirst=True)

    df_columns_table = metadata.tables['dfs']
    ins = df_columns_table.insert().values(id=tablename,
                                           df_columns=columns.values.tolist())
    with engine.connect() as conn:
        result = conn.execute(ins)


def get_dataframe_columns(tablename, engine):

    metadata = MetaData(engine)
    metadata.reflect()

    if 'dfs' not in metadata.tables.keys():
        return None
    else:
        df_columns_table = metadata.tables['dfs']

    sel = df_columns_table.select().where(df_columns_table.c.Id == tablename)
    with engine.connect() as conn:
        result = conn.execute(sel).fetchone()

    return result.df_columns
