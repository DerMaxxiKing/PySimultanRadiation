import numpy as np
import datetime
import pandas as pd
from pandas.io import sql
from sqlalchemy import Table, Column, Integer, String, MetaData, Boolean, DateTime
import psycopg2
import sys
import pickle

from sqlalchemy import Table, Column, MetaData, LargeBinary
from .logger import logger
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql


class DBInterface(object):

    def __init__(self, *args, **kwargs):

        self.engine = kwargs.get('engine')
        self._object_table = None

    @property
    def object_table(self):
        if self._object_table is None:
            df_table = self.get_table('dfs')
            if df_table is None:
                self.create_df_table()
            df_table = self.get_table('dfs')

            self._object_table = df_table
        return self._object_table

    @property
    def tables(self):
        metadata = MetaData(self.engine)
        metadata.reflect()
        return metadata.tables

    @property
    def object_names(self):
        names = [r for r, in self.object_table.select().with_only_columns(self.object_table.c.id).execute().fetchall()]
        return names

    def get_table(self, tablename):

        if tablename in self.tables.keys():
            return self.tables[tablename]
        else:
            return None

    def create_df_table(self):

        metadata = MetaData(self.engine)
        metadata.reflect()

        if 'dfs' not in metadata.tables.keys():  # If table don't exist, Create.

            # Create a table with the appropriate Columns
            objects_table = Table('dfs', metadata,
                                     Column('id', String, primary_key=True, nullable=False),
                                     Column('df_pickle', LargeBinary),
                                     Column('created', DateTime, default=datetime.datetime.utcnow),
                                     Column('type', String)
                                     )
            metadata.create_all(tables=[objects_table], checkfirst=True)
        else:
            raise Exception(f'Table dfs already exists')

    def get_dataframe(self, name):
        df = self.object_table.select().where(self.object_table.c.id == name).execute().fetchone()
        if df is None:
            return
        return pickle.loads(df.df_pickle)

    def object_exists(self, name):
        return name in self.object_names

    def save_object(self, obj, name, if_exists='drop'):
        if not self.object_exists(name):
            result = self.object_table.insert().values(id=name,
                                                       df_pickle=pickle.dumps(obj),
                                                       type=str(type(obj))).execute()
        else:
            if if_exists == 'drop':
                result = self.object_table.update().where(self.object_table.c.id == name).values(df_pickle=pickle.dumps(obj)).execute()

    def load_object(self, name):
        obj = self.object_table.select().where(self.object_table.c.id == name).execute().fetchone()
        if obj is None:
            return
        return pickle.loads(obj.df_pickle)

    def save_dataframe(self, df, name, if_exists='drop'):
        # check if name exists:

        if not self.object_exists(name):
            logger.debug(f'Database: {name} does not exist. Create new entry....')
            result = self.object_table.insert().values(id=name,
                                                       df_pickle=pickle.dumps(df),
                                                       type=str(type(df))).execute()
        else:
            logger.debug(f'Database: {name} exist. if_exists is {if_exists}')
            if if_exists == 'drop':
                logger.debug(f'Database: {name} exists. overwriting')
                result = self.object_table.update().where(self.object_table.c.id == name).values(df_pickle=pickle.dumps(df),
                                                                                                 type=str(type(df))).execute()
            elif if_exists == 'append':
                logger.debug(f'Database: {name} exists. appending')

        # with self.engine.connect() as conn:
        #     result = conn.execute(ins)
