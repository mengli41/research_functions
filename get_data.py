#from database import indexdb
from sqlalchemy import create_engine
import os
import pandas as pd

pgsql6 = 'postgresql+psycopg2://limeng:123456@192.168.1.119'
indexdb = create_engine('{}/indexdb'.format(pgsql6))

def get_data(date_start=None, file_load='data.msg', file_save='data.msg',
             con=indexdb,is_update=False):
    """Download daily data from indexdb_adjusted@192.168.1.6.
    """
    if  is_update:
        data_old = None
        if file_load is not None and os.path.exists(file_load):
            data_old = pd.read_msgpack(file_load)
            if date_start is None:
                date_start = data_old.major_axis.max().strftime('%Y%m%d')

        cmd = """select * from daily"""
        if date_start is not None:
            cmd += """ where "trading_day">='{}'""".format(date_start)

        data_new_raw = pd.read_sql(cmd, con, parse_dates=['trading_day'])\
                .set_index('trading_day')
        data_new_dict = dict(list(data_new_raw.groupby('symboltype')))
        data_new = pd.Panel(data_new_dict).transpose(2, 1, 0).drop('update_time')

        for field in data_new.items:
            if field != 'symboltype':
                data_new[field] = data_new[field].astype(float)
        data_new.drop('symboltype', inplace=True)

        if data_old is not None:
            data = pd.concat([data_old.loc[:, data_old.major_axis < date_start],
                              data_new], axis=1)
        else:
            data = data_new

        if file_save is not None:
            data.to_msgpack(file_save)
    else:
        data_old = None
        if file_load is not None and os.path.exists(file_load):
            data_old = pd.read_msgpack(file_load)
        data = data_old

    return data

if __name__ == '__main__':
    get_data(is_update=True)
