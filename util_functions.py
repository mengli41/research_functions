# encoding: UTF-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as ts
import datetime as dt
import inspect
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os, sys
import copy

from numpy import abs
from collections import defaultdict
from types import FunctionType
from scipy.stats import rankdata
from matplotlib.dates import date2num
from functools import wraps
from sqlalchemy import create_engine
#from datetime import time

from statsmodels.compat import lzip
from statsmodels import regression
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose


###############################################################################
###############################################################################
# Read data from SQL
#------------------------------------------------------------------------------
def read_intraday_data_from_sql(items, freq_indicator, 
                                date_start = '20150101', 
                                date_end = None, iprint = 0):
    db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb')
#        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb_adjusted')
    
    columns = ['datetime', 'open', 'high', 'low', 'close', 
               'volume', 'opint as openint', 'adjustment', 
               #'vwap', 
               'symboltype', 'symbol']
    
    columns_place = ",".join(columns)
    data_base_name = 'data_con_' + freq_indicator
    tickers_place = "'" + "', '".join(items) + "'"
    
    cmd = """select {0} from "{1}" where "symboltype" in ({2})""".format(
        columns_place, data_base_name, tickers_place)
    
    if date_start is not None:
        cmd += " and trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)
        
    df = pd.read_sql(cmd, db)
    df.datetime = pd.to_datetime(df.datetime)
    
    df['close'] = df['close'] / df['adjustment']
    df['open'] = df['open'] / df['adjustment']
    df['low'] = df['low'] / df['adjustment']
    df['high'] = df['high'] / df['adjustment']
#    df['vwap'] = df['vwap'] / df['adjustment']
#    df['amount'] = df['vwap'] * df['volume']

    df.rename(columns = {'symbol': 'symbol_con'}, inplace = True)
    
    df.set_index(['symboltype', 'datetime'], inplace = True)
    df.sort_index(level = 0, inplace = True, sort_remaining = True)
    df.index.names = ['symbol', 'datetime']
    df = df.unstack(0)
    df.columns.names = ['type', 'symbol']

    return df

#------------------------------------------------------------------------------
def read_daily_data_from_sql(items, date_start = '20150101', 
                             date_end = None, iprint = 0):
    db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb')
#        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb_adjusted')
    columns = ['trading_day as datetime', 'open', 'high', 'low', 'close', 
               'volume', 'opint as openint', 'vwap', 'adjustment', 
               'symboltype'] 

    tickers_place = "'" + "', '".join(items) + "'"
    columns_place = ",".join(columns)
    
    cmd = """select {0} from "daily_con" where "symboltype" in ({1})""".format(
        columns_place, tickers_place)
    
    if date_start is not None:
        cmd += " and trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)
    
    df = pd.read_sql(cmd, db)
    df.datetime = pd.to_datetime(df.datetime)
    
    df['raw_close'] = df['close'].copy()
    df['close'] = df['close'] / df['adjustment']
    df['open'] = df['open'] / df['adjustment']
    df['low'] = df['low'] / df['adjustment']
    df['high'] = df['high'] / df['adjustment']
    df['vwap'] = df['vwap'] / df['adjustment']
    df['amount'] = df['vwap'] * df['volume']
    
    df.set_index(['symboltype', 'datetime'], inplace = True)
    df.sort_index(level = 0, inplace = True, sort_remaining = True)
    df.index.names = ['symbol', 'datetime']
    df = df.unstack(0)
    df.columns.names = ['type', 'symbol']

    return df

#------------------------------------------------------------------------------
def read_all_intraday_data_from_sql(
    freq_indicator, date_start = '20150101', date_end = None, iprint = 0):
    db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb')
#        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb_adjusted')
    
    columns = ['datetime', 'open', 'high', 'low', 'close', 
               'volume', 'opint as openint', 'adjustment', #'vwap', 
               'symboltype']

    data_base_name = 'data_con_' + freq_indicator
    columns_place = ",".join(columns)
    
    cmd = """select {0} from "{1}" """.format(
        columns_place, data_base_name)
    
    if date_start is not None:
        cmd += "where trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)
        
    df = pd.read_sql(cmd, db)
    df.datetime = pd.to_datetime(df.datetime)
    
    df['close'] = df['close'] / df['adjustment']
    df['open'] = df['open'] / df['adjustment']
    df['low'] = df['low'] / df['adjustment']
    df['high'] = df['high'] / df['adjustment']
#    df['vwap'] = df['vwap'] / df['adjustment']
#    df['amount'] = df['vwap'] * df['volume']

    df.set_index(['symboltype', 'datetime'], inplace = True)
    df.sort_index(level = 0, inplace = True, sort_remaining = True)
    df.index.names = ['symbol', 'datetime']
    df = df.unstack(0)
    df.columns.names = ['type', 'symbol']

    return df

#------------------------------------------------------------------------------
def read_all_daily_data_from_sql(
    date_start = '20150101', date_end = None, iprint = 0):
    db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb')
#        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb_adjusted')

    columns = ['trading_day as datetime', 'open', 'high', 'low', 'close', 
               'vwap', 'volume', 'opint as openint', 'adjustment', 
               'symboltype'] 

    columns_place = ",".join(columns)
    data_base_name = 'daily_con'
    
    cmd = """select {0} from "{1}" """.format(
        columns_place, data_base_name)
    
    if date_start is not None:
        cmd += "where trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)
    
    df = pd.read_sql(cmd, db)
    df.datetime = pd.to_datetime(df.datetime)
    
    df['raw_close'] = df['close'].copy()
    df['close'] = df['close'] / df['adjustment']
    df['open'] = df['open'] / df['adjustment']
    df['low'] = df['low'] / df['adjustment']
    df['high'] = df['high'] / df['adjustment']
    df['vwap'] = df['vwap'] / df['adjustment']
    df['amount'] = df['vwap'] * df['volume']
    
    df.set_index(['symboltype', 'datetime'], inplace = True)
    df.sort_index(level = 0, inplace = True, sort_remaining = True)
    df.index.names = ['symbol', 'datetime']
    df = df.unstack(0)
    df.columns.names = ['type', 'symbol']

    return df

#------------------------------------------------------------------------------
def read_all_intraday_data_v2(
    freq_indicator, columns, 
    date_start = '20150101', date_end = None, iprint = 0):
    db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb')
#        'postgresql+psycopg2://limeng:123456@192.168.1.119/indexdb_adjusted')
    
#    columns = ['datetime', 'open', 'high', 'low', 'close', 
#               'volume', 'opint as openint', 'adjustment', #'vwap', 
#               'symboltype']

    data_base_name = 'data_con_' + freq_indicator
    columns_place = ",".join(columns)
    
    cmd = """select {0} from "{1}" """.format(
        columns_place, data_base_name)
    
    if date_start is not None:
        cmd += "where trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)
        
    df = pd.read_sql(cmd, db)
    df.datetime = pd.to_datetime(df.datetime)
    
#    df['close'] = df['close'] / df['adjustment']
#    df['open'] = df['open'] / df['adjustment']
#    df['low'] = df['low'] / df['adjustment']
#    df['high'] = df['high'] / df['adjustment']
#    df['vwap'] = df['vwap'] / df['adjustment']
#    df['amount'] = df['vwap'] * df['volume']

    df.set_index(['symboltype', 'datetime'], inplace = True)
    df.sort_index(level = 0, inplace = True, sort_remaining = True)
    df.index.names = ['symbol', 'datetime']
    df = df.unstack(0)
    df.columns.names = ['type', 'symbol']

    return df

#------------------------------------------------------------------------------
def read_tick_data_from_sql(contract_type, date_start, date_end, iprint = 0):
    db = create_engine(
        'postgresql+psycopg2://readonly:123456@192.168.1.120/ticks_timescale')
#        'postgresql+psycopg2://limeng:123456@192.168.1.188/ticks_timescale')

    database_name = contract_type

    cmd = """select * from "{0}" """.format(database_name)

    if date_start is not None:
        cmd += "where datetime>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and datetime<='{0}'".format(date_end)
    if iprint:
        print(cmd)

    df = pd.read_sql(cmd, db)
    df.datetime = pd.to_datetime(df.datetime)
    df.set_index('datetime', inplace = True)
    df.sort_index(inplace = True)

    return df

#------------------------------------------------------------------------------
def read_macro_data_from_sql(
    macro_table_name = 'macros', 
    date_start = None, date_end = None, iprint = 0):

    macro_db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/macro_data')

    columns = ['trading_day as datetime', 'value', 'macro']
    columns_place = ",".join(columns)
    cmd = """select {0} from "{1}" """.format(columns_place, macro_table_name)

    if date_start is not None:
        cmd += "where trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)

    macro_df = pd.read_sql(cmd, macro_db)
    macro_df.datetime = pd.to_datetime(macro_df.datetime)

    macro_df.set_index(['macro', 'datetime'], inplace = True)
    macro_df.sort_index(level = 1, inplace = True, sort_remaining = True)
    macro_df.index.names = ['macro', 'datetime']
    macro_df = macro_df.unstack(0)['value']

    return macro_df

#------------------------------------------------------------------------------
def read_macro_beta_from_sql(
    database_name = 'us_ir_betas', 
    date_start = None, date_end = None, iprint = 0):

    db = create_engine(
        'postgresql+psycopg2://limeng:123456@192.168.1.119/comm_factors')

    columns = ['trading_day as datetime', 'symboltype', 'value', 'label']
    columns_place = ",".join(columns)

    cmd = """select {0} from "{1}" """.format(
        columns_place, database_name)

    if date_start is not None:
        cmd += "where trading_day>='{0}'".format(date_start)
    if date_end is not None:
        cmd += " and trading_day<='{0}'".format(date_end)
    if iprint:
        print(cmd)

    df = pd.read_sql(cmd, db)

    df.set_index(['label', 'symboltype', 'datetime'], inplace = True)
    df.sort_index(level = 1, inplace = True, sort_remaining = True)
    df.index.names = ['label', 'symbol', 'datetime']
    df = df.unstack(0).unstack(0)['value']

    return df

#------------------------------------------------------------------------------
def read_daily_data_from_msg(data_address_str):
    tmp = pd.read_msgpack(data_address_str)
    tmp_df = tmp.to_frame()

    tmp_df['close'] = tmp_df['close'] / tmp_df['adjustment']
    tmp_df['open'] = tmp_df['open'] / tmp_df['adjustment']
    tmp_df['low'] = tmp_df['low'] / tmp_df['adjustment']
    tmp_df['high'] = tmp_df['high'] / tmp_df['adjustment']
    tmp_df['vwap'] = tmp_df['vwap'] / tmp_df['adjustment']
    tmp_df['amount'] = tmp_df['vwap'] * tmp_df['volume']

    df = tmp_df.unstack(1)

    return df


###############################################################################
###############################################################################
# Data Preprocess
#------------------------------------------------------------------------------
def industry_demean_return(return_df, industry_dict, 
                           industry_neutral_list = [], 
                           if_all_industry = True): 
    total_industry_demean_df = return_df.copy()
    
    if if_all_industry:
        for key,value in industry_dict.items():
            industry_df = return_df[value]
            industry_demean_df = industry_df.sub(
                industry_df.mean(axis = 1), axis = 0)
            total_industry_demean_df[value] = industry_demean_df
    else:
        if industry_neutral_list == []:
            print 'Please provide industry list!'
        else:
            for key in industry_neutral_list:
                if key in industry_dict.keys():
                    value = industry_dict[key]
                    industry_df = return_df[value]
                    industry_demean_df = industry_df.sub(
                        industry_df.mean(axis = 1), axis = 0)
                    total_industry_demean_df[value] = industry_demean_df
                else:
                    print 'Industry not in industry_dict!'

    return total_industry_demean_df

#------------------------------------------------------------------------------
def get_liquid_multiindex_data(data_df, liquid_contract_df):
    liquid_data_df = data_df.copy()

    for column in liquid_contract_df.columns:
        if column in liquid_data_df.columns.get_level_values(1):
            not_liquid_index = (
                liquid_contract_df.loc[
                    liquid_contract_df.loc[liquid_data_df.index, column] == 0, 
                    column].index)
            liquid_data_df.loc[
                not_liquid_index, 
                liquid_data_df.columns.get_level_values(1) == column] = np.nan

    return liquid_data_df

#------------------------------------------------------------------------------
def get_liquid_contract_data(data_df, liquid_contract_df):
    liquid_data_df = data_df.copy()

    for column in liquid_contract_df.columns:
        if column in liquid_data_df.columns:
            liquid_data_df.loc[:, column] = np.where(
                liquid_contract_df.loc[
                    liquid_data_df.index, column] == 0, 
                np.nan, liquid_data_df.loc[:, column])
    
    return liquid_data_df

#------------------------------------------------------------------------------
def filter_extreme_MAD(factor_df, n):
    median = factor_df.median(axis = 1)
    new_median = factor_df.sub(
        median, axis = 0).abs().median(axis = 1)

    max_range = median + n * new_median
    min_range = median - n * new_median

    processed_factor = factor_df.copy()
    for date in processed_factor.index:
        processed_factor.loc[date] = np.clip(
            processed_factor.loc[date], 
            min_range.loc[date], max_range.loc[date])

    return processed_factor

#------------------------------------------------------------------------------
def panel_data_standardization(df):
    df = df.sub(
        df.mean(axis = 1), axis = 0).div(
        df.std(axis = 1), axis = 0)
    
    return df

#------------------------------------------------------------------------------
def panel_data_demean(df):
    df = df.sub(df.mean(axis = 1), axis = 0)
    
    return df

#------------------------------------------------------------------------------
def cs_to_panel_data_transformation(df, var_name):
    df = pd.DataFrame(df.unstack(0))
    df.columns = [var_name]

    return df

#------------------------------------------------------------------------------
def cs_to_panel_data_standardization(df, var_name):
    df = df.sub(
        df.mean(axis = 1), axis = 0).div(
        df.std(axis = 1), axis = 0)

    df = pd.DataFrame(df.unstack(0))
    df.columns = [var_name]

    return df

#------------------------------------------------------------------------------
def cs_to_panel_data_demean(df, var_name):
    df = df.sub(df.mean(axis = 1), axis = 0)
    df = pd.DataFrame(df.unstack(0))
    df.columns = [var_name]

    return df

#------------------------------------------------------------------------------
def rolling_standardize(data, series_var, window = 20):
    series_mean = data[series_var].rolling(window = window).mean()
    series_std = data[series_var].rolling(window = window).std()
    standardized_series = (data[series_var] - series_mean) / series_std
    return standardized_series

#------------------------------------------------------------------------------
def rolling_normalization(data, series_var, window = 20):
    series_max = data[series_var].rolling(window = window).max()
    series_min = data[series_var].rolling(window = window).min()
    normalized_series = ((data[series] - series_min) 
                         / (series_max - series_min))
    return normalized_series

#------------------------------------------------------------------------------
def forward_feature_selection(model_data, dep_var, standard):
    model_df = model_data
    response = dep_var
    standard = standard

    remaining = set(model_df.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining: 
        scores_with_candidates = []
        for candidate in remaining:

            X1 = model_df[selected + [candidate]]
            X = sm.add_constant(X1)
            Y = model_df[response]
            model = regression.linear_model.OLS(Y, X).fit()

            if standard == 'rsquared_adj':
                score = model.rsquared_adj
            
            if standard == 'aic':
                score = -model.aic
                
            if standard == 'bic':
                score = -model.bic

            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            selected.append(best_candidate)
            current_score = best_new_score
        remaining.remove(best_candidate)
    
    return selected

#------------------------------------------------------------------------------
def backward_feature_selection(model_data, dep_var, standard):
    model_df = model_data
    response = dep_var
    standard = standard

    remaining = list(set(model_df.columns))
    remaining.remove(response)

    current_score, best_new_score = 0.0, 0.0
    keep = True
    
    while keep: 
        scores_with_candidates = []
        temp_remaining = copy.copy(remaining)
        for candidate in temp_remaining:
            keep_candidate = copy.copy(temp_remaining)
            keep_candidate.remove(candidate)

            X1 = model_df[keep_candidate]
            X = sm.add_constant(X1)
            Y = model_df[response]
            model = regression.linear_model.OLS(Y, X).fit()

            if standard == 'rsquared_adj':
                score = model.rsquared_adj
            
            if standard == 'aic':
                score = -model.aic
                
            if standard == 'bic':
                score = -model.bic

            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            current_score = best_new_score
        else:
            keep = False
    
    return remaining

#------------------------------------------------------------------------------
def preprocess_tick_data(data, if_night, day_start_time, day_end_time, 
                         night_start_time = None, night_end_time = None):
    use_df = data.loc[:, ['交易日', '合约代码', '最新价', '数量', 
                          '成交金额', '持仓量', '最后修改时间', '最后修改毫秒', 
                          '申卖价一', '申卖量一', 
                          '申买价一', '申买量一']].copy()
    use_df.columns = ['tradingday', 'symbol', 'last_price', 'volume', 
                      'turnover', 'openint', 'time', 'second', 
                      'ask_price1', 'ask_volume1', 
                      'bid_price1', 'bid_volume1']

    # Drop the observations where either the bid price or the ask price is 0.
    # These cases indicate either there are data errors or the contract price 
    # reach the stop price.
    use_df = use_df.loc[
        (use_df['bid_price1'] != 0.0) & (use_df['ask_price1'] != 0.0), :]

    use_df['mid_price'] = (use_df['bid_price1'] + use_df['ask_price1']) / 2

    use_df = use_df.reset_index(drop = True)

    if if_night:
        use_df = use_df.loc[
            (((use_df['time'] >= night_start_time) 
              & (use_df['time'] <= night_end_time))
             | ((use_df['time'] >= day_start_time) 
                & (use_df['time'] <= day_end_time))), :]
    else:
        use_df = use_df.loc[
            ((use_df['time'] >= day_start_time) 
             & (use_df['time'] <= day_end_time)), :]

    use_df = use_df.sort_index().reset_index(drop = True)

    use_df['second_rank'] = np.where(
        use_df['time'] == use_df['time'].shift(), 2, 1)

    use_df['datetime'] = (use_df['tradingday'].map(str) + ' ' 
                          + use_df['time'].map(str) + ' ' 
                          + use_df['second_rank'].map(str))
    use_df = use_df.reset_index(drop = True).set_index('datetime')

    use_df = use_df[['symbol', 'last_price', 'volume', 'turnover', 
                     'openint', 'ask_price1', 'ask_volume1', 
                     'bid_price1', 'bid_volume1', 'mid_price']]

    return use_df


###############################################################################
###############################################################################
# Rolling Estimation and Prediction
#------------------------------------------------------------------------------
def generate_training_backtest_periods(
    total_range, test_range, 
    backtest_panel_size, training_panel_size, prediction_size):

    backtest_date_list = []
    training_date_list = []

    backtest_period = test_range
    training_period = total_range

    backtest_start_index = 0

    while backtest_start_index < len(backtest_period):
        end_index = min(len(backtest_period) - 1,
                        backtest_start_index + backtest_panel_size - 1)

        backtest_start_date = backtest_period[backtest_start_index]
        backtest_end_date = backtest_period[end_index]

        backtest_date_list.append([backtest_start_date, backtest_end_date])

        backtest_start_index += backtest_panel_size

        # build the training date list based on the backtest end date
        training_end_index = training_period.index(backtest_end_date)
        training_start_index = max(
            0, training_end_index - training_panel_size + 1)

        training_start_date = training_period[training_start_index]
        training_end_date = training_period[training_end_index]

        training_date_list.append([training_start_date, training_end_date])
    
    periods_dict = {'training_date_list': training_date_list, 
                    'backtest_date_list': backtest_date_list}

    return periods_dict

#------------------------------------------------------------------------------
def normal_rolling_model_building(
    pv_feature_df, close_df, 
    target_vars, ind_vars, 
    prediction_size, liquid_contract_df, 
    training_date_list, backtest_date_list, 
    rolling_params = False, print_dates = False):

    dummy_model_list = []
    dummy_factor_return_df = pd.DataFrame()
    dummy_factor_return_t_value_df = pd.DataFrame()
    dummy_prediction_df = pd.DataFrame()
    trading_commodity_list = pv_feature_df.index.levels[0]

    for i in range(0, len(backtest_date_list)-1):
        #----------------------------------------------------------------------
        # cross-sectional estimation
        panel_start_date = training_date_list[i][0]
        panel_end_date = training_date_list[i][1]

        #----------------------------------------------------------------------
        # Build the DataFrame containing all the features.
        model_use_df = pv_feature_df.loc[
            (slice(None), slice(panel_start_date, panel_end_date)), ind_vars]

        #----------------------------------------------------------------------
        model_close_df = close_df.loc[panel_start_date:panel_end_date, 
                                      trading_commodity_list]
        model_return_df = pd.DataFrame(
            (np.log(model_close_df.shift(-prediction_size))
             - np.log(model_close_df)))

        model_return_df = get_liquid_contract_data(
            model_return_df, liquid_contract_df)

        # Use original return as the dependent variable.
        model_original_return_df = pd.DataFrame(
            model_return_df.unstack(0), columns = ['original_return'])

        #----------------------------------------------------------------------
        final_model_df = pd.concat(
            [model_use_df, model_original_return_df], axis = 1)
        final_model_df = final_model_df.dropna()

        dep_var = 'original_return'
        mod_ind_vars = [ele for ele in final_model_df.columns 
                        if ele != dep_var]

        X = sm.add_constant(final_model_df[mod_ind_vars].astype(np.float64))
        Y = final_model_df[dep_var].astype(np.float64)

        model = regression.linear_model.OLS(Y, X).fit()

        dummy_model_list.append(model)

        temp_factor_return_df = pd.DataFrame(model.params).T
        temp_factor_return_df['start_date'] = panel_start_date
        temp_factor_return_df['end_date'] = panel_end_date
        dummy_factor_return_df = pd.concat(
            [dummy_factor_return_df, temp_factor_return_df])

        temp_factor_return_t_value_df = pd.DataFrame(model.tvalues).T
        temp_factor_return_t_value_df['start_date'] = panel_start_date
        temp_factor_return_t_value_df['end_date'] = panel_end_date
        dummy_factor_return_t_value_df = pd.concat(
            [dummy_factor_return_t_value_df, temp_factor_return_t_value_df])

        #----------------------------------------------------------------------
        test_start_date = backtest_date_list[i+1][0]
        test_end_date = backtest_date_list[i+1][1]

        if print_dates:
            print 'training: ', panel_start_date, panel_end_date
            print 'testing: ', test_start_date, test_end_date

        test_df = pv_feature_df.loc[
            (slice(None), slice(test_start_date, test_end_date)), mod_ind_vars]
        
        #----------------------------------------------------------------------
        if rolling_params: 
            prediction_df = pd.DataFrame(
                test_df[target_vars].multiply(
                    dummy_factor_return_df[target_vars].ewm(
                        span = 5).mean().tail(1).loc[0, :]).sum(
                            axis = 1, skipna = False), 
                columns = ['prediction'])
        else: 
            prediction_df = pd.DataFrame(
                test_df[target_vars].multiply(
                    model.params[target_vars]).sum(axis = 1, skipna = False),
                columns = ['prediction'])

        dummy_prediction_df = pd.concat(
            [dummy_prediction_df, prediction_df.unstack(0)], axis = 0)

    result_dict = {
        'model_list': dummy_model_list, 
        'factor_return_df': dummy_factor_return_df, 
        'factor_return_t_value_df': dummy_factor_return_t_value_df, 
        'prediction_df': dummy_prediction_df}

    return result_dict

#------------------------------------------------------------------------------
def rolling_model_building_with_industry_neutral(
    pv_feature_df, close_df, 
    target_vars, ind_vars, 
    prediction_size, liquid_contract_df, 
    training_date_list, backtest_date_list, 
    industry_dict, industry_neutral_list = [], 
    if_all_industry = True, 
    rolling_params = False, print_dates = False):

    dummy_model_list = []
    dummy_factor_return_df = pd.DataFrame()
    dummy_factor_return_t_value_df = pd.DataFrame()
    dummy_prediction_df = pd.DataFrame()
    trading_commodity_list = pv_feature_df.index.levels[0]

    for i in range(0, len(backtest_date_list)-1):
        #----------------------------------------------------------------------
        # cross-sectional estimation
        panel_start_date = training_date_list[i][0]
        panel_end_date = training_date_list[i][1]

        #----------------------------------------------------------------------
        # Build the DataFrame containing all the features.
        model_use_df = pv_feature_df.loc[
            (slice(None), slice(panel_start_date, panel_end_date)), ind_vars]

        #----------------------------------------------------------------------
        model_close_df = close_df.loc[panel_start_date:panel_end_date, 
                                      trading_commodity_list]
        model_return_df = pd.DataFrame(
            (np.log(model_close_df.shift(-prediction_size))
             - np.log(model_close_df)))

        model_return_df = get_liquid_contract_data(
            model_return_df, liquid_contract_df)

        # Remove the industry mean return from the individual commodities.
        model_return_df = industry_demean_return(
            model_return_df, industry_dict, 
            industry_neutral_list = industry_neutral_list, 
            if_all_industry = if_all_industry)

        # Use original return as the dependent variable.
        model_original_return_df = pd.DataFrame(
            model_return_df.unstack(0), columns = ['original_return'])

        #----------------------------------------------------------------------
        final_model_df = pd.concat(
            [model_use_df, model_original_return_df], axis = 1)
        final_model_df = final_model_df.dropna()

        dep_var = 'original_return'
        mod_ind_vars = [ele for ele in final_model_df.columns 
                        if ele != dep_var]

        X = sm.add_constant(final_model_df[mod_ind_vars].astype(np.float64))
        Y = final_model_df[dep_var].astype(np.float64)

        model = regression.linear_model.OLS(Y, X).fit()

        dummy_model_list.append(model)

        temp_factor_return_df = pd.DataFrame(model.params).T
        temp_factor_return_df['start_date'] = panel_start_date
        temp_factor_return_df['end_date'] = panel_end_date
        dummy_factor_return_df = pd.concat(
            [dummy_factor_return_df, temp_factor_return_df])

        temp_factor_return_t_value_df = pd.DataFrame(model.tvalues).T
        temp_factor_return_t_value_df['start_date'] = panel_start_date
        temp_factor_return_t_value_df['end_date'] = panel_end_date
        dummy_factor_return_t_value_df = pd.concat(
            [dummy_factor_return_t_value_df, temp_factor_return_t_value_df])

        #----------------------------------------------------------------------
        test_start_date = backtest_date_list[i+1][0]
        test_end_date = backtest_date_list[i+1][1]

        if print_dates:
            print 'training: ', panel_start_date, panel_end_date
            print 'testing: ', test_start_date, test_end_date

        test_df = pv_feature_df.loc[
            (slice(None), slice(test_start_date, test_end_date)), mod_ind_vars]
        
        #----------------------------------------------------------------------
        if rolling_params: 
            prediction_df = pd.DataFrame(
                test_df[target_vars].multiply(
                    dummy_factor_return_df[target_vars].ewm(
                        span = 5).mean().tail(1).loc[0, :]).sum(
                            axis = 1, skipna = False), 
                columns = ['prediction'])
        else: 
            prediction_df = pd.DataFrame(
                test_df[target_vars].multiply(
                    model.params[target_vars]).sum(axis = 1, skipna = False),
                columns = ['prediction'])

        dummy_prediction_df = pd.concat(
            [dummy_prediction_df, prediction_df.unstack(0)], axis = 0)

    result_dict = {
        'model_list': dummy_model_list, 
        'factor_return_df': dummy_factor_return_df, 
        'factor_return_t_value_df': dummy_factor_return_t_value_df, 
        'prediction_df': dummy_prediction_df}

    return result_dict



###############################################################################
###############################################################################
# Strategy Backtest
class StrategyBacktest:
    #--------------------------------------------------------------------------
    def __init__(self, prediction_df, trans_cost, 
                 intraday_return_df, interday_return_df):
        self.prediction_df = prediction_df
        self.trans_cost = trans_cost
        self.intraday_return_df = intraday_return_df
        self.interday_return_df = interday_return_df
        self.pos_df = None

    #--------------------------------------------------------------------------
    def rolling_prediction(self, rolling_window, ewm_rolling = True):
        if ewm_rolling:
            self.prediction_df = self.prediction_df.ewm(
                span = rolling_window).mean()
        else:
            self.prediction_df = self.prediction_df.rolling(
                window = rolling_window).mean()

    #--------------------------------------------------------------------------
    def rolling_pos(self, rolling_window, ewm_rolling = False):
        if ewm_rolling:
            self.pos_df = self.pos_df.ewm(span = rolling_window).mean()
        else:
            self.pos_df = self.pos_df.rolling(window = rolling_window).mean()

    #--------------------------------------------------------------------------
    def generate_normal_pos(self, compare_er_to_tc = True):
        if type(self.pos_df) == pd.core.frame.DataFrame:
            self.pos_df = None

#        total_pos_df = self.prediction_df.shift()
        total_pos_df = self.prediction_df.dropna(axis = 0, how = 'all')
#        total_pos_df = self.prediction_df.copy()
        pos_df = total_pos_df.div(total_pos_df.abs().sum(axis = 1), axis = 0)
        pos_df = pos_df.fillna(0)

        trans_cost = [self.trans_cost] * pos_df.shape[1]
#        trans_cost_df = pos_df.diff().abs().multiply(trans_cost)
        trans_cost_df = np.abs(
            pos_df.shift() - pos_df.shift(2)).multiply(trans_cost)
        trans_cost_df['trans_cost'] = trans_cost_df.sum(axis = 1)

        if compare_er_to_tc:
            #------------------------------------------------------------------
            # Check if the expected return based on the current prediction 
            # could cover the transaction costs. If it can't, then keep 
            # the current position, otherwise, modify the position accordingly.
#            expected_return = pos_df.multiply(self.prediction_df.shift())
            expected_return = pos_df.multiply(self.prediction_df.dropna(
                axis = 0, how = 'all'))
            expected_return['total_er'] = expected_return.sum(axis = 1)

            er_tc_gap = pd.DataFrame(
                expected_return['total_er'] - trans_cost_df['trans_cost'], 
                columns = ['er_tc_gap'])

            self.pos_df = pos_df.copy()
            self.pos_df[er_tc_gap['er_tc_gap'] <= 0] = np.nan
            self.pos_df = self.pos_df.fillna(method = 'ffill')
        else:
            self.pos_df = pos_df.copy()

    #--------------------------------------------------------------------------
    def generate_quantile_pos(self, quantile, reverse_ind = 1): 
        if type(self.pos_df) == pd.core.frame.DataFrame:
            self.pos_df = None

        test_feature_df = self.prediction_df.dropna(axis = 0, how = 'all')
#        test_feature_df = self.prediction_df.copy()

        #----------------------------------------------------------------------
        # This is the quantile signal generation function from Zhang. 
        # This function is faster, but it may contain potential errors.
#        rank_df = test_feature_df.T.rank()
#        peak = rank_df.max(axis = 0)
#        signal_df = rank_df * 0
#        signal_df[rank_df <= np.ceil(peak*quantile)] = (
#            -1 * reverse_ind)
#        signal_df[rank_df >= np.ceil(peak*(1-quantile)+1e-6)] = (
#            1 * reverse_ind)
#        total_pos_df = signal_df.T
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        # This is my quantile signal generation function.
        rank_df = test_feature_df.rank(axis = 1)

        low_quantile = rank_df.quantile(
            q = quantile, axis = 1, interpolation = 'lower')
        high_quantile = rank_df.quantile(
            q = (1 - quantile), axis = 1, interpolation = 'higher')

        low_rank_df = rank_df.sub(low_quantile, axis = 0)
        high_rank_df = rank_df.sub(high_quantile, axis = 0)

        signal_df = rank_df.copy()
        signal_df[signal_df.notnull()] = 0
        signal_df[low_rank_df <= 0] = -1 * reverse_ind
        signal_df[high_rank_df >= 0] = 1 * reverse_ind
        signal_df[rank_df.isnull()] = 0 

        total_pos_df = signal_df.copy()
        #----------------------------------------------------------------------

        self.pos_df = (
            total_pos_df.div(total_pos_df.abs().sum(axis = 1), axis = 0))
        self.pos_df = self.pos_df.fillna(0)

    #--------------------------------------------------------------------------
    def generate_sign_pos(self, reverse_ind = 1): 
        if type(self.pos_df) == pd.core.frame.DataFrame:
            self.pos_df = None

        signal_df = self.prediction_df.copy()
        signal_df[self.prediction_df < 0] = -1 * reverse_ind
        signal_df[self.prediction_df > 0] = 1 * reverse_ind

#        total_pos_df = signal_df.shift()
        total_pos_df = signal_df.copy()
        self.pos_df = (
            total_pos_df.div(total_pos_df.abs().sum(axis = 1), axis = 0))
        self.pos_df = self.pos_df.fillna(0)

    #--------------------------------------------------------------------------
    def backtest(self):
        if type(self.pos_df) != pd.core.frame.DataFrame:
            print 'Please generate the position DataFrame first!'
        else:
            trans_cost = [self.trans_cost] * self.pos_df.shape[1]
#            trans_cost_df = self.pos_df.diff().abs().multiply(trans_cost)
            trans_cost_df = np.abs(
                self.pos_df.shift() - self.pos_df.shift(2)).multiply(trans_cost)
            trans_cost_df['trans_cost'] = trans_cost_df.sum(axis = 1)

            self.port_return_df = (
                self.pos_df.shift().multiply(
                    self.intraday_return_df.reindex(self.pos_df.index)) 
                + self.pos_df.shift(2).multiply(
                    self.interday_return_df.reindex(self.pos_df.index)))

            self.port_return_df['port_return'] = (
                self.port_return_df.sum(axis = 1))
            self.port_return_df['net_port_return'] = (
                self.port_return_df['port_return'] 
                - trans_cost_df['trans_cost'])
            self.port_return_df['NAV'] = (
                1 + self.port_return_df['net_port_return'].cumsum())
            self.port_return_df['drawdown'] = (
                self.port_return_df['NAV'].cummax() 
                - self.port_return_df['NAV'])

            self.port_turnover = pd.DataFrame(
                self.pos_df.diff().abs().sum(axis = 1), 
                columns = ['turnover'])


###############################################################################
###############################################################################
# Strategy Performance
#------------------------------------------------------------------------------
def calculate_sharpe_ratio(return_series, risk_free_rate = 0.0):
    sharpe_ratio = ((return_series.mean() - risk_free_rate / 242.0) 
                    / np.std(return_series) * np.sqrt(242.0))
    return sharpe_ratio

#------------------------------------------------------------------------------
def industry_pos_return_sum(pos_df, port_return_df, industry_commodity_dict):
    '''
    This function is used for summing the industry position and returns. 
    The pos_df and port_return_df are the typical DataFrames used for 
    storing the positions and returns of the portfolio.
    The industry_commodity_dict is the dict used for storing the 
    industry-commodity relationship. 
    Note that the returns are before transaction cost.
    '''
    industry_pos_df = pd.DataFrame(index = pos_df.index)
    industry_return_df = pd.DataFrame(index = port_return_df.index)

    for key,value in industry_commodity_dict.items():
        trading_commodities = [ele for ele in value if ele in pos_df.columns]
        industry_pos_df[key] = pos_df[trading_commodities].sum(axis = 1)
        industry_return_df[key] = port_return_df[
            trading_commodities].sum(axis = 1)

    result_dict = {'industry_pos_df': industry_pos_df, 
                   'industry_return_df': industry_return_df}

    return result_dict


###############################################################################
###############################################################################
# Factor Performance
#------------------------------------------------------------------------------
def factor_quantile_test(feature_df, intraday_return_df, interday_return_df):
    test_feature_df = feature_df.dropna(axis = 0, how = 'all')
    rank_df = test_feature_df.rank(axis = 1)

    #--------------------------------------------------------------------------
    quantile_list = np.arange(0.2, 1.0, 0.2)
    feature_quantile_list = []
    for quantile in quantile_list:
        feature_quantile_list.append(rank_df.quantile(q = quantile, axis = 1))

    #--------------------------------------------------------------------------
    quantile_signal_dict = defaultdict()
    for i in range(len(feature_quantile_list)):
        key = '_'.join(['q', str(i)])

        if i == 0:
            low_quantile = feature_quantile_list[i]
            low_rank_df = rank_df.sub(low_quantile, axis = 0)
            
            signal_df = rank_df * 0
            signal_df[low_rank_df < 0] = 1
            signal_df[rank_df.isnull()] = 0
            
            quantile_signal_dict[key] = signal_df
        elif i == (len(feature_quantile_list)-1):
            low_quantile = feature_quantile_list[i-1]
            high_quantile = feature_quantile_list[i]
            
            low_rank_df = rank_df.sub(low_quantile, axis = 0)
            high_rank_df = rank_df.sub(high_quantile, axis = 0)

            signal_df = rank_df * 0
            signal_df[(low_rank_df >= 0) & (high_rank_df <= 0)] = 1
            signal_df[rank_df.isnull()] = 0
            quantile_signal_dict[key] = signal_df
            
            high_signal_df = rank_df * 0
            high_signal_df[high_rank_df > 0] = 1
            high_signal_df[rank_df.isnull()] = 0
            quantile_signal_dict['q_{0}'.format(str(i+1))] = high_signal_df
        else:
            low_quantile = feature_quantile_list[i-1]
            high_quantile = feature_quantile_list[i]
            
            low_rank_df = rank_df.sub(low_quantile, axis = 0)
            high_rank_df = rank_df.sub(high_quantile, axis = 0)

            signal_df = rank_df * 0
            signal_df[(low_rank_df >= 0) & (high_rank_df <= 0)] = 1
            signal_df[rank_df.isnull()] = 0
            quantile_signal_dict[key] = signal_df

    #--------------------------------------------------------------------------
    quantile_port_return_dict = defaultdict()
    quantile_sharpe_dict = defaultdict()

    for key,value in quantile_signal_dict.items():
        pos_df = value.div(value.abs().sum(axis = 1), axis = 0)
        port_return_df = (
            pos_df.shift().multiply(
                intraday_return_df.loc[pos_df.index])
            + pos_df.shift(2).multiply(
                interday_return_df.loc[pos_df.index]))
        port_return_df['port_return'] = port_return_df.sum(axis = 1)
        
        quantile_port_return_dict[key] = port_return_df
        quantile_sharpe_dict[key] = calculate_sharpe_ratio(
            port_return_df['port_return'])

    return quantile_port_return_dict, quantile_sharpe_dict

#------------------------------------------------------------------------------
class FactorQuantileTest:

    #--------------------------------------------------------------------------
    def __init__(self, feature_df):
        self.feature_df = feature_df

        self.quantile_feature_dict = {}
        self.quantile_port_return_dict = {}
        self.quantile_sharpe_dict = {}

    #--------------------------------------------------------------------------
    def __calculate_sharpe_ratio(self, return_series, risk_free_rate):
        sharpe_ratio = ((return_series.mean() - risk_free_rate / 242.0) 
                        / np.std(return_series) * np.sqrt(242.0))
        return sharpe_ratio

    #--------------------------------------------------------------------------
    def generate_quantile_dict(self, quantile_num = 5, 
                               feature_keyword = 'feature'):
        test_feature_df = self.feature_df.dropna(axis = 0, how = 'all')
        rank_df = test_feature_df.rank(axis = 1)
        use_feature_rank = rank_df[rank_df.min(axis = 1) == 1]
        self.dropped_dates = rank_df[rank_df.min(axis = 1) != 1].index

        q_labels = ['{0}_q_{1}'.format(feature_keyword, str(i)) 
                    for i in range(quantile_num)]
        self.categorical_feature_df = use_feature_rank.transform(
            lambda x: pd.qcut(x, quantile_num, labels=q_labels), axis = 1)

        for q_label in q_labels:
            q_dummy_df = 1 * (self.categorical_feature_df == q_label)
            self.quantile_feature_dict[q_label] = q_dummy_df

    #--------------------------------------------------------------------------
    def factor_backtest(self, intraday_return_df, interday_return_df, 
                        risk_free_rate = 0.0):
        for key,value in self.quantile_feature_dict.items():
            value_sum_df = value.abs().sum(axis = 1)
            value_sum_df[value_sum_df == 0.0] = 1.0
            pos_df = value.div(value_sum_df, axis = 0)
            port_return_df = (
                pos_df.shift().multiply(
                    intraday_return_df.loc[pos_df.index])
                + pos_df.shift(2).multiply(
                    interday_return_df.loc[pos_df.index]))
            port_return_df['port_return'] = port_return_df.sum(axis = 1)

            self.quantile_port_return_dict[key] = port_return_df
            self.quantile_sharpe_dict[key] = self.__calculate_sharpe_ratio(
                port_return_df['port_return'], risk_free_rate)

    #--------------------------------------------------------------------------
    def plot_quantile_NAV(self):
        for key,value in self.quantile_port_return_dict.items():
            value['port_return'].cumsum().plot()
        plt.legend(self.quantile_port_return_dict.keys())

    #--------------------------------------------------------------------------
    def plot_quantile_sharpe_dict(self):
        plt.bar(range(len(self.quantile_sharpe_dict)), 
                list(self.quantile_sharpe_dict.values()), 
                align = 'center')
        plt.xticks(range(len(self.quantile_sharpe_dict)), 
                   list(self.quantile_sharpe_dict.keys()))


###############################################################################
###############################################################################
# Plot
#------------------------------------------------------------------------------
def plot_heatmap(data, title = 'Heatmap', show_legend = True,
                 show_labels = True, label_fmt = '.2f',
                 vmin = None, vmax = None, figsize = None,
                 cmap = 'RdYlGn_r', **kwargs):
    """
    Plot a heatmap using matplotlib's pcolor.

    Args:
        * data (DataFrame): DataFrame to plot. Usually small matrix (ex.
            correlation matrix).
        * title (string): Plot title
        * show_legend (bool): Show color legend
        * show_labels (bool): Show value labels
        * label_fmt (str): Label format string
        * vmin (float): Min value for scale
        * vmax (float): Max value for scale
        * cmap (string): Color map  cmap='RdYlGn_r' or  'RdBu'
        * kwargs: Passed to matplotlib's pcolor

    """
    fig, ax = plt.subplots(figsize = figsize)

    heatmap = ax.pcolor(data, vmin = vmin, vmax = vmax, cmap = cmap)
    # for some reason heatmap has the y values backwards....
    ax.invert_yaxis()

    plt.title(title)

    if show_legend:
        fig.colorbar(heatmap)

    if show_labels:
        vals = data.values
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                plt.text(y + 0.5, x + 0.5, format(vals[x, y], label_fmt),
                         horizontalalignment = 'center',
                         verticalalignment = 'center',
                         color = 'w')

    plt.yticks(np.arange(0.5, len(data.index), 1), data.index)
    plt.xticks(np.arange(0.5, len(data.columns), 1), data.columns)

    plt.show()

#------------------------------------------------------------------------------
def plot_NAV_curve(port_return_df):
    i = np.argmax(port_return_df['drawdown'])
    j = np.argmax(port_return_df['NAV'][:i])

    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.plot(port_return_df['NAV'])
    plt.plot([i, j], 
             [port_return_df['NAV'][i], port_return_df['NAV'][j]], 
             'o', color = 'Red', markersize = 8)

    ax2 = ax.twinx()
    ax2.bar(port_return_df.index, port_return_df['drawdown'], color = 'g')
    
#------------------------------------------------------------------------------
def draw_acf_pacf(ts, lags = 31):
    f = plt.figure(facecolor = 'white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags = lags, ax = ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags = lags, ax = ax2)
    plt.show()

#------------------------------------------------------------------------------
def plot_monthly_return_distribution(port_return_df, after_cost = False):
    '''
    This function is used for plotting the monthly distribution of the 
    portrolio return. The return after or before transaction cost could 
    be chosen.
    The portfolio return DataFrame must be the typical port_return_df, and 
    the variable after_cost is a bool that could be chosen among True or False.
    '''

    if after_cost:
        return_var = 'net_port_return'
    else:
        return_var = 'port_return'

    month_return = port_return_df[return_var].groupby(
        pd.Grouper(freq = 'M')).sum()

    month = month_return.index.month
    year = month_return.index.year

    month_return.index = pd.MultiIndex.from_tuples(list(zip(month, year)))
    month_return.unstack().plot(kind = 'bar', title = 'Average Monthly Return')

    plt.show()


###############################################################################
###############################################################################
# Statistical Tests
#------------------------------------------------------------------------------
'''
   Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''
def test_stationarity(ts):
    df_test = adfuller(ts)
    df_output = pd.Series(
        df_test[0:4], 
        index = ['Test Statistic', 'p-value', 
                 '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' %key] = value
    return df_output

#------------------------------------------------------------------------------
# Breush-Pagan test
def breush_pagan_test(model):
    name = ['Lagrange multiplier statistic', 'p-value', 
            'f-value', 'f p-value']
    test = sms.het_breushpagan(model.resid, model.model.exog)
    return lzip(name, test)


###############################################################################
###############################################################################
#------------------------------------------------------------------------------
# Factors Orthogonalization
def GS_orthogonalize(model_df, ind_vars):
    '''
    This function use the Gram-Schmidt algorithm to commit the factor 
    orthogonalization procedure.
    It should be noted that the independent variables (ind_vars) should be 
    ordered outside of this function, otherwise the orthogonalizing order 
    would be random.
    '''
    orthogonalized_ind_vars = pd.DataFrame()
    orthogonalized_ind_vars[ind_vars[0]] = model_df[ind_vars[0]]

    for i in range(1, len(ind_vars)):
        temp_dep_var = ind_vars[i]
        temp_ind_var = ind_vars[0:i]
    
        temp_dep_var_beta_dict = {}
    
        temp_model_df = pd.DataFrame()
        temp_model_df[temp_dep_var] = model_df[temp_dep_var]
    
        orthogonalized_temp_dep_var = model_df[temp_dep_var]
    
        for orth_column in temp_ind_var:
            temp_model_df[orth_column] = orthogonalized_ind_vars[orth_column]
            temp_model_df = temp_model_df.dropna()
        
            temp_y = temp_model_df[temp_dep_var]
            temp_x = temp_model_df[orth_column]
        
            temp_model = regression.linear_model.OLS(temp_y, temp_x).fit()
        
            orthogonalized_temp_dep_var = (
                orthogonalized_temp_dep_var 
                - orthogonalized_ind_vars[orth_column].multiply(
                    temp_model.params[orth_column]))
        
        orthogonalized_ind_vars[temp_dep_var] = orthogonalized_temp_dep_var

    return orthogonalized_ind_vars

#------------------------------------------------------------------------------
def GS_orthogonalize_v2(model_df, ind_vars):
    '''
    This function use the Gram-Schmidt algorithm to commit the factor 
    orthogonalization procedure.
    It should be noted that the independent variables (ind_vars) should be 
    ordered outside of this function, otherwise the orthogonalizing order 
    would be random.
    '''
    orthogonalized_ind_vars = pd.DataFrame()
    orthogonalized_ind_vars[ind_vars[0]] = model_df[ind_vars[0]]

    for i in range(1, len(ind_vars)):
        temp_dep_var = ind_vars[i]
        temp_ind_var = ind_vars[0:i]
    
        temp_dep_var_beta_dict = {}
    
        temp_model_df = pd.DataFrame()
        temp_model_df[temp_dep_var] = model_df[temp_dep_var]
    
        orthogonalized_temp_dep_var = model_df[temp_dep_var]
    
        for orth_column in temp_ind_var:
            temp_model_df[orth_column] = orthogonalized_ind_vars[orth_column]
            temp_model_df = temp_model_df.dropna()
        
            y = np.array(temp_model_df.loc[:, temp_dep_var])
            x = np.array(temp_model_df.loc[:, orth_column])
            x = x.reshape((len(x), 1))

            beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

            orthogonalized_temp_dep_var = (
                orthogonalized_temp_dep_var 
                - x.dot(beta))
#                - orthogonalized_ind_vars[orth_column].dot(beta))
        
        orthogonalized_ind_vars[temp_dep_var] = (
            orthogonalized_temp_dep_var.reindex(
                orthogonalized_temp_dep_var.index))

    return orthogonalized_ind_vars


###############################################################################
###############################################################################
# Regression related
#------------------------------------------------------------------------------
def hedge_ratio(y, x, add_const = True):
    if add_const:
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        return model.params[1]
    model = sm.OLS(y, x).fit()
    return model.params.values

#------------------------------------------------------------------------------
def OLSReg(X1,Y):
    # Run the linear regression with only one independent variable.
    X = sm.add_constant(X1)
    model = regression.linear_model.OLS(Y, X).fit()
    alpha, beta1 = model.params
    return alpha, beta1


###############################################################################
###############################################################################
# Estimation of Beta

#------------------------------------------------------------------------------
def beta_daily_rolling_estimation_with_lagged_inde_var(
    close_return_df, macro_features_df, estimate_size, minimum_estimate_size):

    '''
    This Beta estimation method is from the [2013-DM]momentum_crashes[SSRN].
    It uses the contemporaneous and lagged market returns as the 
    independent variable, and fit a regression without constant. 
    Then the Betas of every single independent variable are added together as 
    the market Beta as the asset.
    '''

    N = close_return_df.shape[0]
    date_list = [
        [close_return_df.index[i-estimate_size+1], close_return_df.index[i]] 
        for i in range(estimate_size-1, N)]

    ind_var_num = len(macro_features_df.columns)
    ind_vars = macro_features_df.columns

    beta_df = pd.DataFrame(index = close_return_df.index[estimate_size:])

    for dep_var in close_return_df.columns:
        estimate_df = close_return_df[[dep_var]]
        estimate_df = pd.concat([estimate_df, macro_features_df], axis = 1)
        print estimate_df.tail()

        for date_pair in date_list:
            start_date = date_pair[0]
            end_date = date_pair[1]

            data_df = estimate_df.loc[start_date:end_date].dropna()
            if data_df.shape[0] > minimum_estimate_size:
                x = np.array(data_df.loc[:, [ind_vars]])
                x = x.reshape((len(x), ind_var_num))
                y = np.array(data_df.loc[:, dep_var])

                beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                beta_df.loc[end_date, dep_var] = np.sum(beta)
            else:
                beta_df.loc[end_date, dep_var] = np.nan

    return beta_df

#------------------------------------------------------------------------------
# Daily Rolling Estimation of Beta
def beta_daily_rolling_estimation_v2(
    close_return_df, macro_features_df, estimate_size, minimum_estimate_size):

    N = close_return_df.shape[0]
    date_list = [
        [close_return_df.index[i-estimate_size+1], close_return_df.index[i]] 
        for i in range(estimate_size-1, N)]

    if len(macro_features_df.columns) == 1:
        ind_var = macro_features_df.columns[0]
        print 'Start estimating the original Beta value of {0}...'.format(
            ind_var)

        beta_df = pd.DataFrame(index = close_return_df.index[estimate_size:])

        for dep_var in close_return_df.columns:
            estimate_df = close_return_df[[dep_var]]
            estimate_df[ind_var] = macro_features_df[ind_var].copy()
            estimate_df['const'] = 1

            for date_pair in date_list:
                start_date = date_pair[0]
                end_date = date_pair[1]

                data_df = estimate_df.loc[start_date:end_date].dropna()
                if data_df.shape[0] > minimum_estimate_size:
                    x = np.array(data_df.loc[:, [ind_var, 'const']])
                    x = x.reshape((len(x), 2))
                    y = np.array(data_df.loc[:, dep_var])

                    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                    beta_df.loc[end_date, dep_var] = beta[0]
                else:
                    beta_df.loc[end_date, dep_var] = np.nan
    else:
        print 'Only one variable could be estimated at one time!'

    print 'Finish estimating original Beta value of {0}!'.format(ind_var)

    return beta_df

#------------------------------------------------------------------------------
# Daily Rolling Estimation of Beta
def beta_daily_rolling_estimation(
    close_return_df, macro_features_df, liquid_contract_df, 
    total_range, test_range, 
    estimate_size, minimum_estimate_size): 

    original_macro_feature_beta_df = pd.DataFrame()
    standardized_macro_feature_beta_df = pd.DataFrame()
    demeaned_macro_feature_beta_df = pd.DataFrame()
    t_value_macro_feature_beta_df = pd.DataFrame()

    if len(macro_features_df.columns) == 1:
        
        ind_var = macro_features_df.columns[0]
        print 'Start estimating the Beta value of {0}'.format(ind_var)

        beta_df = pd.DataFrame()
        beta_t_value_df = pd.DataFrame()

        for test_date in test_range:
            training_end_index = total_range.index(test_date)
            training_start_index = max(
                0, training_end_index - estimate_size - 1)

            start_date = total_range[training_start_index]
            end_date = total_range[training_end_index]

            beta_dict = {}
            beta_t_value_dict = {}
            for dep_var in close_return_df.columns:
                estimate_df = close_return_df.loc[
                    start_date:end_date, [dep_var]]
                estimate_df[ind_var] = macro_features_df.loc[
                    start_date:end_date, ind_var]
                model_df = estimate_df.dropna()

                if model_df.shape[0] > minimum_estimate_size:
                    X1 = model_df[ind_var]
                    X = sm.add_constant(X1)
                    Y = model_df[dep_var]

                    model = regression.linear_model.OLS(Y, X).fit()

                    beta_dict[dep_var] = model.params[ind_var]
                    beta_t_value_dict[dep_var] = model.tvalues[ind_var]
                else:
                    beta_dict[dep_var] = np.nan
                    beta_t_value_dict[dep_var] = np.nan

            beta_dict['datetime'] = test_date
            temp_beta_df = pd.DataFrame.from_dict(
                beta_dict, orient = 'index').T
            temp_beta_df = temp_beta_df.set_index('datetime')

            beta_t_value_dict['datetime'] = test_date
            temp_beta_t_value_df = pd.DataFrame.from_dict(
                beta_t_value_dict, orient = 'index').T
            temp_beta_t_value_df = temp_beta_t_value_df.set_index('datetime')

            beta_df = pd.concat([beta_df, temp_beta_df])
            beta_t_value_df = pd.concat([beta_t_value_df, temp_beta_t_value_df])

#        beta_liquid_df = get_liquid_contract_data(beta_df, liquid_contract_df)
#        standardize_beta_df = panel_data_standardization(beta_liquid_df)
#        demean_beta_df = panel_data_demean(beta_liquid_df)
#        beta_t_value_liquid_df = get_liquid_contract_data(
#            beta_t_value_df, liquid_contract_df)
        
#        original_macro_feature_beta_df = beta_liquid_df
#        standardized_macro_feature_beta_df = standardize_beta_df
#        demeaned_macro_feature_beta_df = demean_beta_df
#        t_value_macro_feature_beta_df = beta_t_value_liquid_df
    else:
        print 'Only one variable could be estimated at one time!'

    result_dict = {'original_beta_df': beta_df}
    
    print 'Finish estimating Beta value of {0}!'.format(ind_var)
    
    return result_dict

#------------------------------------------------------------------------------
# Daily Rolling Estimation of Beta with resampling returns.
def beta_daily_rolling_estimation_with_resampling(
    close_return_df, macro_features_df, liquid_contract_df, 
    total_range, test_range, 
    estimate_size, minimum_estimate_size, resample_length): 

    '''
    This function resamples the daily return to weekly or monthly return, and 
    then use the resampled returns to estimate the Beta.
    The reason for doing so is to eliminate the huge noise in daily data and 
    try to get a more robust estimation of the Beta.
    The optional values of the parameter resample_length is the same as the 
    ones in the pd.resample function.
    '''

#    beta_df = pd.DataFrame()
    total_beta_dict = {}

    if len(macro_features_df.columns) == 1:
        
        ind_var = macro_features_df.columns[0]
        print 'Start estimating the Beta value of {0}'.format(ind_var)

        for test_date in test_range:
            training_end_index = total_range.index(test_date)
            training_start_index = max(
                0, training_end_index - estimate_size - 1)

            start_date = total_range[training_start_index]
            end_date = total_range[training_end_index]

            beta_dict = {}
            for dep_var in close_return_df.columns:
                estimate_df = close_return_df.loc[
                    start_date:end_date, [dep_var]]
                estimate_df[ind_var] = macro_features_df.loc[
                    start_date:end_date, ind_var]
                model_df = estimate_df.dropna()

                if model_df.shape[0] > minimum_estimate_size:

                    # This is the core difference: resampling the daily data 
                    # into weekly or monthly data.
#                    resample_model_df = model_df.resample(
#                        resample_length, label = 'right', closed = 'right').sum()

                    # A new version of self-defined resampling method.
                    cum_daily_return_df = model_df.rolling(window = resample_length).sum()
                    index_range = sorted(range(model_df.shape[0]-1, 0, -resample_length))
                    date_range = model_df.index[index_range]
                    resample_model_df = cum_daily_return_df.loc[date_range, :].dropna()

#                    X1 = model_df[ind_var]
#                    X = sm.add_constant(X1)
                    X = sm.add_constant(resample_model_df[ind_var])
                    Y = resample_model_df[dep_var]
                    model = regression.linear_model.OLS(Y, X).fit()

                    beta_dict[dep_var] = model.params[ind_var]
                else:
                    beta_dict[dep_var] = np.nan

#            beta_dict['datetime'] = test_date
#            temp_beta_df = pd.DataFrame.from_dict(
#                beta_dict, orient = 'index').T
#            temp_beta_df = temp_beta_df.set_index('datetime')

#            beta_df = pd.concat([beta_df, temp_beta_df])
            total_beta_dict[test_date] = beta_dict
    else:
        print 'Only one variable could be estimated at one time!'
        
#    result_dict = {'original_beta_df': beta_df}
    result_dict = {'original_beta': total_beta_dict}
    
    print 'Finish estimating Beta value of {0}!'.format(ind_var)
    
    return result_dict


#------------------------------------------------------------------------------
def beta_with_market_return_eliminated_daily_rolling_estimation(
    close_return_df, market_return_df, macro_features_df, liquid_contract_df,
    market_return_var, total_range, test_range, 
    estimate_size, minimum_estimate_size):

    original_macro_feature_beta_df = pd.DataFrame()
    standardized_macro_feature_beta_df = pd.DataFrame()
    demeaned_macro_feature_beta_df = pd.DataFrame()
    t_value_macro_feature_beta_df = pd.DataFrame()

    if len(macro_features_df.columns) == 1:
        
        ind_var = macro_features_df.columns[0]
        print 'Start estimating the Beta value of {0}'.format(ind_var)

        beta_df = pd.DataFrame()
        beta_t_value_df = pd.DataFrame()

        for test_date in test_range:
            training_end_index = total_range.index(test_date)
            training_start_index = max(
                0, training_end_index - estimate_size - 1)

            start_date = total_range[training_start_index]
            end_date = total_range[training_end_index]

            beta_dict = {}
            beta_t_value_dict = {}
            for dep_var in close_return_df.columns:
                demarket_df = close_return_df.loc[
                    start_date:end_date, [dep_var]]
                demarket_df[market_return_var] = market_return_df[
                    market_return_var]
                demarket_model_df = demarket_df.dropna()
                
                if demarket_model_df.shape[0] > minimum_estimate_size:
                    demarket_X = sm.add_constant(
                        demarket_model_df[market_return_var])
                    demarket_Y = demarket_model_df[dep_var]
                    demarket_model = regression.linear_model.OLS(
                        demarket_Y, demarket_X).fit()
                    
                    estimate_df = pd.DataFrame(
                        demarket_model.resid, columns = [dep_var])
                    estimate_df[ind_var] = macro_features_df.loc[
                        start_date:end_date, ind_var]
                    model_df = estimate_df.dropna()
                    
                    if model_df.shape[0] > minimum_estimate_size:
                        X = sm.add_constant(model_df[ind_var])
                        Y = model_df[dep_var]
                        model = regression.linear_model.OLS(Y, X).fit()
                    
                        beta_dict[dep_var] = model.params[ind_var]
                        beta_t_value_dict[dep_var] = model.tvalues[ind_var]
                    else:
                        beta_dict[dep_var] = np.nan
                        beta_t_value_dict[dep_var] = np.nan
                else:
                    beta_dict[dep_var] = np.nan
                    beta_t_value_dict[dep_var] = np.nan                    
                           
            beta_dict['datetime'] = test_date
            temp_beta_df = pd.DataFrame.from_dict(
                beta_dict, orient = 'index').T
            temp_beta_df = temp_beta_df.set_index('datetime')

            beta_t_value_dict['datetime'] = test_date
            temp_beta_t_value_df = pd.DataFrame.from_dict(
                beta_t_value_dict, orient = 'index').T
            temp_beta_t_value_df = temp_beta_t_value_df.set_index('datetime')

            beta_df = pd.concat([beta_df, temp_beta_df])
            beta_t_value_df = pd.concat([beta_t_value_df, temp_beta_t_value_df])

#        beta_liquid_df = get_liquid_contract_data(beta_df, liquid_contract_df)
#        standardize_beta_df = panel_data_standardization(beta_liquid_df)
#        demean_beta_df = panel_data_demean(beta_liquid_df)
#        beta_t_value_liquid_df = get_liquid_contract_data(
#            beta_t_value_df, liquid_contract_df)
        
#        original_macro_feature_beta_df = beta_liquid_df
#        standardized_macro_feature_beta_df = standardize_beta_df
#        demeaned_macro_feature_beta_df = demean_beta_df
#        t_value_macro_feature_beta_df = beta_t_value_liquid_df
    else:
        print 'Only one variable could be estimated at one time!'
        
#    result_dict = {'original_beta_df': original_macro_feature_beta_df, 
#                   'standardized_beta_df': standardized_macro_feature_beta_df, 
#                   'demeaned_beta_df': demeaned_macro_feature_beta_df, 
#                   't_value_df': t_value_macro_feature_beta_df}
    result_dict = {'original_beta_df': beta_df}
    
    print 'Finish estimating Beta value of {0}!'.format(ind_var)
    
    return result_dict
