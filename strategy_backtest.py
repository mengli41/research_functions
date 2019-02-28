# encoding: UTF-8

import numpy as np
import pandas as pd
import datetime as dt

###############################################################################
###############################################################################
class StrategyBakctester:
    #--------------------------------------------------------------------------
    def __init__(self, trans_cost, intraday_return_df, interday_return_df):
        self.trans_cost = trans_cost
        self.intraday_return_df = intraday_return_df
        self.interday_return_df = interday_return_df
        self.pos_df = None
        self.prediction_df = None
        self.quantile_feature_dict = {}
        self.quantile_port_return_dict = {}
        self.quantile_sharpe_dict = {}

    #--------------------------------------------------------------------------
    def __calculate_sharpe_ratio(self, return_series, risk_free_rate):
        sharpe_ratio = ((return_series.mean() - risk_free_rate / 242.0) 
                        / np.std(return_series) * np.sqrt(242.0))
        return sharpe_ratio

    #--------------------------------------------------------------------------
    def input_prediction(self, prediction_df):
        self.prediction_df = prediction_df

    #--------------------------------------------------------------------------
    def input_position(self, pos_df):
        self.pos_df = pos_df

    #--------------------------------------------------------------------------
    def rolling_prediction(self, rolling_window, ewm_rolling = True):
        if type(self.prediction_df) == pd.core.frame.DataFrame:
            if ewm_rolling:
                self.prediction_df = self.prediction_df.ewm(
                    span = rolling_window).mean()
            else:
                self.prediction_df = self.prediction_df.rolling(
                    window = rolling_window).mean()
        else:
            print 'Please input the prediction DataFrame first!'

    #--------------------------------------------------------------------------
    def rolling_pos(self, rolling_window, ewm_rolling = False):
        if type(self.pos_df) == pd.core.frame.DataFrame:
            if ewm_rolling:
                self.pos_df = self.pos_df.ewm(span = rolling_window).mean()
            else:
                self.pos_df = self.pos_df.rolling(
                    window = rolling_window).mean()
        else:
            print 'Please input the position DataFrame first!'

    #--------------------------------------------------------------------------
    def generate_sign_pos(self, reverse_ind = 1): 
        if type(self.prediction_df) == pd.core.frame.DataFrame:
            if type(self.pos_df) == pd.core.frame.DataFrame:
                self.pos_df = None

            signal_df = self.prediction_df.copy()
            signal_df[self.prediction_df < 0] = -1 * reverse_ind
            signal_df[self.prediction_df > 0] = 1 * reverse_ind

            total_pos_df = signal_df.copy()
            self.pos_df = (
                total_pos_df.div(total_pos_df.abs().sum(axis = 1), axis = 0))
            self.pos_df = self.pos_df.fillna(0)
        else:
            print 'Please input the prediction DataFrame first!'

    #--------------------------------------------------------------------------
    def generate_normal_pos(self, compare_er_to_tc = True):
        if type(self.prediction_df) == pd.core.frame.DataFrame: 
            if type(self.pos_df) == pd.core.frame.DataFrame:
                self.pos_df = None

            total_pos_df = self.prediction_df.dropna(axis = 0, how = 'all')
            pos_df = total_pos_df.div(
                total_pos_df.abs().sum(axis = 1), axis = 0)
            pos_df = pos_df.fillna(0)

            trans_cost = [self.trans_cost] * pos_df.shape[1]
            trans_cost_df = np.abs(
                pos_df.shift() - pos_df.shift(2)).multiply(trans_cost)
            trans_cost_df['trans_cost'] = trans_cost_df.sum(axis = 1)

            if compare_er_to_tc:
                #--------------------------------------------------------------
                # Check if the expected return based on the current 
                # prediction could cover the transaction costs. 
                # If it can't, then keep the current position, 
                # otherwise, modify the position accordingly.
                expected_return = pos_df.multiply(self.prediction_df.dropna(
                    axis = 0, how = 'all'))
                expected_return['total_er'] = expected_return.sum(axis = 1)

                er_tc_gap = pd.DataFrame(
                    (expected_return['total_er'] 
                     - trans_cost_df['trans_cost']), 
                    columns = ['er_tc_gap'])

                self.pos_df = pos_df.copy()
                self.pos_df[er_tc_gap['er_tc_gap'] <= 0] = np.nan
                self.pos_df = self.pos_df.fillna(method = 'ffill')
            else:
                self.pos_df = pos_df.copy()
        else:
            print 'Please input the prediction DataFrame first!'

    #--------------------------------------------------------------------------
    def generate_high_low_quantile_pos(self, quantile, reverse_ind = 1): 
        if type(self.prediction_df) == pd.core.frame.DataFrame: 
            if type(self.pos_df) == pd.core.frame.DataFrame:
                self.pos_df = None

            test_feature_df = self.prediction_df.dropna(axis = 0, how = 'all')

            #------------------------------------------------------------------
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
            #------------------------------------------------------------------

            self.pos_df = (
                total_pos_df.div(total_pos_df.abs().sum(axis = 1), axis = 0))
            self.pos_df = self.pos_df.fillna(0)
        else:
            print 'Please input the prediction DataFrame first!'

    #--------------------------------------------------------------------------
    def generate_quantile_dict(self, quantile_num = 5, 
                               feature_keyword = 'feature', 
                               include_market = False):
        if type(self.prediction_df) != pd.core.frame.DataFrame:
            print 'Please input the prediction DataFrame first!'
        else:
            if len(self.quantile_feature_dict) > 0:
                self.quantile_feature_dict = {}

            test_feature_df = self.prediction_df.dropna(axis = 0, how = 'all')
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

            if include_market:
                # This is to calculate the equanlly weighted market mean return.
                market_df = 1 * use_feature_rank.notnull()
                self.quantile_feature_dict['market'] = market_df

    #--------------------------------------------------------------------------
    def generate_quantile_dict_v2(self, quantile_num = 5, 
                                  feature_keyword = 'feature', 
                                  include_market = False):
        if type(self.prediction_df) != pd.core.frame.DataFrame:
            print 'Please input the prediction DataFrame first!'
        else:
            if len(self.quantile_feature_dict) > 0:
                self.quantile_feature_dict = {}

            test_feature_df = self.prediction_df.dropna(axis = 0, how = 'all')
            rank_df = test_feature_df.rank(axis = 1)
            self.duplicate_date_list = []

            q_labels = ['{0}_q_{1}'.format(feature_keyword, str(i)) 
                        for i in range(quantile_num)]
            self.categorical_feature_df = np.nan * rank_df

            for rank_date in rank_df.index:
                try:
                    tmp_df = pd.qcut(rank_df.loc[rank_date], 
                                     quantile_num, q_labels)
                    self.categorical_feature_df.loc[rank_date] = tmp_df
                except:
                    self.duplicate_date_list.append(rank_date)

            if len(self.duplicate_date_list) > 0:
                for dup_date in self.duplicate_date_list:
                    max_rank = rank_df.loc[dup_date].max()
                    min_rank = rank_df.loc[dup_date].min()
                    step = max_rank / float(quantile_num)

                    q_range = [step*i for i in range(1, quantile_num)]
                    q_range = sorted(q_range + [min_rank, max_rank])

                    dup_df = rank_df.loc[dup_date]
                    for i in range(len(q_labels)):
                        self.categorical_feature_df.loc[
                            dup_date, 
                            ((dup_df > q_range[i]) 
                             & (dup_df <= q_range[i+1]))] = q_labels[i]

            for q_label in q_labels:
                q_dummy_df = 1 * (self.categorical_feature_df == q_label)
                self.quantile_feature_dict[q_label] = q_dummy_df

            if include_market:
                market_df = 1 * rank_df.notnull()
                self.quantile_feature_dict['market'] = market_df

    #--------------------------------------------------------------------------
    def generate_selected_quantile_pos(self, quantile_name_1, quantile_name_2):
        if len(self.quantile_feature_dict) > 0:
            if type(self.pos_df) == pd.core.frame.DataFrame:
                self.pos_df = None

            quantile_1_df = self.quantile_feature_dict[quantile_name_1]
            quantile_2_df = self.quantile_feature_dict[quantile_name_2]

            quantile_1_cs_sum_df = quantile_1_df.abs().sum(axis = 1)
            quantile_1_cs_sum_df.loc[quantile_1_cs_sum_df == 0.0] = 1.0

            quantile_2_cs_sum_df = quantile_2_df.abs().sum(axis = 1)
            quantile_2_cs_sum_df.loc[quantile_2_cs_sum_df == 0.0] = 1.0

            self.pos_df = 0.5 * (
                quantile_1_df.div(quantile_1_cs_sum_df, axis = 0) 
                - quantile_2_df.div(quantile_2_cs_sum_df, axis = 0))
        else:
            print 'Please generate the quantile dict first!'

    #--------------------------------------------------------------------------
    def factor_backtest(self, risk_free_rate = 0.0):
        if len(self.quantile_feature_dict) > 0:
            for key,value in self.quantile_feature_dict.items():
                value_sum_df = value.abs().sum(axis = 1)
                value_sum_df[value_sum_df == 0.0] = 1.0
                pos_df = value.div(value_sum_df, axis = 0)
                port_return_df = (
                    pos_df.shift().multiply(
                        self.intraday_return_df.loc[pos_df.index])
                    + pos_df.shift(2).multiply(
                        self.interday_return_df.loc[pos_df.index]))
                port_return_df['port_return'] = port_return_df.sum(axis = 1)

                self.quantile_port_return_dict[key] = port_return_df
                self.quantile_sharpe_dict[key] = self.__calculate_sharpe_ratio(
                    port_return_df['port_return'], risk_free_rate)
        else:
            print 'Please generate the quantile dict first!'

    #--------------------------------------------------------------------------
    def plain_backtest(self):
        if type(self.pos_df) != pd.core.frame.DataFrame:
            print 'Please generate or input the position DataFrame first!'
        else:
            trans_cost = [self.trans_cost] * self.pos_df.shape[1]
            trans_cost_df = np.abs(
                self.pos_df.shift() 
                - self.pos_df.shift(2)).multiply(trans_cost)
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

