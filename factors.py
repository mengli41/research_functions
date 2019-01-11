import pandas as pd
import numpy as np
import scipy as sp
import bottleneck as bn
import itertools as it


###############################################################################
###############################################################################
class IndustryClassification:

    #--------------------------------------------------------------------------
    def __init__(self, industry_class_indicator, if_financial_futures, 
                 if_get_inverse_classes):
        self.industry_class_indicator = industry_class_indicator
        self.if_financial_futures = if_financial_futures
        self.if_get_inverse_classes = if_get_inverse_classes

        self.industry_class_1 = {
            'PreciousMetal': ['au', 'ag'],
            'IndustrialMetal': ['cu', 'zn', 'al', 'ni', 'pb', 'sn'],
            'BlackChain': ['rb', 'j', 'i', 'jm', 'hc', 'SM', 'SF', 'FG', 'ZC'],
            'EnergyChem': ['l', 'MA', 'pp', 'TA', 'ru', 'bu', 'v', 'sc'],
            'Agriculture': ['cs', 'c', 'a', 'm', 'RM', 'y', 'p', 'OI', 'b'],
            'SoftComm': ['CF', 'SR', 'jd', 'AP']}

        self.industry_class_2 = {
            'Metal': ['cu', 'zn', 'al', 'ni', 'pb', 'sn', 'au', 'ag'], 
            'BlackChain': ['rb', 'j', 'i', 'jm', 'hc', 'SM', 'SF'], 
            'EnergyChem': ['l', 'MA', 'pp', 'TA', 'ru', 'bu', 'v', 
                           'sc', 'ZC', 'FG'], 
            'Agri': ['CF', 'SR', 'a', 'm', 'RM', 'y', 'p', 'OI', 'cs', 'c'], 
            'SoftComm': ['jd', 'AP']}

        self.industry_class_3 = {
            'Indust': ['cu', 'zn', 'al', 'ni', 'pb', 'sn', 'au', 'ag', 'rb', 
                       'j', 'i', 'jm', 'hc', 'SM', 'SF', 'FG', 'l', 'MA', 
                       'pp', 'TA', 'ru', 'bu', 'v', 'sc', 'ZC'], 
            'Agri': ['a', 'm', 'RM', 'y', 'p', 'OI', 'cs', 'c', 'CF', 
                     'SR', 'jd', 'AP']}

        self.financial_futures = {
            'Index': ['IF', 'IH', 'IC'], 
            'Rate': ['TF', 'T', 'TS']}

        self.reverse_industry_class = {}

    #--------------------------------------------------------------------------
    def get_industry_classes(self):
        if self.industry_class_indicator == 'industry_class_1':
            final_industry_class = self.industry_class_1
        elif self.industry_class_indicator == 'industry_class_2':
            final_industry_class = self.industry_class_2
        elif self.industry_class_indicator == 'industry_class_3':
            final_industry_class = self.industry_class_3
        else:
            final_industry_class = {}

        if self.if_financial_futures == True:
            final_industry_class.update(self.financial_futures)

        self.final_industry_class = final_industry_class

        return final_industry_class

    #--------------------------------------------------------------------------
    def get_inverse_industry_classes(self):
        if self.if_get_inverse_classes:
            if len(self.final_industry_class) > 0:
                list0 = sum(
                    [list(it.product([x], self.final_industry_class[x])) 
                     for x in self.final_industry_class.keys()], [])
                self.reverse_industry_class = dict(
                    [[a[1], a[0]] for a in list0])

        return self.reverse_industry_class


###############################################################################
###############################################################################
class PVFactors:

    #--------------------------------------------------------------------------
    def __init__(self, data):
        self.open_price = data['open']
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.pre_close = data['close'].shift()
        self.vwap = data['vwap']
        self.volume = data['volume']
        self.amount = data['amount']

    #--------------------------------------------------------------------------
    def time_series_rank(self, x):
        return bn.rankdata(x)[-1]

    #--------------------------------------------------------------------------
    def get_liquid_contract_data(self, data_df, liquid_contract_df):
        liquid_data_df = data_df.copy()

        for column in liquid_contract_df.columns: 
            if column in liquid_data_df.columns:
                liquid_data_df.loc[:, column] = np.where(
                    liquid_contract_df.loc[liquid_data_df.index, column] == 0, 
                    np.nan, liquid_data_df.loc[:, column])

        return liquid_data_df

    #--------------------------------------------------------------------------
    def rsi(self, price, n = 14):
        delta = price.diff()

        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = dUp.rolling(window = n).mean()
        RolDown = dDown.rolling(window = n).mean().abs()

        rs = RolUp / RolDown

        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    #--------------------------------------------------------------------------
    def new_rsi(self, lookback = 1, windows = 20):
        ''' an altered rsi indicator '''
#        gain = self.close.pct_change()
        gain = np.log(self.close).diff()
        avgGain = gain[gain>0].fillna(0).rolling(lookback).sum() 
        avgLoss = -gain[gain<0].fillna(0).rolling(lookback).sum()
        alter_rsi = (0.5 * (avgGain - avgLoss) / (avgGain + avgLoss)).rolling(
            windows).mean()

#        return SMA(0.5*(avgGain-avgLoss) /(avgGain+avgLoss), windows )
        return alter_rsi

    #--------------------------------------------------------------------------
    def ma_close_ratio(self, window = 20):
        ma_close = self.close.rolling(window).mean()
        ma_close_ratio = ma_close.div(self.close)

        return ma_close_ratio

    #--------------------------------------------------------------------------
#    def alpha_001(self, rolling_window):
#        data1 = self.volume.diff(periods = 1).rank(axis = 1, pct = True)
#        data2 = ((self.close - self.open_price) / self.open_price).rank(
#            axis = 1, pct = True)
#        alpha = -data1.rolling(window = rolling_window).corr(
#            data2, pairwise = False)
#        alpha = alpha.dropna(how = 'all')
#
#        return alpha
    def alpha_001(self, rolling_window, liquid_contract_df):
        tmp_1 = self.volume.diff()
        tmp_1_liquid = self.get_liquid_contract_data(tmp_1, liquid_contract_df)
        data1 = tmp_1_liquid.rank(axis = 1, pct = True)

        tmp_2 = (self.close - self.open_price) / self.open_price
        tmp_2_liquid = self.get_liquid_contract_data(tmp_2, liquid_contract_df)
        data2 = tmp_2_liquid.rank(axis = 1, pct = True)

        alpha = -data1.rolling(window = rolling_window).corr(
            data2, pairwise = False).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_002(self, delay_window = 1):
        result = (((self.close - self.low) - (self.high - self.close)) 
                  / ((self.high - self.low))).diff(delay_window)
        m = result.dropna(how = 'all')
        alpha = m[(m < np.inf) & (m > -np.inf)]

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_003(self, delay_window = 1, rolling_window = 6):
        delay1 = self.close.shift(delay_window)
        condition1 = (self.close == delay1)
        condition2 = (self.close > delay1)
        condition3 = (self.close < delay1)
    
#        part2 = (self.close - np.minimum(delay1[condition2], 
#                                         self.low[condition2]))
#        part3 = (self.close - np.maximum(delay1[condition3], 
#                                         self.low[condition3]))
        part2 = (np.log(self.close) 
                 - np.log(np.minimum(delay1[condition2], 
                                     self.low[condition2])))
        part3 = (np.log(self.close) 
                 - np.log(np.maximum(delay1[condition3], 
                                     self.low[condition3])))

        result = part2.fillna(0) + part3.fillna(0)
        alpha = result.rolling(window = rolling_window).sum()

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_004(self, short_window = 2, long_window = 8, volume_window = 20):
        condition1 = (
            (self.close.rolling(window = long_window).mean()
             + self.close.rolling(window = long_window).std()) 
            < (self.close.rolling(window = short_window).mean()))
        condition2 = (
            (self.close.rolling(window = short_window).mean()) 
            < (self.close.rolling(window = long_window).mean()
               - self.close.rolling(window = long_window).std()))
        condition3 = (
            1 <= (self.volume 
                  / self.volume.rolling(window = volume_window).mean()))

        indicator1 = pd.DataFrame(
            np.ones(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns)
        indicator2 = -pd.DataFrame(
            np.ones(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns)

        part1 = indicator2[condition1].fillna(0)
        part2 = (indicator1[~condition1][condition2]).fillna(0)
        part3 = (indicator1[~condition1][~condition2][condition3]).fillna(0)
        part4 = (indicator2[~condition1][~condition2][~condition3]).fillna(0)

        result = part1 + part2 + part3 + part4
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_005(self, rank_window = 5, corr_window = 3):
#        pctrank = lambda x: bn.rankdata(x)[-1]

        ts_volume = self.volume.rolling(window = rank_window).apply(
            self.time_series_rank)
        ts_high = self.high.rolling(window = rank_window).apply(
            self.time_series_rank)
        
        corr_ts = ts_high.rolling(window = rank_window).corr(
            ts_volume, pairwise = False)
        
        alpha = corr_ts.rolling(window = corr_window).max()

        return alpha

    #--------------------------------------------------------------------------
    def alpha_006(self, liquid_contract_df, open_mult = 0.85, diff_window = 4):
        condition1 = ((self.open_price * open_mult 
                       + self.high * (1 - open_mult)).diff(diff_window) > 0)
        condition2 = ((self.open_price * open_mult 
                       + self.high * (1 - open_mult)).diff(diff_window) == 0)
        condition3 = ((self.open_price * open_mult 
                       + self.high * (1 - open_mult)).diff(diff_window) < 0)

        indicator1 = pd.DataFrame(
            np.ones(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns)
        indicator2 = pd.DataFrame(
            np.zeros(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns)
        indicator3 = -pd.DataFrame(
            np.ones(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns) 

        part1 = indicator1[condition1].fillna(0)
        part2 = indicator2[condition2].fillna(0)
        part3 = indicator3[condition3].fillna(0)

        result = part1 + part2 + part3
        result_liquid = self.get_liquid_contract_data(
            result, liquid_contract_df)
        alpha = result_liquid.rank(axis = 1, pct = True)

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_006_alter(self, liquid_contract_df, 
                        open_mult = 0.85, diff_window = 4):
#        result = (self.open_price*open_mult + self.high*(1 - open_mult)).diff(
#            diff_window)
        result = np.log(self.open_price * open_mult 
                        + self.high * (1 - open_mult)).diff(diff_window)
        result_liquid = self.get_liquid_contract_data(
            result, liquid_contract_df)
        alpha = result_liquid.rank(axis = 1, pct = True).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_007(self, liquid_contract_df, com_num_1 = 3, 
                  com_num_2 = 3, com_num_3 = 3):
        part1 = np.maximum(self.vwap - self.close, com_num_1)
        part1_liquid = self.get_liquid_contract_data(part1, liquid_contract_df)
        part1_rank = part1_liquid.rank(axis = 1, pct = True)

        part2 = np.minimum(self.vwap - self.close, com_num_2)
        part2_liquid = self.get_liquid_contract_data(part2, liquid_contract_df)
        part2_rank = part2_liquid.rank(axis = 1, pct = True)

        part3 = self.volume.diff(com_num_3)
        part3_liquid = self.get_liquid_contract_data(part3, liquid_contract_df)
        part3_rank = part3_liquid.rank(axis = 1, pct = True)

        alpha = (part1_rank + part2_rank * part3_rank).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_007_alter(self, liquid_contract_df, diff_window = 3):
        part1 = (np.log(self.vwap) - np.log(self.close)).rolling(
            window = diff_window).max()
        part1_liquid = self.get_liquid_contract_data(part1, liquid_contract_df)
        part1_rank = part1_liquid.rank(axis = 1, pct = True)

        part2 = (np.log(self.vwap) - np.log(self.close)).rolling(
            window = diff_window).min()
        part2_liquid = self.get_liquid_contract_data(part2, liquid_contract_df)
        part2_rank = part2_liquid.rank(axis = 1, pct = True)

        part3 = np.log(self.volume).diff(diff_window)
        part3_liquid = self.get_liquid_contract_data(part3, liquid_contract_df)
        part3_rank = part3_liquid.rank(axis = 1, pct = True)

        alpha = (part1_rank + part2_rank * part3_rank).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_008(self, liquid_contract_df, 
                  high_low_mult = 0.2, diff_window = 4):
        temp = -np.log((self.high + self.low) * 0.5 * high_low_mult 
                       + self.vwap * (1 - high_low_mult)).diff(diff_window)
        temp_liquid = self.get_liquid_contract_data(temp, liquid_contract_df)

        alpha = temp_liquid.rank(axis = 1, pct = True).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_009(self, alpha = 2.0 / 7.0):
        temp = (
            ((self.high + self.low) * 0.5 
             - (self.high.shift() + self.low.shift()) * 0.5) 
            * (self.high - self.low) / self.volume)
        result = temp.ewm(alpha = alpha).mean()
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_010(self, liquid_contract_df, std_window = 20, com_num = 5):
        ret = np.log(self.close).diff()
        condition = (ret < 0)

        part1 = (ret.rolling(window = std_window).std()[condition]).fillna(0)
        part2 = (self.close[~condition]).fillna(0)

        result = np.maximum((part1 + part2) ** 2, com_num)
        result_liquid = self.get_liquid_contract_data(
            result, liquid_contract_df)
        alpha = result_liquid.rank(axis = 1, pct = True).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_010_alter(self, liquid_contract_df, 
                        std_window = 20, com_num = 5):
        ret = np.log(self.close).diff()
        condition = (ret < 0)

        part1 = (ret.rolling(window = std_window).std()[condition]).fillna(0)
        part2 = (self.close[~condition]).fillna(0)

        result = ((part1 + part2) ** 2).rolling(window = com_num).max()
        result_liquid = self.get_liquid_contract_data(
            result, liquid_contract_df)
        alpha = result_liquid.rank(axis = 1, pct = True).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_011(self, rolling_window = 6):
        temp = (((self.close - self.low) - (self.high - self.close)) 
                / (self.high - self.low))
        result = temp * self.volume
        alpha = result.rolling(window = rolling_window).sum()

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_011_alter(self, rolling_window = 6):
        temp = (((self.close - self.low) - (self.high - self.close)) 
                / (self.high - self.low))
        alpha = temp.rolling(window = rolling_window).sum()

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_012(self, liquid_contract_df, vwap_window = 10):
        vwap_ma = self.vwap.rolling(window = vwap_window).mean()
        temp1 = self.open_price - vwap_ma
        temp1_liquid = self.get_liquid_contract_data(temp1, liquid_contract_df)
        part1 = temp1_liquid.rank(axis = 1, pct = True)

        temp2 = (self.close - self.vwap).abs()
        temp2_liquid = self.get_liquid_contract_data(temp2, liquid_contract_df)
        part2 = -temp2_liquid.rank(axis = 1, pct = True)

        alpha = (part1 * part2).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_013(self):
        result = ((self.high * self.low) ** 0.5) - self.vwap
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_013_alter(self):
        result = np.log((self.high * self.low) ** 0.5) - np.log(self.vwap)
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_014(self, shift_window = 5):
        result = self.close - self.close.shift(shift_window)
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_014_alter(self, shift_window = 5):
        result = np.log(self.close) - np.log(self.close.shift(shift_window))
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_015(self, close_shift = 1):
        result = self.open_price / self.close.shift(close_shift) - 1
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_016(self, liquid_contract_df, corr_window = 5, max_window = 5):
        volume_liquid = self.get_liquid_contract_data(
            self.volume, liquid_contract_df)
        vwap_liquid = self.get_liquid_contract_data(
            np.log(self.vwap).diff(), liquid_contract_df)

        temp1 = volume_liquid.rank(axis = 1, pct = True)
        temp2 = vwap_liquid.rank(axis = 1, pct = True) 

        part = temp1.rolling(window = corr_window).corr(
            temp2, pairwise = False)
        part = part[(part < np.inf) & (part > -np.inf)]

        result = part.rank(axis = 1, pct = True)
        alpha = result.rolling(window = max_window).max().dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_017(self, liquid_contract_df, 
                  ts_max_window = 15, close_diff_window = 5):
        temp1 = self.vwap.rolling(window = ts_max_window).max()
        temp2 = (self.vwap - temp1)#.dropna(how = 'all')
        temp2_liquid = self.get_liquid_contract_data(temp2, liquid_contract_df)

        part1 = temp2_liquid.rank(axis = 1, pct = True)
        part2 = np.log(self.close).diff(close_diff_window)

        alpha = (part1 ** part2).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_018(self, delay_window = 5):
        delay = self.close.shift(delay_window)
        alpha = self.close / delay

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_019(self, delay_window = 5):
        delay = self.close.shift(delay_window)
        
        condition1 = (self.close < delay)
        condition3 = (self.close > delay)
        
        part1 = ((self.close[condition1] - delay[condition1]) 
                 / delay[condition1])
        part1 = part1.fillna(0)
        
        part2 = ((self.close[condition3] - delay[condition3]) 
                 / self.close[condition3])
        part2 = part2.fillna(0)
        
        result = part1 + part2
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_020(self, delay_window = 6):
        delay = self.close.shift(delay_window)
        result = (self.close - delay) * 100 / delay
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_021(self, close_rolling_window = 6, minimum_estimate_size = 6):
        part1 = self.close.rolling(window = close_rolling_window).mean()
        part2 = np.arange(1, close_rolling_window + 1)

#        temp = part1.rolling(window = close_rolling_window).apply(
#            lambda x: sp.stats.linregress(x, part2))
#        beta_list = [temp[i].slope for i in range(len(temp))]
#
#        alpha = pd.Series(beta_list, index = temp.index).dropna(how = 'all')
        
        N = part1.shape[0]
        date_list = [
            [part1.index[i-close_rolling_window+1], part1.index[i]]
            for i in range(close_rolling_window-1, N)]

        beta_df = pd.DataFrame(index = part1.index[close_rolling_window:])

        for dep_var in part1.columns:
            estimate_df = part1[[dep_var]]
#            estimate_df['x'] = part2.copy()
            estimate_df['const'] = 1

            for date_pair in date_list:
                start_date = date_pair[0]
                end_date = date_pair[1]

                data_df = estimate_df.loc[start_date:end_date]#.dropna()
                data_df['x'] = part2.copy()
                data_df = data_df.dropna()
                if data_df.shape[0] > minimum_estimate_size:
                    x = np.array(data_df.loc[:, ['x', 'const']])
                    x = x.reshape((len(x), 2))
                    y = np.array(data_df.loc[:, dep_var])

                    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
                    beta_df.loc[end_date, dep_var] = beta[0]
                else:
                    beta_df.loc[end_date, dep_var] = np.nan

        alpha = beta_df

        return alpha

    #--------------------------------------------------------------------------
    def alpha_021_alter(self, close_rolling_window = 6):
        part1 = self.close.rolling(window = close_rolling_window).mean()
        part2 = np.arange(1, close_rolling_window + 1)

        temp = part1.rolling(window = close_rolling_window).apply(
            lambda x: np.corrcoef(x, part2)[0,1])

        alpha = temp.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_022(self, close_window = 6, shift_window = 3, 
                  alpha = 1.0 / 12.0):
        part1 = ((self.close 
                  - self.close.rolling(window = close_window).mean()) 
                 / self.close.rolling(window = close_window).mean())
        temp = ((self.close 
                 - self.close.rolling(window = close_window).mean()) 
                / self.close.rolling(window = close_window).mean())
        part2 = temp.shift(shift_window)
        result = part1 - part2
        result = result.ewm(alpha = alpha).mean()
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_023(self, shift_window = 1, rolling_window = 20, 
                  alpha = 1.0 / 20.0):
        condition1 = (self.close > self.close.shift(shift_window))

        temp1 = self.close.rolling(window = rolling_window).std()[condition1]
        temp1 = temp1.fillna(0)

        temp2 = self.close.rolling(window = rolling_window).std()[~condition1]
        temp2 = temp2.fillna(0)
        
        part1 = temp1.ewm(alpha = alpha).mean()
        part2 = temp2.ewm(alpha = alpha).mean()
        
        result = part1 * 100 / (part1 + part2)
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_024(self, shift_window = 5, alpha = 1.0 / 5.0):
        delay = self.close.shift(shift_window)
        result = self.close - delay
        result = result.ewm(alpha = alpha).mean()
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_024_alter(self, shift_window = 5, alpha = 1.0 / 5.0):
        delay = self.close.shift(shift_window)
        result = np.log(self.close) - np.log(delay)
        alpha = result.ewm(alpha = alpha).mean()

        return alpha.dropna(how = 'all')


    #--------------------------------------------------------------------------
    def alpha_026(self, close_window = 7, shift_window = 5, vwap_window = 230):
        part1 = self.close.rolling(window = close_window).mean() - self.close
        delay = self.close.shift(shift_window)
        part2 = self.vwap.rolling(window = vwap_window).corr(
            delay, pairwise = False)
        alpha = part1 + part2

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_028(self, rolling_window = 9, alpha = 1.0 / 3.0, 
                  part1_multi = 3, part2_multi = 2):
        temp1 = self.close - self.low.rolling(window = rolling_window).min()
        temp2 = (self.high.rolling(window = rolling_window).max() 
                 - self.low.rolling(window = rolling_window).min())
        part1 = part1_multi * (temp1 * 100 / temp2).ewm(alpha = alpha).mean()

        temp3 = (temp1 * 100 / temp2).ewm(alpha = alpha).mean()
        part2 = part2_multi * temp3.ewm(alpha = alpha).mean()

        result = part1 - part2
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_029(self, delay_window = 6):
        delay = self.close.shift(delay_window)
        result = (self.close - delay) * self.volume / delay
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_031(self, close_window = 12):
        result = ((self.close 
                   - self.close.rolling(window = close_window).mean()) 
                  / self.close.rolling(window = close_window).mean() * 100)
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_034(self, close_window = 12):
        result = self.close.rolling(window = close_window).mean() / self.close
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_038(self, rolling_window = 20, diff_window = 2):
        sum_20 = (self.high.rolling(window = rolling_window).sum() 
                  / rolling_window)
        delta2 = self.high.diff(diff_window)
        condition = (sum_20 < self.high)
        result = -delta2[condition].fillna(0)
        alpha = result

        return alpha

    #--------------------------------------------------------------------------
    def alpha_040(self, shift_window = 1, vol_window = 26):
        delay1 = self.close.shift(shift_window)
        condition = (self.close > delay1)
        
        vol = self.volume[condition].fillna(0)
        vol_sum = vol.rolling(window = vol_window).sum()
        
        vol1 = self.volume[~condition].fillna(0)
        vol1_sum = vol1.rolling(window = vol_window).sum()
        
        result = 100 * vol_sum / vol1_sum
        alpha = result
        alpha = alpha[(alpha < np.inf) & (alpha > -np.inf)]
        alpha = alpha.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_043(self, shift_window = 1, rolling_window = 6):
        delay1 = self.close.shift(shift_window)
        condition1 = (self.close > delay1)
        condition2 = (self.close < delay1)
        temp1 = self.volume[condition1].fillna(0)
        temp2 = -self.volume[condition2].fillna(0)
        result = temp1 + temp2
        result = result.rolling(window = rolling_window).sum()
        alpha = result

        return alpha

    #--------------------------------------------------------------------------
    def alpha_044(self, n = 6.0, m = 10.0, 
                  low_window = 7, volume_window = 10, weight_window = 6, 
                  rank_window = 4, vwap_shift_window = 3, 
                  vwap_weight_window = 10, vwap_rank_window = 15):
        seq1 = [2 * i / (n * (n + 1)) for i in np.arange(1, n + 1)]
        seq2 = [2 * i / (m * (m + 1)) for i in np.arange(1, m + 1)]
        weight1 = np.array(seq1)
        weight2 = np.array(seq2)
        
        temp1 = self.low.rolling(window = low_window).corr(
            self.volume.rolling(window = volume_window).mean())
        
        part1 = temp1.rolling(window = weight_window).apply(
            lambda x: (x.T * weight1).T.sum())
        part1 = part1.rolling(window = rank_window).apply(
            self.time_series_rank)

        temp2 = self.vwap.diff(vwap_shift_window)
        
        part2 = temp2.rolling(window = vwap_weight_window).apply(
            lambda x: (x.T * weight2).T.sum())
        part2 = part2.rolling(window = vwap_rank_window).apply(
            self.time_series_rank)
        
        alpha = part1 + part2
        alpha = alpha.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_046(self, window_1 = 3, window_2 = 6, 
                  window_3 = 12, window_4 = 24):
        part1 = self.close.rolling(window = window_1).mean()
        part2 = self.close.rolling(window = window_2).mean()
        part3 = self.close.rolling(window = window_3).mean()
        part4 = self.close.rolling(window = window_4).mean()
        
        result = (part1 + part2 + part3 + part4) * 0.25 / self.close
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_047(self, rolling_window = 6, alpha = 1.0 / 9.0):
        part1 = (self.high.rolling(window = rolling_window).max() 
                 - self.close)
        part2 = (self.high.rolling(window = rolling_window).max() 
                 - self.low.rolling(window = rolling_window).min())
        result = (100 * part1 / part2).ewm(alpha = alpha).mean()
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_052(self, delay_window = 1, rolling_window = 26):
        delay = ((self.high + self.low + self.close) / 3).shift(delay_window)
        
        part1 = np.maximum(self.high - delay, 0)
        part2 = np.maximum(delay - self.low, 0)
        
        alpha = (part1.rolling(window = rolling_window).sum() 
                 / part2.rolling(window = rolling_window).sum() * 100)

        return alpha

    #--------------------------------------------------------------------------
    def alpha_053(self, delay_window = 1, rolling_window = 12):
        delay = self.close.shift(delay_window) 
        condition = self.close > delay
        result = self.close[condition]
        alpha = (result.rolling(window = rolling_window).count() 
                 * 100 / rolling_window)

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_057(self, rolling_window = 9, alpha = 1.0 / 3.0):
        part1 = self.close - self.low.rolling(window = rolling_window).min()
        part2 = (self.high.rolling(window = rolling_window).max() 
                 - self.low.rolling(window = rolling_window).min())
        
        result = (100 * part1 / part2).ewm(alpha = alpha).mean()
        alpha = result

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_058(self, delay_window = 1, rolling_window = 20):
        delay = self.close.shift(delay_window) 
        condition = self.close > delay
        result = condition.rolling(window = rolling_window).sum()
        alpha = result / rolling_window * 100

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_059(self, delay_window = 1, rolling_window = 20):
        delay = self.close.shift(delay_window)
        condition1 = (self.close > delay)
        condition2 = (self.close < delay)
        
        part1 = np.minimum(self.low[condition1], delay[condition1]).fillna(0)
        part2 = np.maximum(self.high[condition2], delay[condition2]).fillna(0)

        result = self.close - part1 - part2
        alpha = result.rolling(window = rolling_window).sum()

        return alpha

    #--------------------------------------------------------------------------
    def alpha_060(self, rolling_window = 20): 
        part1 = (self.close - self.low) - (self.high - self.close)
        part2 = (self.high - self.low)
        
        result = self.volume * part1 / part2
        alpha = result.rolling(window = rolling_window).sum()

        return alpha

    #--------------------------------------------------------------------------
    def alpha_065(self, rolling_window = 6):
        part1 = self.close.rolling(window = rolling_window).mean()
        alpha = part1 / self.close

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_066(self, rolling_window = 6):
        part1 = self.close.rolling(window = rolling_window).mean()
        alpha = (self.close - part1) / part1

        return alpha

    #--------------------------------------------------------------------------
    def alpha_067(self, delay_window = 1, alpha = 1.0 / 24.0):
        temp1 = self.close - self.close.shift(delay_window)
        part1 = np.maximum(temp1, 0)
        part1 = part1.ewm(alpha = alpha).mean()
        
        temp2 = temp1.abs()
        part2 = temp2.ewm(alpha = alpha).mean()
        
        result = part1 / part2 * 100
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_068(self, delay_window = 1, alpha = 2.0 / 15.0):
        part1 = (self.high + self.low) / 2
        part2 = ((self.high.shift(delay_window) 
                  + self.low.shift(delay_window)) / 2)
        result = ((part1 + part2) * (self.high - self.low) 
                  / self.volume)
        result = result.ewm(alpha = alpha).mean()
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_070(self, rolling_window = 6):
        alpha = self.amount.rolling(
            window = rolling_window).std().dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_071(self, rolling_window = 24):
        rolling_mean = self.close.rolling(window = rolling_window).mean()
        result = (self.close - rolling_mean) / rolling_mean

        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_072(self, rolling_window = 6, alpha = 1.0 / 15.0):
        part1 = (self.high.rolling(window = rolling_window).max() 
                 - self.close)
        part2 = (self.high.rolling(window = rolling_window).max() 
                 - self.low.rolling(window = rolling_window).min())
        
        alpha = (part1 / part2 * 100).ewm(
            alpha = alpha).mean().dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_076(self, delay_window = 1, rolling_window = 20):
        delay = self.close.shift(delay_window)
        part1 = np.abs(self.close / delay - 1) / self.volume
        alpha = (part1.rolling(window = rolling_window).std() 
                 / part1.rolling(window = rolling_window).mean())

        return alpha

    #--------------------------------------------------------------------------
    def alpha_078(self, rolling_window = 12, mod_multi = 0.015):
        data1 = ((self.high + self.low + self.close) / 3
                 - ((self.high + self.low + self.close) / 3).rolling(
                     window = rolling_window).mean())
        data2 = abs(self.close
                    - ((self.high + self.low + self.close) / 3).rolling(
                        window = rolling_window).mean())
        data3 = data2.rolling(window = rolling_window).mean() * mod_multi
        alpha = (data1 / data3).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_079(self, delay_window = 1, alpha = 1.0 / 12.0):
        data1 = np.maximum(
            (self.close - self.close.shift(delay_window)), 0).ewm(
                alpha = alpha).mean()
        data2 = abs(self.close - self.close.shift(delay_window)).ewm(
            alpha = alpha).mean()
        alpha = (data1 / data2 * 100).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_080(self, delay_window = 5):
        alpha = (((self.volume - self.volume.shift(delay_window)) 
                  / self.volume.shift(delay_window) * 100).dropna(how = 'all'))

        return alpha

    #--------------------------------------------------------------------------
    def alpha_081(self, alpha = 2.0 / 21.0):
        result = self.volume.ewm(alpha = alpha).mean()
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_082(self, rolling_window = 6, alpha = 1.0 / 20.0):
        part1 = self.high.rolling(window = rolling_window).max() - self.close
        part2 = (self.high.rolling(window = rolling_window).max() 
                 - self.low.rolling(window = rolling_window).min())
        result = (100 * part1 / part2).ewm(alpha = alpha).mean()
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_084(self, delay_window = 1, rolling_window = 20):
        condition1 = (self.close > self.close.shift(delay_window))
        condition2 = (self.close < self.close.shift(delay_window))
        
        part1 = self.volume[condition1].fillna(0)
        part2 = -self.volume[condition2].fillna(0)
        
        result = part1 + part2
        alpha = result.rolling(window = rolling_window).sum().dropna(
            how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_085(self, volume_window = 20, 
                  delay_window = 7, rank_window = 8):
        temp1 = (self.volume 
                 / self.volume.rolling(window = volume_window).mean())
        part1 = temp1.rolling(window = volume_window).apply(
            self.time_series_rank)

        delta = self.close.diff(delay_window)
        temp2 = -delta
        part2 = temp2.rolling(window = rank_window).apply(
            self.time_series_rank)

        alpha = part1 * part2

        return alpha.dropna(how = 'all')

    #--------------------------------------------------------------------------
    def alpha_086(self, short_shift = 10, long_shift = 20, 
                  upper_threshold = 0.25, lower_threshold = 0.0):
        delay_gap = long_shift - short_shift
        delay10 = self.close.shift(short_shift)
        delay20 = self.close.shift(long_shift)
        
        indicator1 = pd.DataFrame(
            -np.ones(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns) 
        indicator2 = pd.DataFrame(
            np.ones(self.close.shape), 
            index = self.close.index, 
            columns = self.close.columns) 

        temp = ((delay20 - delay10) / delay_gap 
                - (delay10 - self.close) / delay_gap)
        condition1 = (temp > upper_threshold)
        condition2 = (temp < lower_threshold)
        temp2 = (self.close - self.close.shift()) * indicator1

        part1 = indicator1[condition1].fillna(0)
        part2 = indicator2[~condition1][condition2].fillna(0)
        part3 = temp2[~condition1][~condition2].fillna(0)
        result = part1 + part2 + part3
        alpha = result.dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_088(self, delay_window = 20):
        alpha = (((self.close - self.close.shift(delay_window)) 
                  / self.close.shift(20)) * 100).dropna(how = 'all')

        return alpha

    #--------------------------------------------------------------------------
    def alpha_089(self, alpha_1 = 2.0 / 13.0, 
                  alpha_2 = 2.0 / 27.0, alpha_3 = 2.0 / 10.0, 
                  mod_multi = 2): 
        data1 = self.close.ewm(alpha = alpha_1).mean()
        data2 = self.close.ewm(alpha = alpha_2).mean()
        data3 = (data1 - data2).ewm(alpha = alpha_3).mean()
        alpha = ((data1 - data2 - data3) * mod_multi).dropna(how = 'all')

        return alpha


###############################################################################
###############################################################################
# The factors from here are from Zhang's code.
# The original codes are in the folder ./reference/180820_features.txt 
# and ./reference/180820_func.txt.
class ReturnFeatures:

    #--------------------------------------------------------------------------
    def __init__(self, data, return_df, return_var):
        self.return_series = return_df[return_var].copy()
        self.open_price = data['open']
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.pre_close = data['close'].shift()
        self.vwap = data['vwap']
        self.volume = data['volume']
        self.amount = data['amount']
        self.opint = data['openint']

    #--------------------------------------------------------------------------
    def open_gap(self):
        alpha = self.open_price / self.pre_close - 1
        return alpha

    #--------------------------------------------------------------------------
    def ewm_avg(self, com):
        alpha = self.return_series.ewm(com = com).mean()
        return alpha

    #--------------------------------------------------------------------------
    def mv(self, rolling_window):
        temp_return_series = self.return_series.dropna(how = 'all').copy()
        alpha = (
            temp_return_series.rolling(window = rolling_window).mean()
            / temp_return_series.rolling(window = rolling_window).std())
        return alpha

    #--------------------------------------------------------------------------
    def mv_variable_std(self, rolling_window, std_rolling_window):
        temp_return_series = self.return_series.dropna(how = 'all').copy()
        alpha = (temp_return_series.rolling(window = rolling_window).mean() 
                 / temp_return_series.rolling(
                     window = std_rolling_window).std())
        return alpha

    #--------------------------------------------------------------------------
    def ewmv(self, com):
        alpha = (self.return_series.ewm(com = com).mean() 
                 / self.return_series.ewm(com = com).std())
        return alpha

    #--------------------------------------------------------------------------
    def mom_std(self, rolling_window):
        temp_return_series = self.return_series.dropna(how = 'all').copy()
        alpha = (temp_return_series.rolling(window = rolling_window).sum() 
                 / temp_return_series.rolling(window = rolling_window).std())
        return alpha

    #--------------------------------------------------------------------------
    def mom(self, rolling_window):
        temp_return_series = self.return_series.dropna(how = 'all').copy()
        alpha = temp_return_series.rolling(window = rolling_window).sum()
        return alpha

    #--------------------------------------------------------------------------
    def new_mom(self, rolling_window):
        '''
        This method is desinged to deal with the problem that the rolling sum 
        method could not skip the NaN in the DataFrame, thus will yield the 
        wrong rolling sum, or just NaN.
        This method calculate the momentum of each column in the return series 
        DataFrame separately by dropping the NaN first, thus forming a 
        continuous return series.
        '''
        temp_return_series = self.return_series.copy()
        alpha_df = pd.DataFrame(index = self.return_series.index)

        for column in temp_return_series:
            column_mom = temp_return_series.loc[:, [column]]
            column_mom = column_mom.dropna()
            column_mom_complete = column_mom.rolling(
                window = rolling_window).sum()
            alpha_df = pd.concat([alpha_df, column_mom_complete], axis = 1)

        return alpha_df

    #--------------------------------------------------------------------------
    def barra_mom(self, rolling_window, half_life, lag_window):
        '''
        This method of momentum calculation is based on Barra model. 
        It is calculated as the exponential weighted average of the 
        past return with a lag of L days.
        '''
        total_weight = sum([2**(-i/half_life) 
                            for i in range(1, rolling_window+1)])
        half_life_weight = [
            2**(float(i-rolling_window-1)/half_life)/float(total_weight) 
            for i in range(1, rolling_window+1)]

        temp_return_series = self.return_series.dropna(
            how = 'all').shift(lag_window).copy()
        alpha = temp_return_series.rolling(window = rolling_window).apply(
            lambda x: (x.T * half_life_weight).T.sum(), raw = True)

        return alpha

    #--------------------------------------------------------------------------
    def ret_h1(self, threshold, rolling_window):
        temp_return_series = self.return_series.dropna(how = 'all').copy()
        temp_return_series[temp_return_series.abs() <= threshold] = 0.0
        alpha = temp_return_series.rolling(window = rolling_window).sum()
        return alpha

    #--------------------------------------------------------------------------
    def opvr(self, rolling_window):
        opv = (abs(self.opint - self.opint.shift()).div(
            self.volume).multiply(np.sign(self.return_series), axis = 0))
        alpha = opv.rolling(window = rolling_window).mean()
        return alpha

    #--------------------------------------------------------------------------
    def retv(self, com):
        alpha = (self.return_series / np.sqrt(self.volume)).ewm(
            com = com).mean()
        return alpha

    #--------------------------------------------------------------------------
    def retv2(self, com):
        alpha = (self.return_series / np.log(self.volume)).ewm(
            com = com).mean()
        return alpha

    #--------------------------------------------------------------------------
    def retl(self, threshold, com):
        temp_return_series = self.return_series.dropna(how = 'all').copy()
        temp_return_series[temp_return_series > threshold] = threshold
        temp_return_series[temp_return_series < -threshold] = -threshold

        alpha = temp_return_series.ewm(com = com).mean()
        return alpha

    #--------------------------------------------------------------------------
    def ovr(self, rolling_window):
        op_chg = self.opint / self.opint.shift() - 1
        alpha = (op_chg * self.return_series).rolling(
            window = rolling_window).mean()
        return alpha

    #--------------------------------------------------------------------------
    def ovr2(self, rolling_window):
        op_chg = self.opint / self.opint.shift() - 1
        target_series = op_chg * self.return_series
        alpha = (target_series.rolling(window = rolling_window).mean() 
                 / target_series.rolling(window = rolling_window).std())
        return alpha

    #--------------------------------------------------------------------------
    def ovr3(self, rolling_window):
        op_chg = self.opint / self.opint.shift() - 1
        target_series = np.sign(op_chg) * self.return_series
        target_series[target_series == np.inf] = 0
        alpha = target_series.rolling(window = rolling_window).mean() 
        return alpha

    #--------------------------------------------------------------------------
    def close_high_ratio(self, rolling_window):
        '''
        This feature is calculated as the ratio of the current close price 
        to the highest close price in the recent period, which is denoted 
        by rolling_window.
        '''
        alpha = (self.close.astype(np.float32) 
                 / self.close.rolling(
                     window = rolling_window).max().astype(np.float32))
        return alpha

    #--------------------------------------------------------------------------
    def high_high_ratio(self, rolling_window):
        '''
        This feature is calculated as the ratio of the current close price 
        to the highest high price in the recent period, which is denoted 
        by rolling_window.
        '''
        alpha = (self.close.astype(np.float32)
                 / self.high.rolling(
                     window = rolling_window).max().astype(np.float32))
        return alpha

    #--------------------------------------------------------------------------
    def close_low_ratio(self, rolling_window):
        '''
        This feature is calculated as the ratio of the lowest close price 
        to the current close price during the recent period, which is 
        denoted by rolling_window.
        '''
        alpha = (self.close.rolling(
            window = rolling_window).min().astype(np.float32) 
                 / self.close.astype(np.float32))
        return alpha

    #--------------------------------------------------------------------------
    def low_low_ratio(self, rolling_window):
        '''
        This feature is calculated as the ratio of the lowest low price 
        to the current close price during the recent period, which is 
        denoted by rolling_window.
        '''
        alpha = (self.low.rolling(
            window = rolling_window).min().astype(np.float32) 
                 / self.close.astype(np.float32))
        return alpha

    #--------------------------------------------------------------------------
    def return_consistency(self, rolling_window, consistent_threshold):
        temp_return_series = self.return_series.dropna(how = 'all').copy()

        momentum = temp_return_series.rolling(window = rolling_window).sum()

        return_sign = temp_return_series.copy()
        return_sign[temp_return_series > 0.0] = 1
        return_sign[temp_return_series <= 0.0] = 0

        pos_pct = (return_sign.rolling(
            window = rolling_window).sum().astype(np.float32) 
            / float(rolling_window))
        neg_pct = 1 - pos_pct
        
        alpha = pos_pct.copy()
#        alpha[((pos_pct >= consistent_threshold) 
#               | (neg_pct >= consistent_threshold))] = 1
        alpha[pos_pct >= consistent_threshold] = 1
        alpha[neg_pct >= consistent_threshold] = -1
        alpha[((pos_pct < consistent_threshold) 
               & (neg_pct < consistent_threshold))] = 0

        final_alpha = alpha.multiply(momentum, axis = 0)
        
        return final_alpha

    #--------------------------------------------------------------------------
    def continuous_information(self, rolling_window):
        temp_return_series = self.return_series.dropna(how = 'all').copy()

        momentum = temp_return_series.rolling(window = rolling_window).sum()
        momentum_sign = momentum.copy()
        momentum_sign[momentum > 0.0] = 1
        momentum_sign[momentum <= 0.0] = -1

        return_sign = temp_return_series.copy()
        return_sign[temp_return_series > 0.0] = 1
        return_sign[temp_return_series <= 0.0] = 0

        pos_pct = (return_sign.rolling(window = rolling_window).sum().astype(
            np.float32) / float(rolling_window))
        neg_pct = 1 - pos_pct
        
        alpha = (pos_pct - neg_pct).multiply(momentum_sign, axis = 0)

        return alpha

    #--------------------------------------------------------------------------
    def volume_adj(self, short_window, long_window):
        alpha = ((self.volume.rolling(window = short_window).mean() 
                  - self.volume.rolling(window = long_window).mean()) 
                 / self.volume.rolling(window = long_window).std())

        return alpha

    #--------------------------------------------------------------------------
    def amount_adj(self, short_window, long_window):
        alpha = ((self.amount.rolling(window = short_window).mean()
                  - self.amount.rolling(window = long_window).mean()) 
                 / self.amount.rolling(window = long_window).std())

        return alpha

    #--------------------------------------------------------------------------
    def amihud(self, rolling_window):
        alpha = self.return_series.abs().div(self.volume, axis = 0).rolling(
            window = rolling_window).mean()

        return alpha

    #--------------------------------------------------------------------------
    def amihud_adj(self, short_window, long_window):
        amihud = self.return_series.abs().div(self.amount, axis = 0).rolling(
            window = short_window).mean()

        alpha = -1 * ((amihud - amihud.rolling(window = long_window).mean()) 
                      / amihud.rolling(window = long_window).std())

        return alpha

    #--------------------------------------------------------------------------
    def MAR(self, price, short_window, long_window):
        '''
        Moving average ratio. Calculated as the ratio of the short-term 
        price moving average and the long-term price moving average.
        '''
        alpha = price.rolling(window = short_window).mean().div(
            price.rolling(window = long_window).mean(), axis = 0)

        return alpha

    #--------------------------------------------------------------------------
    def vol_opint_diff_ratio(self):
        alpha = (self.opint.diff().diff(20).astype(np.float32) 
                 / self.volume.astype(np.float32))

        return alpha


###############################################################################
class HFFeatures:

    #--------------------------------------------------------------------------
    def __init__(self, data):
        self.last_price = data['last_price']
        self.mid_price = data['mid_price']
        self.bid_price1 = data['bid_price1']
        self.bid_volume1 = data['bid_volume1']
        self.ask_price1 = data['ask_price1']
        self.ask_volume1 = data['ask_volume1']
        self.volume = data['volume']
        self.turnover = data['turnover']
        self.openint = data['openint']

    #--------------------------------------------------------------------------
    def LR_trade_class(self):
        '''
        Lee-Ready algorithm for trade direction classification.
        '''

        last_mid_gap_df = pd.DataFrame(index = self.last_price.index)
        last_mid_gap_df = self.last_price - self.mid_price

        last_prev_gap_df = self.last_price.diff()
        last_prev_gap_df = last_prev_gap_df.fillna(0)

        # Get the last price change for tick test.
        last_change_df = last_prev_gap_df.replace(
            to_replace = 0, method = 'ffill')

        # The Lee-Ready algorithm.
        # First commit the mid-price test; if the mid-price test can not 
        # tell the direction of the tick, use the tick test.
        # buy: 1, sell: -1
        buy_sell_df = last_mid_gap_df.copy()
        buy_sell_df[last_mid_gap_df != 0] = np.sign(last_mid_gap_df)
        buy_sell_df[last_mid_gap_df == 0] = np.sign(last_change_df)
        buy_sell_df = buy_sell_df.replace(
            to_replace = 0, method = 'bfill')

        return buy_sell_df

    #--------------------------------------------------------------------------
    def voi(self):
        # Calculate the volume order imbalance (VOI) feature.
        ov_bid_df = self.bid_price1.copy()
        ov_bid_df[self.bid_price1 < self.bid_price1.shift()] = 0
        ov_bid_df[self.bid_price1 == self.bid_price1.shift()] = (
            self.bid_volume1.diff())
        ov_bid_df[self.bid_price1 > self.bid_price1.shift()] = (
            self.bid_volume1)

        ov_ask_df = self.ask_price1.copy()
        ov_ask_df[self.ask_price1 < self.ask_price1.shift()] = (
            self.ask_volume1)
        ov_ask_df[self.ask_price1 == self.ask_price1.shift()] = (
            self.ask_volume1.diff())
        ov_ask_df[self.ask_price1 > self.ask_price1.shift()] = 0

        voi_df = ov_bid_df - ov_ask_df
        voi_df[(voi_df.abs() < 1.0) & (voi_df.abs() > 0.0)] = np.nan

        return voi_df

    #--------------------------------------------------------------------------
    def oir(self):
        # Calculate order imbalance ratio (OIR).
        oir_df = ((self.bid_volume1 - self.ask_volume1) 
                  / (self.bid_volume1 + self.ask_volume1))

        return oir_df

    #--------------------------------------------------------------------------
    def reversion_rate(self, size):
        tp_df = pd.DataFrame(index = self.volume.index)

        for column in self.volume.columns:
            tp_df[column] = np.nan

        tp_df[self.volume != self.volume.shift()] = (
            (self.turnover - self.turnover.shift()) 
            / (self.volume - self.volume.shift()) 
            / float(size))
        tp_df = tp_df.fillna(method = 'ffill')

        rev_df = tp_df - (self.mid_price + self.mid_price.shift()) * 0.5

        return rev_df

    #--------------------------------------------------------------------------
    def bid_ask_spread(self):
        spread_df = self.ask_price1 - self.bid_price1

        return spread_df

    #--------------------------------------------------------------------------
    def ma_diff(self, price_var, rolling_window):
        price_df = getattr(self, price_var)

        ma_diff_df = price_df.rolling(window = rolling_window).mean().diff()

        return ma_diff_df

    #--------------------------------------------------------------------------
    def cur_ma_diff(self, price_var, rolling_window):
        price_df = getattr(self, price_var)

        rolling_mean_df = price_df.rolling(window = rolling_window).mean()
        cur_ma_diff_df = price_df - rolling_mean_df

        return cur_ma_diff_df




