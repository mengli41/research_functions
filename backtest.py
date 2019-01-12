import numpy as np
import pandas as pd



def backtest(pos, data, time_range=slice('2012', None), open_price='open', with_summary=True):
    """Calculate daily profits and statistics like sharpe ratio.

    Parameters
    ----------
    pos : pandas.DataFrame
        The target position at each day.
    data : pandas.Panel
        Must have close and close_preday in item labels.
    time_range : slice
        Starting date of the backtesting period.
    open_price : string
        Label used to get trading price at each day.
    with_summary : bool
        See below.
    Returns
    -------
    r : pandas.Series
        r is a series of returns. If with_summary == True, r has an addition
        attribute r.summary which contains some statistics like sharpe ratio.
    """
    pos.index=pos.index.strftime("%Y%m%d")

    r_sep = pos.shift() * (data[open_price] / data['close_preday'] - 1) +\
        pos * (data['close'] / data[open_price] - 1) -\
        pos.diff().abs() * 0.001
    r = r_sep.sum(axis=1)[time_range]

    if with_summary:
        summary = pd.Series()
        summary['price'] = open_price
        summary['sharpe'] = '{:.2f}'.format(r.mean() / r.std() * np.sqrt(242))
        summary['sr2018'] = '{:.2f}'.format(r.loc["2018":].mean() / r.loc["2018":].std() * np.sqrt(242))
        summary['calmar'] = '{:.2f}'.format(r.mean() * 242 / (r.cumsum().cummax() -
                                            r.cumsum()).max())
        summary['ret/y'] = '{:.2%}'.format(r.mean() * 242)
        summary['std'] = '{:.2%}'.format(r.std() * np.sqrt(242))
        lever = pos[time_range].abs().sum(axis=1).mean()
        summary['lever'] = '{:.1%}'.format(lever)
        turnover = pos[time_range].diff().abs().sum(axis=1).mean() / lever
        summary['turnover'] = '{:.1%}'.format(turnover)
        summary['mdd'] = '{:.2%}'.format((r.cumsum().cummax() - r.cumsum()).max())
        summary['cdd'] = '{:.2%}'.format(r.cumsum().max() - r.sum())
        summary['ret_16'] = '{:.2%}'.format(r['2016'].sum())
        summary['ret_17'] = '{:.2%}'.format(r['2017'].sum())
        summary['ret_18'] = '{:.2%}'.format(r['2018'].sum())
        summary['ret_19'] = '{:.2%}'.format(r['2019'].sum())
        summary['ret_1d'] = '{:.2%}'.format(r.iloc[-1].sum())
        summary['ret_5d'] = '{:.2%}'.format(r.iloc[-5:].sum())
        summary['ret_20d'] = '{:.2%}'.format(r.iloc[-20:].sum())
        summary['ret_60d'] = '{:.2%}'.format(r.iloc[-60:].sum())
        r.summary = summary
    return r

def backtest_summary(r, silent=False):
    """Calculate daily profits.

    Parameters
    ----------
    r : dict
        Dictionary of returns with summary
    silent : bool
        Print summary if silent == True.

    Returns
    -------
    summary : pd.DataFrame
        Statistics like Sharpe ratio.
    """
    summary = pd.DataFrame({k:v.summary for k, v in r.items()})
    if not silent:
        print(summary)
    return summary
