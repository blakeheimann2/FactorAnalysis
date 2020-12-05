import pandas as pd
import numpy as np

import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
import pandas_datareader.data as web  # module for reading datasets directly from the web

import pandas_datareader.data as web



def getEquity(ticker, start, end, subset = None):
    stockdata = web.DataReader(ticker,
                           start=start,
                           end=end,
                           data_source='yahoo')
    if bool(subset):
        stockdata = stockdata[subset]
    return stockdata

def getEquityPortfolio(ticker_list, start, end, price_type = "Adj Close"):
    daterange = pd.date_range(start=start, end=end)
    portfoliodata = pd.DataFrame(index=daterange)
    for stock in ticker_list:
        stockdata = getEquity(stock, start, end, price_type)
        stockdata.name = str(stock)
        portfoliodata = pd.concat([portfoliodata, stockdata], axis=1)
    return portfoliodata

def calcLogReturns(price_series):
    logreturns = np.log(price_series / price_series.shift(1))
    return logreturns


def PortfolioFactorReg(df_stk):
    # Reading in factor data
    df_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
    df_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
    df_factors['MKT'] = df_factors['MKT'] / 100
    df_factors['SMB'] = df_factors['SMB'] / 100
    df_factors['HML'] = df_factors['HML'] / 100
    df_factors['RMW'] = df_factors['RMW'] / 100
    df_factors['CMA'] = df_factors['CMA'] / 100
    df_stk.name = "Returns"
    df_stock_factor = pd.concat([df_stk, df_factors], axis=1).dropna()  # Merging the stock and factor returns dataframes together
    df_stock_factor['XsRet'] = df_stock_factor['Returns'] - df_stock_factor['RF']  # Calculating excess returns

    # Running CAPM, FF3, and FF5 models.
    CAPM = sm.ols(formula='XsRet ~ MKT', data=df_stock_factor).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    FF3 = sm.ols(formula='XsRet ~ MKT + SMB + HML', data=df_stock_factor).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    FF5 = sm.ols(formula='XsRet ~ MKT + SMB + HML + RMW + CMA', data=df_stock_factor).fit(cov_type='HAC',
                                                                                          cov_kwds={'maxlags': 1})

    CAPMtstat = CAPM.tvalues
    FF3tstat = FF3.tvalues
    FF5tstat = FF5.tvalues

    CAPMcoeff = CAPM.params
    FF3coeff = FF3.params
    FF5coeff = FF5.params

    # DataFrame with coefficients and t-stats
    results_df = pd.DataFrame({'CAPMcoeff': CAPMcoeff, 'CAPMtstat': CAPMtstat,
                               'FF3coeff': FF3coeff, 'FF3tstat': FF3tstat,
                               'FF5coeff': FF5coeff, 'FF5tstat': FF5tstat},
                              index=['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

    dfoutput = summary_col([CAPM, FF3, FF5], stars=True, float_format='%0.4f',
                           model_names=['CAPM', 'FF3', 'FF5'],
                           info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                      'Adjusted R2': lambda x: "{:.4f}".format(x.rsquared_adj)},
                           regressor_order=['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

    print(dfoutput)

    return results_df

