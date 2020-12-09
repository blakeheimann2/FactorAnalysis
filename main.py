from lib.analysis import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def main():
    #PORTFOLIO ALLOVCATIONS BY PERCENT
    equalWeightPortfolio = {"AAPL" : .1,
                       "TSLA" : .1,
                       "SPCE": .1,
                       "AMZN": .1,
                       "PPL": .1,
                       "DIS": .1,
                       "TDOC": .1,
                       "NKE": .1,
                       "ICLN": .1,
                       "ESRT": .1}
    portfolio = getEquityPortfolio(equalWeightPortfolio.keys(), start= "2019-01-01", end="2020-12-01")
    portfolioReturns = portfolio.dropna().apply(calcLogReturns, axis =0)
    myPortfolioReturns = np.sum(portfolioReturns * list(equalWeightPortfolio.values()), axis=1)
    PortfolioFactorReg(myPortfolioReturns)
    print("My Portfolio Cummulative Returns: {}".format(myPortfolioReturns.sum()))

def modeling():
    raw = pd.read_csv("ratios-preprocessed-nov7.csv")
    ratios = list(raw.columns[-20:])
    ratios.remove("asset_turnover_ratio")
    ratios.remove("asset_turnover_ratio_usd")
    important_cols = ["ticker", "sector", "reportperiod", "price"] + ratios
    print(important_cols)
    data = raw[important_cols]
    data['price_lag'] = (data.sort_values(by=['reportperiod'], ascending=True)
                         .groupby(['ticker'])['price'].shift(1))
    data['price_lead'] = (data.sort_values(by=['reportperiod'], ascending=True)
                         .groupby(['ticker'])['price'].shift(-1))
    data['returns'] = np.log(data.price_lead / data.price_lag)
    data['future_returns'] = np.log(data.price_lead / data.price_lag)


    df = data[ratios + ["future_returns"]].dropna()
    X = df[ratios]
    y = df["future_returns"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Principle components regression
    steps = [
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('estimator', SMWrapper(sm.GLS, fit_intercept=False))
    ]
    pipe = Pipeline(steps)
    pipe.set_params(pca__n_components=3)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(pipe['estimator'].summary())






if __name__ == "__main__":
     modeling()
