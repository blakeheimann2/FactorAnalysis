from lib.analysis import *

def main():
    #PORTFOLIO ALLOVCATIONS BY PERCENT
    equalWeightPortfolio = {"AAPL" : .1,
                       "TSLA" : .1,
                       "SPCE": .1,
                       "GE": .1,
                       "TMUS": .1,
                       "DIS": .1,
                       "DAL": .1,
                       "NKE": .1,
                       "GLD": .1,
                       "USO": .1}
    portfolio = getEquityPortfolio(equalWeightPortfolio.keys(), start= "2019-01-01", end="2020-12-01")
    portfolioReturns = portfolio.dropna().apply(calcLogReturns, axis =0)
    myPortfolioReturns = np.sum(portfolioReturns * list(equalWeightPortfolio.values()), axis=1)
    PortfolioFactorReg(myPortfolioReturns)

if __name__ == "__main__":
    main()
