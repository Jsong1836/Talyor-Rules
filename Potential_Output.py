# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 02:36:14 2021

@author: SONG
"""

import numpy as np
import pandas as pd
import datetime as dt

# CBO Potential Output is computed from the Real Gross Domestic Product
# including <nonfarm business, goverment, farm, household and non-profit institutions and residential housing  

class DataLoader(object):
    def __init__(self, start_date = None, end_date = None):
        self.start_date = start_date
        self.end_date = end_date
        if start_date is not None:
            self.start_date = start_date
        else:
            start_date = "1800-01-01"
        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = dt.date.today()
            
    # Single Data import function         
    def request(self, ticker):
        self.ticker = str(ticker)
        from fredapi import Fred
        import datetime as dt
        fred = Fred(api_key = api_key)
        dataframe = fred.get_series(self.ticker, observation_start = self.start_date,
                                    observation_end = self.end_date)
        dataframe = pd.DataFrame(dataframe, columns= [str(self.ticker)])
        return dataframe
    
    # Main Data import function, all data needed is called by this function
    def Fred(self, tickers):
        self.tickers = tickers
        dataframe = [self.request(data) for data in self.tickers]
        dataframe = pd.concat((dataframe[i] for i in range(len(self.tickers))), axis = 1)  
        return dataframe
    
    def Stock(self, tickers):
        import yfinance
        self.tickers = tickers
        dataframe = yfinance.download(tickers = tickers, start = self.start_date, end = self.end_date)["Adj Close"]
        return dataframe 
    
    # As Fred data source came from BEA calculations based on weighted Fisher index, 
    # there is no need to calculate Fisher weighted GDP calculations. 
    
    def GDP(self):
        # Gross value added: GDP: Business: Nonfarm
        # Gross value added: GDP: General government
        # Gross value added: GDP: Business: Farm
        # Households and institutions
        # Gross value added: GDP: Housing
        # GNP
        tickers = ["A358RC1Q027SBEA","A767RC1Q027SBEA", "B359RC1Q027SBEA", 
                   "B702RC1Q027SBEA", "A2009C1Q027SBEA"]
        data = self.Fred(tickers = tickers)
        data["gdp"] = data.sum(axis = 1)
        return data
    
    # Long-term Solow Growth Model. For Labor, it is "hours worked in the nonfarm business sectors 
    # that can be decomposed into 3 categories
    # Labor force, employment, average weekly hours 
    
class PotentialOutput(DataLoader):
    def __init__(self):
        super().__init__()
        
    def Inflation(self):
        # Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average
        # Unemployment rate
        tickers = ["CPILFESL", "UNRATE"]
        data = self.Fred(tickers = tickers)
        data["CPILFESL"] = round(data["CPILFESL"].pct_change(periods = 12) * 100, 3)
        data.dropna(inplace = True)
        return data
    
    # Taking 12 Lags for CPI and UNRATE  
    def AR_data(self):
        tickers = ["UNRATE", "CPILFESL"]
        data = self.Fred(tickers = tickers)
        data["CPILFESL"] = data["CPILFESL"] - data["CPILFESL"].shift(1)
        data = pd.concat([data.shift(i) for i in range(12)], axis = 1)
        data.dropna(inplace = True)
        pi = data.get("CPILFESL")
        u = data.get("UNRATE")
        # Pi = inflation rate, u = Unemployement rate
        concat = pd.concat([u, pi], axis = 1)
        return concat
    
    # This function is a standard OLS estimation function with a constant  
    def OLS(self, endog, exog):
        import statsmodels.api as sm
        
        if isinstance(endog, pd.DataFrame) == True:
            endog = endog.values
        if isinstance(exog, pd.DataFrame) == True:
            exog = exog.values
            
        if len(endog) != len(exog):
            raise ValueError("Dimension mismatch")
            
        X = np.matrix(endog)
        constant = np.ones(endog.shape[0])
        X = np.insert(X, 0, constant, axis = 1)
        y = np.array(exog)
        coef = (np.linalg.pinv(X.T @ X)) @ X.T @ y 
        return coef
    
    # Comparing NAIRU with CBO's caluclation
    def nairu(self):
        tickers = ["UNRATE", "NROU"]
        data = self.Fred(tickers = tickers)
        data = data.resample("Q").mean()
        data.dropna(inplace = True)
        data["u_star"] = data["UNRATE"] - data["NROU"]
        return data
        
    # Calculating NAIRU with AR (12).
    # returning U* being NAIRU and U - U* 
    def NAIRU(self):
        data = self.AR_data()
        unrate = data.get("UNRATE").iloc[:, 0]
        index = data.index[12:]
        coef = list()
        
        for i in range(int(data.shape[0] - 12)):
            coef.append(self.OLS(endog = data.iloc[i:i+12, :], exog = data.iloc[i:i+12, 0]))
        
        # taking mu + lag 12 of unrate
        coef = np.array(coef)
        mu = np.array(coef[:, :, 0]).flatten()
        beta = np.array(coef[:, :, 1:13])
        beta = beta.reshape(beta.shape[0], -1)
        beta_sum = np.sum(beta, axis = 1)
        u_star = -mu / beta_sum
        u_start_df = pd.DataFrame(u_star, index = index, columns = ["nairu"])
        concat = pd.concat([u_start_df, unrate], axis = 1)
        concat.dropna(inplace = True)
        gap = concat.get("UNRATE") - concat.get("nairu")
        return u_start_df, gap
        
    # Equation tpye of Cobb-Douglas
    # Y = A * np.power(L, (1 - alpha)) * np.power(K, alpha)
    # CBO sets alpha to 0.3 empirically. Econometric estimation dosn't yield a good result

    # LF_Star correspond to potential Labor force estimatied by next function
    # log(LF_start) = f(Ti) where Ti takes zeros until the business-cycles peak occuring in year i 
    # otherwise it takes the number of questers elapsed
    
    def LFStar(self):
        from scipy.signal import find_peaks
        #  Nonfarm Business Sector: Hours Worked for All Employed Persons
        nairu = self.nairu()["u_star"]
        index = nairu.index
        peaks, _ = find_peaks(nairu, prominence= 1)
        LF_star = np.zeros(nairu.shape[0])
        
        d = np.diff(peaks)
        d = np.insert(d, 0, 3)
        
        for i in range(len(d)):
            LF_star[peaks[i]] = d[i]
            
        return pd.DataFrame(np.exp(LF_star), index = index, columns = ["lfstar"])
    
    # Cyclical Adjustment Equation 
    # log(LF_Star) = f(Ti)
    # log(LF / LF *) = alpha * (U - U*) + epsillon
    
    # Potential Employment 
    def Empl(self):
        nairu = self.nairu()
        lfstar = self.LFStar()
        
        concat = pd.concat([nairu, lfstar], axis = 1)
        empl = (1 - (concat.get("u_star") / 100) * concat.get("lfstar"))
        return empl
    
# CBO method calculation is difficult because of limited access to necessary data. 