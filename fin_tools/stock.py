import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tabulate import tabulate
import yfinance as yf
import datetime


class Stock:
    def __init__(self, stock_price, stock_name):
        self.stock_price = stock_price
        self.trading_days = len(stock_price)
        self.stock_name = stock_name


        self.returns = np.diff(self.stock_price) / self.stock_price[:-1]
        self.returns = np.insert(self.returns, 0, 0)
        self.volatility = np.std(self.returns)
        self.expected_return = np.mean(self.returns)

        # generate a random normal distribution
        self.stock_dist = np.random.normal(self.expected_return, self.volatility, self.trading_days)
        self.expected_return_fit, self.volatility_fit = stats.norm.fit(self.stock_dist)
    
    def copy(self):
        return Stock(self.stock_price, self.stock_name)

    def get_stock_price(self):
        return self.stock_price

    def get_stock_name(self):
        return self.stock_name

    def get_returns(self):
        return self.returns
    
    def get_volatility(self):
        return self.volatility
    
    def get_expected_return(self):
        return self.expected_return
        
    def get_expected_return_fit(self):
        return self.expected_return_fit
    
    def get_volatility_fit(self):
        return self.volatility_fit
        
    def get_stock_dist(self):
        return self.stock_dist