import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tabulate import tabulate
import yfinance as yf
import datetime

class Option:
    def __init__(self, ticker_symbol):
        if (ticker_symbol is None):
            self.stock_price = None
            self.strikes = None
            self.volatility = None
            self.call_prices = None
            self.time_to_maturity = None
        else:
            self.ticker = yf.Ticker(ticker_symbol)
            self.stock_price = self.ticker.history(period="1d")['Close'].iloc[-1]
            self.option_dates = self.ticker.options
            self.opt_chain = None
            self.risk_free_rate = None
            self.volatility = None
            self.option_dates_dict = {}

            if self.option_dates is not None:
                sorted_dates = sorted(self.option_dates)
            for idx, date in enumerate(sorted_dates, 1):
                self.option_dates_dict[idx] = date

            self.expiry = None
            self.time_to_maturity = None
            self.calls = None
            self.strikes = None
            self.last_price = None
            self.last_price_avg = None
            self.implied_volatilities = None
            self.option_contracts = None
            # calculated values
            self.calculated_IV = None
            self.calculated_IV_avg = None
            self.calculated_C = None

    def set_risk_free_rate(self, risk_free_rate):
        self.risk_free_rate = risk_free_rate
        
    def set_option_manual(self, stock_price, strikes, volatility, time_to_maturity, risk_free_rate):
        self.stock_price = stock_price
        self.strikes = strikes
        self.volatility = volatility
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate

    def set_option_chain(self, expiry):
        self.expiry = self.option_dates_dict[expiry]

        # calculate the time to maturity
        today = datetime.datetime.now().date()
        expiry_date = datetime.datetime.strptime(self.expiry, "%Y-%m-%d").date()
        self.time_to_maturity = (expiry_date - today).days / 365 

        # get the option chain
        self.opt_chain = self.ticker.option_chain(self.expiry)
        # get the calls
        self.calls = self.opt_chain.calls

        # get the strikes
        self.strikes = self.calls['strike'].values
        # get the option prices
        self.last_price = self.calls['lastPrice'].values
        # get the option prices average
        self.last_price_avg = (self.calls['bid'].values + self.calls['ask'].values) / 2
        # get the implied volatilities
        self.implied_volatilities = self.calls['impliedVolatility'].values
        # get the option contracts
        self.option_contracts = self.calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]



    def set_strikes(self, strike_price, N):
        # # Find strikes: at least 2 below, 2 above, and 1 closest to S0
        strikes_sorted = self.option_contracts['strike'].sort_values().values

        # # Find the index of the strike closest to S0
        closest_idx = (abs(strikes_sorted - strike_price)).argmin()

        indices = []
        for offset in range(-N, N+1):
            idx = closest_idx + offset
            if 0 <= idx < len(strikes_sorted):
                indices.append(idx)

        # # Remove duplicates and sort
        indices = sorted(set(indices))

        selected_strikes = [strikes_sorted[i] for i in indices]

        # Filter the calls DataFrame for these strikes
        selected_calls = self.option_contracts[self.option_contracts['strike'].isin(selected_strikes)]


        self.calls = selected_calls
        self.strikes = selected_calls['strike'].values
        self.last_price = selected_calls['lastPrice'].values
        self.last_price_avg = (selected_calls['bid'].values + selected_calls['ask'].values) / 2
        self.implied_volatilities = selected_calls['impliedVolatility'].values

        self.option_contracts = selected_calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]

        self.implied_volatility_solver()

    def BS_call_price(self, K, vol):
        """
        Calculate the Black-Scholes call option price.

        """

        K = np.array(K)
        vol = np.array(vol)

        d1 = (np.log(self.stock_price/K) + (self.risk_free_rate + 0.5*vol**2)*self.time_to_maturity) / (vol*np.sqrt(self.time_to_maturity))
        d2 = d1 - vol*np.sqrt(self.time_to_maturity)

        return self.stock_price*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*self.time_to_maturity)*norm.cdf(d2)


    def implied_volatility_solver(self):
        S=self.stock_price
        C=self.last_price
        T=self.time_to_maturity
        r=self.risk_free_rate

        implied_volatilities = []
        call_prices = []

        for (K, C) in zip(self.strikes, self.last_price_avg):
            interval = [0.000001, 1]
            f_start = self.BS_call_price(K, interval[0]) - C #function to minimize
            f_end = self.BS_call_price(K, interval[1]) - C #function to minimize
            if (f_start * f_end < 0):
                vol = (interval[0] + interval[1]) / 2
                f_mid = self.BS_call_price(K, vol) - C

                while (abs(f_mid) > 0.005):
                    vol = (interval[0] + interval[1]) / 2
                    f_mid = self.BS_call_price(K, vol) - C 
                    if (f_mid < 0):
                        interval[0] = vol
                    elif (f_mid > 0):
                        interval[1] = vol

                implied_volatilities.append(vol)
                call_prices.append(self.BS_call_price(K, vol))
            else:
                print("Interval does not contain the root for strike: ", K)
                implied_volatilities.append(np.nan)
                call_prices.append(np.nan)

        self.calculated_IV = np.array(implied_volatilities)
        self.calculated_C = np.array(call_prices)


    
    def view_option_details_manual(self):
        print("Option Details Overview")
        print("=======================")
        print(f"Stock Price: {self.stock_price:.2f}")
        print(f"Strikes: {self.strikes}")
        print(f"Volatility: {self.volatility:.2f}")
        print(f"Time to Maturity: {self.time_to_maturity:.2f} years")


    def view_option_details(self):
        print("Option Details Overview")
        print("=======================")
        print(f"Ticker symbol: {self.ticker.info['symbol']}")
        print(f"Current Stock Price: {self.stock_price:.2f}")
        if (self.expiry is None):
            print("Option Chain: Not Set")
            print(" ")
            print("Available Option Maturity Dates:")
            print("=======================")
            # Display up to 5 expiration dates per row, with correctly increasing numbering in headers
            total_expirations = len(self.option_dates_dict)
            # Get the expiration dates from the dictionary in order of their keys
            expirations = [self.option_dates_dict[i+1] for i in range(total_expirations)]
            rows = [expirations[i:i+5] for i in range(0, total_expirations, 5)]
            expiration_counter = 1
            for row in rows:
                num_columns = len(row)
                headers = [f"Maturity {expiration_counter + j}" for j in range(num_columns)]
                print(tabulate([row], headers=headers, tablefmt="pretty"))
                expiration_counter += num_columns
            print("=======================")
            print("Please set the option chain with Option.set_option_chain(N) - where N is the number of the available maturity date")
            print(" ")
            return
        else:
            print(f"Option Maturity Date: {self.expiry}, Time to Maturity: {365*self.time_to_maturity} days ({self.time_to_maturity:.2f} years)")

        if (self.option_contracts is None):
            print("No contracts set")
            print("Please choose a contract with set_contracts(N) - where N is the number of contracts to select")
            print(" ")
            print("Available Contracts:")
            print("=======================")
            print(" ")
            print(tabulate(self.option_contracts, headers='keys', tablefmt='github', showindex=False))
            print(" ")
            print("=======================")
            print(" ")
        else:
            print("=======================")
            print("Contracts:")
            print(" ")
            print(tabulate(self.option_contracts, headers='keys', tablefmt='github', showindex=False))
            print(" ")
            print("=======================")
            print(" ")

        # Combined table of (Strike, last_price, last_price_avg, calculated call price, calculated iv, market iv, Difference)
        if (
            self.calculated_IV is not None
            and self.calculated_C is not None
        ):
            # Prepare the data
            strikes = self.strikes
            
            # Use lastPrice as Yahoo's "last price", and bid/ask for mid
            last_price = self.last_price
            last_price_avg = self.last_price_avg

            calculated_call_price = self.calculated_C
            calculated_iv = self.calculated_IV
            market_iv = self.implied_volatilities



            
            # Calculate difference (calculated IV - market IV)
            difference = []
            for ci, mi in zip(calculated_iv, market_iv):
                try:
                    diff = None
                    if ci is not None and mi is not None:
                        diff = ci - mi
                    difference.append(diff)
                except Exception:
                    difference.append(None)
    


            # Assemble as table rows
            rows = []
            for i in range(len(strikes)):
                row = [
                    strikes[i] if i < len(strikes) else None,
                    last_price[i] if i < len(last_price) else None,
                    last_price_avg[i] if i < len(last_price_avg) else None,
                    calculated_call_price[i] if i < len(calculated_call_price) else None,
                    calculated_iv[i] if i < len(calculated_iv) else None,
                    market_iv[i] if i < len(market_iv) else None,
                    difference[i] if i < len(difference) else None,
                ]
                rows.append(row)

            headers = [
                "Strike",
                "last_price",
                "last_price_avg",
                "calculated call price",
                "calculated iv",
                "market iv",
                "Difference (Calc IV - Mkt IV)",
            ]
            print("Combined Overview (Strike, Last Price, Avg, Calculated Call, IVs):")
            print("=======================")
            print(tabulate(rows, headers=headers, tablefmt="github", showindex=False))
            print("=======================")





    

