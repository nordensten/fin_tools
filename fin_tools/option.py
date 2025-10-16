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
    def __init__(self, help_text=False):

        if (help_text):
            print("Option class initialized.")
            print("------------------------------")
            print("Add option details manually with:")
            print("-" + " Option.add_option(S0, K, volatility, T, r) " )
            print("   ")
            print("   " + "S0    - current stock price:  type float")
            print("   " + "K     - strike prices:        type list/array")
            print("   " + "sigma - volatility:           type float")
            print("   " + "T     - exercise time:        type float")
            print("   " + "r     - risk-free rate:       type float")
            print(" ")
            print("Or use a ticker symbol with:")
            print("-" + " Option.add_ticker(ticker_symbol) " )
            print("   ")
            print("   " + "ticker_symbol: type string" )
            print("------------------------------")

        # common variables
        self.stock_price = None
        self.strikes = None
        self.volatility = None
        self.maturity_cdays = None
        self.maturity_tdays = None
        self.risk_free_rate = None

        # ticker variables
        self.ticker = None
        self.option_dates = None
        self.option_dates_dict = {}
        self.opt_chain = None
                    
        # option variables
        self.expiry = None
        self.calls = None
        self.strikes = None
        self.last_price = None
        self.bid_ask_avg = None
        self.market_iv = None
        self.option_contracts = None

        # calculated variables
        self.estimated_iv = None
        self.call_prices = None


    def set_risk_free_rate(self, risk_free_rate):
        self.risk_free_rate = risk_free_rate
        
    def add_option(self, stock_price, strikes, volatility, time_to_maturity, risk_free_rate):
        self.stock_price = stock_price
        self.strikes = strikes
        self.volatility = volatility
        self.maturity_cdays = time_to_maturity
        self.maturity_tdays = time_to_maturity
        self.risk_free_rate = risk_free_rate

    def add_ticker(self, ticker_symbol):
        self.ticker = yf.Ticker(ticker_symbol)
        self.stock_price = self.ticker.history(period="1d")['Close'].iloc[-1]
        self.option_dates = self.ticker.options
        self.option_dates_dict = {}
        sorted_dates = sorted(self.option_dates)
        for i, date in enumerate(sorted_dates, 1):
                self.option_dates_dict[i] = date
            
    def set_option_chain(self, expiry):
        self.expiry = self.option_dates_dict[expiry]

        # calculate the time to maturity
        today = datetime.datetime.now().date()
        expiry_date = datetime.datetime.strptime(self.expiry, "%Y-%m-%d").date()
        self.maturity_cdays = (expiry_date - today).days / 365
        self.maturity_tdays = (expiry_date - today).days / 252

        # get the option chain
        self.opt_chain = self.ticker.option_chain(self.expiry)
        # get the calls
        self.calls = self.opt_chain.calls
        # get the strikes
        self.strikes = self.calls['strike'].values
        # get the option prices
        self.last_price = self.calls['lastPrice'].values
        # get the option prices average
        self.bid_ask_avg = (self.calls['bid'].values + self.calls['ask'].values) / 2
        # get the implied volatilities
        self.implied_volatilities = self.calls['impliedVolatility'].values
        # get the option contracts
        self.option_contracts = self.calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]

    

    def set_closest_strikes(self,N):
        strikes_sorted = self.option_contracts['strike'].sort_values().values
        selected_strikes = []

        # Find the index of the strike closest to S0
        closest_idx = (abs(strikes_sorted - self.stock_price)).argmin()

        # Find strikes: at N below, N above, and 1 closest to S0
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
        self.bid_ask_avg = (selected_calls['bid'].values + selected_calls['ask'].values) / 2
        self.implied_volatilities = selected_calls['impliedVolatility'].values

        self.option_contracts = selected_calls[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]

        self.implied_volatility_solver()

    def BS_call_price(self, K, vol):
        """
        Calculate the Black-Scholes call option price.

        """
        # use maturity in trading days or calendar days
        #T = self.maturity_tdays    
        T = self.maturity_cdays

        K = np.array(K)
        vol = np.array(vol)

        d1 = (np.log(self.stock_price/K) + (self.risk_free_rate + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)

        return self.stock_price*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)

    def plot_implied_volatility(self):
        fig, ax = plt.subplots(1,2, constrained_layout=True)
        ax[0].plot(self.strikes, self.implied_volatilities, marker='o', color='black', label='Market IV')
        ax[0].plot(self.strikes, self.estimated_iv, marker='o', color='darkgrey', label='Estimated IV')
        ax[0].set_title('Implied Volatility: ' + self.ticker.info['symbol'])
        ax[0].set_xlabel('K')
        ax[0].set_ylabel('IV')
        ax[0].set_xticks(self.strikes)
        ax[0].set_xticklabels([str(k) for k in self.strikes])
        ax[0].set_yticks(np.sort(np.concatenate((self.implied_volatilities, self.estimated_iv))))
        ax[0].set_yticklabels([f'{iv:.1%}' for iv in np.sort(np.concatenate((self.implied_volatilities, self.estimated_iv)))])

        ax[1].plot(self.strikes, (self.implied_volatilities - self.estimated_iv) / self.estimated_iv, marker='o', color='darkred')
        ax[1].set_title('Relative Difference of IVs')
        ax[1].set_xlabel('K')
        ax[1].set_ylabel('rel. difference')
        ax[1].set_xticks(self.strikes)
        ax[1].set_xticklabels([str(k) for k in self.strikes])
        ax[1].set_yticks((self.implied_volatilities - self.estimated_iv) / self.estimated_iv)
        ax[1].set_yticklabels([f'{iv:.2%}' for iv in (self.implied_volatilities - self.estimated_iv) / self.estimated_iv])

        ax[0].legend()
        plt.show()


    def implied_volatility_solver(self):

        implied_volatilities = []
        call_prices = []

        for (K, C) in zip(self.strikes, self.bid_ask_avg):
            interval = [0.000001, 200]
            f_start = self.BS_call_price(K, interval[0]) - C #function to minimize
            f_end = self.BS_call_price(K, interval[1]) - C #function to minimize
            if (f_start * f_end < 0):
                vol = (interval[0] + interval[1]) / 2
                f_mid = self.BS_call_price(K, vol) - C

                while (abs(f_mid) > 5e-4):
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

        self.estimated_iv = np.array(implied_volatilities)
        self.call_prices = np.array(call_prices)



    def view_option_details(self):
        if (self.ticker is None):
            print("Option Details Overview")
            print("=======================")
            print(f"Stock Price: {self.stock_price:.2f}")
            print(f"Strikes: {self.strikes}")
            print(f"Volatility: {self.volatility:.2f}")
            print(f"Time to Maturity: {self.maturity_cdays:.2f} years")
            return
        else:
            print("Option Details Overview")
            print("=======================")
            print(f"Ticker symbol: {self.ticker.info['symbol']}")
            print(f"Current Stock Price: {self.stock_price:.2f}")
            if (self.expiry is None):
                print("Option Chain: Not Set")
                print(" ")
                print("   - " + "Set the option chain with Option.set_option_chain(N) - where N is the number of the chosen exercise time from the list below" )
                print(" ")
                print("Available Option Exercise Times:")
                print("  ")
                # Display up to 5 expiration dates per row, with correctly increasing numbering in headers
                total_expirations = len(self.option_dates_dict)
                # Get the expiration dates from the dictionary in order of their keys
                expirations = [self.option_dates_dict[i+1] for i in range(total_expirations)]
                rows = [expirations[i:i+5] for i in range(0, total_expirations, 5)]
                expiration_counter = 1
                for row in rows:
                    num_columns = len(row)
                    headers = [f"Exercise Time {expiration_counter + j}" for j in range(num_columns)]
                    print(tabulate([row], headers=headers, tablefmt="pretty"))
                    expiration_counter += num_columns
                print(" ")
                print(" ")
                return
            else:
                print(f"Option Maturity Date: {self.expiry}, Time to Maturity: {365*self.maturity_cdays} days ({self.maturity_cdays:.2f} years)")

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

            # Combined table of (Strike, last_price, last_price_avg, calculated call price, calculated iv, market iv, Difference)
            if (
                self.estimated_iv is not None
                and self.call_prices is not None
            ):
                # Prepare the data
                strikes = self.strikes

                # Use lastPrice as Yahoo's "last price", and bid/ask for mid
                last_price = self.last_price
                last_price_avg = self.bid_ask_avg

                calculated_call_price = self.call_prices
                calculated_iv = self.estimated_iv
                market_IV = self.implied_volatilities




                # Calculate difference (calculated IV - market IV)
                difference = []
                for ci, mi in zip(calculated_iv, market_IV):
                    try:
                        diff = None
                        if ci is not None and mi is not None:
                            diff = 100*(mi - ci) / ci
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
                        market_IV[i] if i < len(market_IV) else None,
                        difference[i] if i < len(difference) else None,
                    ]
                    rows.append(row)

                headers = [
                    "Strike",
                    "Last C",
                    "C (Bid/Ask Avg)",
                    "C (B&S)",
                    "IV (B&S)",
                    "IV (Market)",
                    "rel. error. IV [%]",
                ]
                print("Calculated Overview")
                print(" ")
                print(tabulate(rows, headers=headers, tablefmt="github", showindex=False))
                print(" ")
                print("=======================")
