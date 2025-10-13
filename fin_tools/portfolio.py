import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tabulate import tabulate
import yfinance as yf
import datetime


class Portfolio:
    """Class to represent and analyze a portfolio of stocks."""
    
    def __init__(self):
        """
        Initialize a Portfolio object.
        """
        self.stocks = []
        self.n_stocks = 0
        self.stock_names = []
    
        #initialize the stock matrix
        self.stock_matrix = None
        self.returns_matrix = None
        self.volatility_vector = None
        self.expected_return_vector = None
        self.expected_return_fit_vector = None
        self.volatility_fit_vector = None
        self.stock_dist_vector = None


        self.cov_matrix = None
        self.corr_matrix = None
        self.inv_cov_matrix = None
        self.inv_corr_matrix = None


    def add_stock(self, stock):
        self.stocks.append(stock)
        self.n_stocks += 1
        self.stock_names.append(stock.get_stock_name())

    def add_stocks(self, stocks):
        for i in range(len(stocks)):
            self.add_stock(stocks[i])

        self.initialize_portfolio()


    def remove_stock(self, stock_name):
        for stock in self.stocks:
            if stock.get_stock_name() == stock_name:
                self.stocks.remove(stock)
                self.stock_names.remove(stock.get_stock_name())
        self.n_stocks -= 1
        self.initialize_portfolio()

    def copy(self):
        new_portfolio = self.__class__()  # Allow inheritence: new object is of the same class as self
        new_portfolio.add_stocks(self.stocks)   # Add the stocks to the new portfolio
        return new_portfolio


    def initialize_portfolio(self):
        self.stock_matrix = np.array([stock.get_stock_price() for stock in self.stocks])
        self.returns_matrix = np.array([stock.get_returns() for stock in self.stocks])
        self.volatility_vector = np.array([stock.get_volatility() for stock in self.stocks])
        self.expected_return_vector = np.array([stock.get_expected_return() for stock in self.stocks])
        self.expected_return_fit_vector = np.array([stock.get_expected_return_fit() for stock in self.stocks])
        self.volatility_fit_vector = np.array([stock.get_volatility_fit() for stock in self.stocks])
        self.stock_dist_vector = np.array([stock.get_stock_dist() for stock in self.stocks])
        self.cov_matrix = np.cov(self.returns_matrix)
        self.corr_matrix = np.corrcoef(self.returns_matrix)
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.inv_corr_matrix = np.linalg.inv(self.corr_matrix)

    def efficient_frontier(self,plot_color='black', dot_color='red', asset_color='black'):
        plt.title('Efficient Frontier')
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$\mu$')

        # initialize the mu range for the portfolios on the efficient frontier
        mu = self.expected_return_vector
        mu_range = np.linspace(np.min(mu), np.max(mu), 20)
        # Retrieve and initialize the e-vector and the inverse covariance matrix
        e = np.ones(len(self.cov_matrix))
        Vi = self.inv_cov_matrix

        # Calculate the coefficients 
        A = e.T @ Vi @ e
        B = e.T @ Vi @ mu
        C = mu.T @ Vi @ mu

        # Calculate the variance range for the portfolios on the efficient frontier

        var_range = A * ( mu_range - B/A) **2 /(A*C - B*B) + 1/A

        # Calculate the minimum variance portfolio coordinates
        min_var_portfolio = np.sqrt(1/A)
        min_var_return = B/A

        # Plot the efficient frontier
        plt.plot(np.sqrt(var_range), mu_range, label=f'EF Assets: {self.stock_names}', color=plot_color)

        # Plot (σ, µ) coordinates of each stock
        for vol, ret, name in zip(self.volatility_vector, self.expected_return_vector, self.stock_names):
            plt.scatter(vol, ret, color=dot_color)
            plt.text(vol, ret, name, fontsize=9, ha='right', va='bottom')

        # Plot the coordinates of the minimum variance portfolio
        plt.scatter(min_var_portfolio, min_var_return, color=dot_color)
        plt.text(min_var_portfolio, min_var_return, r'$\mu_{min}$', fontsize=9, ha='right', va='bottom')
    
        plt.legend()

        




    def fit_portfolio(self):
        n = len(self.stocks)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
        if n == 1:
            axes = [axes]  # handle 1-stock case

        for i, (ax, stock) in enumerate(zip(axes, self.stocks)):
            expected_return_fit = stock.get_expected_return_fit()
            volatility_fit = stock.get_volatility_fit()
            dist = stock.get_stock_dist()
            # Generate a range based on the fitted values (for better alignment)
            x = np.linspace(expected_return_fit - 4*volatility_fit, expected_return_fit + 4*volatility_fit, 1000)
            p = stats.norm.pdf(x, expected_return_fit, volatility_fit)
            ax.plot(x, p, label=f'Fitted', color='red')
            ax.hist(dist, bins=10, density=True, alpha=0.6, color='b', label='Empirical')
            ax.set_title(f'Stock {i+1}: {stock.get_stock_name()}')
            ax.set_xlabel('Returns')
            ax.set_ylabel('Density')
            ax.legend()

    def show_corr_and_cov_matrix(self):
        """
        Show the correlation and covariance matrix.
        Plot both as side by side heatmaps, with stock names for each row/col.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        stock_names = self.stock_names if hasattr(self, 'stock_names') else [f"Stock {i+1}" for i in range(len(self.corr_matrix))]
        im1 = axes[0].imshow(self.corr_matrix, cmap='coolwarm', interpolation='none', vmin=np.min(self.corr_matrix), vmax=np.max(self.corr_matrix))
        axes[0].set_title('Correlation of Assets: ' + ', '.join(stock_names))
        axes[0].set_xticks(np.arange(len(stock_names)))
        axes[0].set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticks(np.arange(len(stock_names)))
        axes[0].set_yticklabels(stock_names, fontsize=8)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        im2 = axes[1].imshow(self.cov_matrix, cmap='coolwarm', interpolation='none', vmin=np.min(self.cov_matrix), vmax=np.max(self.cov_matrix))
        axes[1].set_title('Covariance of Assets: ' + ', '.join(stock_names))
        axes[1].set_xticks(np.arange(len(stock_names)))
        axes[1].set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticks(np.arange(len(stock_names)))
        axes[1].set_yticklabels(stock_names, fontsize=8)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        plt.show()


  

    def view_portfolio(self, plot=False):
        """
        Display a nicely formatted table with detailed information for each stock in the portfolio.
        """
        if self.n_stocks == 0:
            print("Portfolio is empty.")
            return
        
        # Create a DataFrame to store stock info
        table_data = {
            "Stock Name": [],
            "Initial Price": [],
            "Current Price": [],
            "Expected Return": [],
            "Volatility": [],
            "Fitted Expected Return": [],
            "Fitted Volatility": []
        }

        for stock in self.stocks:
            prices = stock.get_stock_price()
            table_data["Stock Name"].append(stock.get_stock_name())
            table_data["Initial Price"].append(f"{prices[0]:.2f}")
            table_data["Current Price"].append(f"{prices[-1]:.2f}")
            table_data["Expected Return"].append(f"{stock.get_expected_return():.4f}")
            table_data["Volatility"].append(f"{stock.get_volatility():.4f}")

            table_data["Fitted Expected Return"].append(f"{stock.get_expected_return_fit():.4f}")
            table_data["Fitted Volatility"].append(f"{stock.get_volatility_fit():.4f}")

        df = pd.DataFrame(table_data)
        print("\nPortfolio Composition and Statistics\n")

        print(tabulate(df, headers='keys', tablefmt='github', showindex=False))

        if plot:
            self.fit_portfolio()