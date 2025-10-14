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

        # efficient frontier variables
        self.ef_mu = None
        self.ef_vol = None
        self.ef_mu_min = None
        self.ef_vol_min = None


    def add_stock(self, stock):
        """
        Add a stock to the portfolio.
        """
        self.stocks.append(stock)
        self.n_stocks += 1
        self.stock_names.append(stock.get_stock_name())

    def add_stocks(self, stocks):
        """
        Add a list of stocks to the portfolio.
        """
        for i in range(len(stocks)):
            self.add_stock(stocks[i])

        self.initialize_portfolio()


    def remove_stock(self, stock_name):
        """
        Remove a stock from the portfolio.
        """
        for stock in self.stocks:
            if stock.get_stock_name() == stock_name:
                self.stocks.remove(stock)
                self.stock_names.remove(stock.get_stock_name())
        self.n_stocks -= 1
        self.initialize_portfolio()

    def copy(self):
        """
        Copy the portfolio.
        """
        pfcopy = self.__class__()  
        pfcopy.add_stocks(self.stocks)   
        return pfcopy


    def initialize_portfolio(self):
        """
        Initialize the portfolio variables.
        """
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
        
        self.ef_mu, self.ef_vol, self.ef_mu_min, self.ef_vol_min = self.efficient_frontier()



    def efficient_frontier(self):
        """
        Return the efficient frontier and the minimum variance portfolio.
        """
        mu = self.expected_return_vector

        mu_ = np.linspace(np.min(mu), np.max(mu), 20)

        # Retrieve and initialize the e-vector and the inverse covariance matrix
        e = np.ones(len(self.cov_matrix))
        Vi = self.inv_cov_matrix

        # Calculate the coefficients 
        A = e.T @ Vi @ e
        B = e.T @ Vi @ mu
        C = mu.T @ Vi @ mu

        # Calculate the variance range for the portfolios on the efficient frontier

        vol_ = np.sqrt(A * ( mu_ - B/A) **2 /(A*C - B*B) + 1/A)

        # Calculate the minimum variance portfolio coordinates
        vol_min = np.sqrt(1/A)
        mu_min = B/A

        return mu_, vol_, mu_min, vol_min


    def plot_efficient_frontier(self):
        """
        Plot the efficient frontier and the minimum variance portfolio.
        """
        plt.plot(self.ef_vol, self.ef_mu, color='blue')
        plt.scatter(self.ef_vol_min, self.ef_mu_min, color='red', label=r'$(\sigma_{min}, \mu_{min})$')
    

        for i, (vol, ret) in enumerate(zip(self.volatility_vector, self.expected_return_vector)):
            plt.scatter(vol, ret, color='blue')
            plt.text(1.008*vol, ret, self.stock_names[i], fontsize=9, ha='left', va='center')

        plt.title('Efficient Frontier: ' + ', '.join(self.stock_names))
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$\mu$')
        plt.legend(loc='upper left')
        plt.show()

    def plot_portfolio_fit(self):
        """
        Plot the portfolio fitted distribution and the empirical distribution.
        """
        n = len(self.stocks)
        n_cols = min(2, n)
        n_rows = (n + n_cols - 1) // n_cols  # ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, constrained_layout=True)
        axes = np.atleast_1d(axes)  # Ensure axes is always an array

        # Flatten axes for easy iteration, if axes is 2D
        axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]

        for i, (stock, ax) in enumerate(zip(self.stocks, axes_flat)):
            # get the fitted values and the empirical distribution
            mu_fit = stock.get_expected_return_fit()
            sigma_fit = stock.get_volatility_fit()
            dist = stock.get_stock_dist()

            # Generate a range based on the fitted values (for better alignment)
            x = np.linspace(mu_fit - 4*sigma_fit, mu_fit + 4*sigma_fit, 1000)
            p = stats.norm.pdf(x, mu_fit, sigma_fit)

            # plot the fitted distribution and the empirical distribution
            if i == 0:
                ax.plot(x, p, color='red', label='Fit')
                ax.hist(dist, bins=10, density=True, alpha=0.6, color='b', edgecolor='black', label='Empirical')
                ax.legend(loc='upper left')
                ax.set_xlabel(r'$\mu$')
                ax.set_ylabel(r'Density')
            else:
                ax.plot(x, p, color='red')
                ax.hist(dist, bins=10, density=True, alpha=0.6, color='b', edgecolor='black')
            ax.set_title(f'Stock: {stock.get_stock_name()}')
        # Hide unused axes if any
        for j in range(i+1, n_rows * n_cols):
            axes_flat[j].set_visible(False)
        plt.show()

    def plot_corr_and_cov_matrix(self):
        """
        Plot the correlation and covariance matrix.
        Plot both as side by side heatmaps, with stock names for each row/col.
        Show numbers for each element in the matrices.
        """
        fig, axes = plt.subplots(1, 2, constrained_layout=True)

        stock_names = self.stock_names 
        # Correlation matrix plot
        im1 = axes[0].imshow(self.corr_matrix, cmap='coolwarm', interpolation='none', vmin=np.min(self.corr_matrix), vmax=np.max(self.corr_matrix))
        axes[0].set_title('Correlation Matrix')
        axes[0].set_xticks(np.arange(len(stock_names)))
        axes[0].set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticks(np.arange(len(stock_names)))
        axes[0].set_yticklabels(stock_names, fontsize=8)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        # Add numbers to each cell
        for i in range(self.corr_matrix.shape[0]):
            for j in range(self.corr_matrix.shape[1]):
                axes[0].text(
                    j, i, f"{self.corr_matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if abs(self.corr_matrix[i, j]) < 0.7 else "white",
                    fontsize=9
                )

        # Covariance matrix plot
        im2 = axes[1].imshow(self.cov_matrix, cmap='coolwarm', interpolation='none', vmin=np.min(self.cov_matrix), vmax=np.max(self.cov_matrix))
        axes[1].set_title('Covariance Matrix')
        axes[1].set_xticks(np.arange(len(stock_names)))
        axes[1].set_xticklabels(stock_names, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticks(np.arange(len(stock_names)))
        axes[1].set_yticklabels(stock_names, fontsize=8)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        # Add numbers to each cell
        for i in range(self.cov_matrix.shape[0]):
            for j in range(self.cov_matrix.shape[1]):
                axes[1].text(
                    j, i, f"{self.cov_matrix[i, j]:.2e}",
                    ha="center", va="center",
                    color="black" if abs(self.cov_matrix[i, j]) < np.max(np.abs(self.cov_matrix)) / 2 else "white",
                    fontsize=9
                )

        plt.show()


  

    def view_portfolio(self):
        """
        Display a nicely formatted table with detailed information for each stock in the portfolio.
        """
        if self.n_stocks == 0:
            print("Portfolio is empty.")
            return
        
        # Create a DataFrame to store stock info
        table_data = {
            "Stock": [],
            "S_0": [],
            "S_T": [],
            "mu": [],
            "sigma": [],
            "mu_fit": [],
            "sigma_fit": []
        }

        for stock in self.stocks:
            prices = stock.get_stock_price()
            table_data["Stock"].append(stock.get_stock_name())
            table_data["S_0"].append(f"{prices[0]:.2f}")
            table_data["S_T"].append(f"{prices[-1]:.2f}")
            table_data["mu"].append(f"{stock.get_expected_return():.4f}")
            table_data["sigma"].append(f"{stock.get_volatility():.4f}")

            table_data["mu_fit"].append(f"{stock.get_expected_return_fit():.4f}")
            table_data["sigma_fit"].append(f"{stock.get_volatility_fit():.4f}")

        df = pd.DataFrame(table_data)
        print("\nPortfolio Composition and Statistics\n")

        print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
