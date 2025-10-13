import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tabulate import tabulate
import yfinance as yf
import datetime

def compare_portfolios(portfolio1, portfolio2):
    """
    Compare two portfolios on the efficient frontier.
    """
    colors = ['black','darkblue']

    # Plot the efficient frontier
    fig, ax = plt.subplots(figsize=(12, 8))


    mu1 = portfolio1.expected_return_vector
    mu2 = portfolio2.expected_return_vector

    vol1 = portfolio1.volatility_vector
    vol2 = portfolio2.volatility_vector

    mu_range = np.linspace(min(np.min(mu1), np.min(mu2)), max(np.max(mu1), np.max(mu2)), 20)

    for i, (portfolio, mu) in enumerate([(portfolio1, mu1), (portfolio2, mu2)]):   
        e = np.ones(len(portfolio.cov_matrix))
        Vi = portfolio.inv_cov_matrix
        A = e.T @ Vi @ e
        B = e.T @ Vi @ mu
        C = mu.T @ Vi @ mu
        var_range = A * ( mu_range - B/A) **2 /(A*C - B*B) + 1/A
        ax.plot(np.sqrt(var_range), mu_range, label=f'Portfolio: {portfolio.stock_names}', color=colors[i])
        ax.scatter(np.sqrt(1/A), B/A, color=colors[i], label=r'$\mu_{min}$')


    for j, (vol, ret) in enumerate(zip(vol1, mu1)):
            ax.scatter(vol, ret, facecolors=colors[0], edgecolors='none')
            ax.text(vol, ret, portfolio1.stock_names[j], fontsize=9, ha='right', va='bottom')

    for j, (vol, ret) in enumerate(zip(vol2, mu2)):
            ax.scatter(vol, ret, facecolors='none', edgecolors=colors[1], s=100.0)
            ax.text(vol, ret, portfolio2.stock_names[j], fontsize=9, ha='right', va='bottom')

    ax.set_title('Comparison of Portfolios')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\mu$')
    ax.legend(loc='upper left')
    plt.show()