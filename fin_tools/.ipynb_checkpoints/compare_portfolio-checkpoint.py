import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from tabulate import tabulate
import yfinance as yf
import datetime

def compare_portfolios(portfolios,N=20):
    """
    Compare multiple portfolios on the efficient frontier.
    
    - portfolios is a list of portfolios to compare.
    - N is the number of portfolios to plot on the efficient frontier.
    """
    plt.title('Comparison of Portfolios')
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$\mu$')

    colors = ['blue','red','green','purple','orange','yellow','brown','pink','gray','black']

    mu_vectors = []
    vol_vectors = []

    for portfolio in portfolios:
        mu_vectors.append(portfolio.expected_return_vector)
        vol_vectors.append(portfolio.volatility_vector)


    min_mu = min([np.min(mu_vec) for mu_vec in mu_vectors])
    max_mu = max([np.max(mu_vec) for mu_vec in mu_vectors])

    mu_range = np.linspace(min_mu, max_mu, N)


    for i, (portfolio, mu) in enumerate(zip(portfolios, mu_vectors)):
  
        e = np.ones(len(portfolio.cov_matrix))
        Vi = portfolio.inv_cov_matrix
        A = e.T @ Vi @ e
        B = e.T @ Vi @ mu
        C = mu.T @ Vi @ mu
        vol_range = A * ( mu_range - B/A) **2 /(A*C - B*B) + 1/A

        plt.plot(np.sqrt(vol_range), mu_range, label=f'Portfolio: {", ".join(portfolio.stock_names)}', color=colors[i])
        plt.scatter(np.sqrt(1/A), B/A, color=colors[i], label=r'$\mu_{min}$')

        # Scatter individual asset points for the current portfolio
        for j, (vol, ret) in enumerate(zip(vol_vectors[i], mu_vectors[i])):
            if i == 0:
                plt.scatter(vol, ret, facecolors=colors[i], edgecolors="none", s=90)
            else:
                plt.scatter(vol, ret, facecolors="none", edgecolors=colors[i], s=(i+1)*90)
            
            plt.text(vol, ret, portfolios[i].stock_names[j], fontsize=9, ha='right', va='bottom')

        


    plt.legend(loc='upper left')
    plt.show()



