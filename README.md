# fin_tools

A small python module developed for an obligatory task in STK-MAT4700 - Introduction to Mathematical Finance.

## Overview

`fin_tools` is a Python package designed to support basic quantitative finance and portfolio analysis. It provides classes and functions for stock analysis, option data handling, portfolio construction, and portfolio comparison, with a focus on educational use.

## Main Components

- **Stock** (`fin_tools.stock.Stock`):  
  Analyze stock price series, compute returns, volatility, expected return, and model return distributions.

- **Option** (`fin_tools.option.Option`):  
  Retrieve and manage option data, including support for Yahoo Finance tickers, calculation of option chain data, and extraction of relevant pricing and volatility statistics.

- **Portfolio** (`fin_tools.portfolio.Portfolio`):  
  Maintain a collection of stocks, compute portfolio statistics, manage assets, and simulate the efficient frontier.

- **compare_portfolios** (`fin_tools.compare_portfolio.compare_portfolios`):  
  Visualize and compare two portfolios on the efficient frontier.

## Installation

Clone this repository and install the package locally:

```bash
git clone https://github.com/nordensten/fin_tools.git
cd fin_tools
pip install .
```

## Usage Example

```python
from fin_tools import Stock, Portfolio, compare_portfolios

# Create Stock objects
prices = [100, 102, 101, 105, 107]
stock = Stock(prices, "ExampleCorp")

# Create a Portfolio and add the stock
portfolio = Portfolio()
portfolio.add_stock(stock)

# Portfolio analysis
print("Expected return:", portfolio.expected_return_vector)
print("Volatility:", portfolio.volatility_vector)
```

To compare portfolios:
```python
from fin_tools import compare_portfolios
compare_portfolios(portfolio1, portfolio2)
```

## File Structure

- `fin_tools/stock.py`: Stock time series analysis and modeling.
- `fin_tools/option.py`: Option data retrieval (with Yahoo Finance) and manipulation.
- `fin_tools/portfolio.py`: Portfolio construction, analysis, and efficient frontier simulation.
- `fin_tools/compare_portfolio.py`: Portfolio comparison and visualization.
- `fin_tools/__init__.py`: Package imports and API exposure.

*Note: The majority of the codebase is in Python, but there may also be Jupyter Notebooks in the repository for demonstrations or assignments.*

## Requirements

- numpy
- pandas
- matplotlib
- scipy
- tabulate
- yfinance

Install requirements with:
```bash
pip install numpy pandas matplotlib scipy tabulate yfinance
```

## License

This project is for educational purposes. See the `LICENSE` file for details if present.

---

*Developed for STK-MAT4700 - Introduction to Mathematical Finance.*

---


