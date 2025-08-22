# StockMarketPredictor

Python project that uses Support Vector Regression (SVR) to model and predict stock prices from historical data.

## Features
- Reads stock data (date + price) from a CSV file
- Fits three models:
  - Linear SVR
  - Polynomial SVR
  - RBF SVR
- Plots actual data and fitted models
- Allows prediction for a future date or offset (e.g., 10 days after last data point)

## Requirements
- Python 3.x
- NumPy
- scikit-learn
- matplotlib

## How to Run
1. Clone this repository or download the files.
2. Install the required libraries:
   ```bash
   pip install numpy scikit-learn matplotlib
3. Run using -> python StockPredictor.py

