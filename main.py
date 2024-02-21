import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from urllib3.exceptions import InsecureRequestWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_log_returns(data):
    data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    return data


def calculate_drift(data):
    return data['Log Returns'].mean()


def calculate_volatility(data):
    return data['Log Returns'].std()


def get_stock_data(ticker):
    data = yf.download(ticker, start='2000-01-01', end='2022-12-31')
    return data


def monte_carlo_simulation(data, drift, volatility, start_date, num_simulations=1000):
    data = data[data.index >= start_date]
    daily_returns = np.exp(drift + volatility * np.random.normal(0, 1, len(data.index)))
    price_paths = np.zeros_like(daily_returns)

    price_paths[0] = data['Close'].iloc[0]

    for t in range(1, len(data.index)):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    simulations_inner = np.zeros((num_simulations, len(data.index)))
    for i in range(num_simulations):
        daily_returns = np.exp(drift + volatility * np.random.normal(0, 1, len(data.index)))
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = data['Close'][0]
        for t in range(1, len(data.index)):
            price_paths[t] = price_paths[t - 1] * daily_returns[t]
        simulations_inner[i] = price_paths

    return simulations_inner


def filter_simulations_by_error(data, simulations, margin_of_error, start_date):

    data = data[data.index >= start_date]['Close'].values

    valid_simulations = []
    for simulation in simulations:
        error = np.abs(simulation - data)
        mean_error = np.mean(error)
        if mean_error <= margin_of_error:
            valid_simulations.append(simulation)

    return valid_simulations


def compare_simulation_with_real(data, ticker, simulations, start_date, margin_of_error):
    data = data[data.index >= start_date]['Close']

    for i, simulation in enumerate(simulations):
        error = np.abs(simulation - data.values)
        mean_error = np.mean(error)
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, simulation, label='Simulation')
        plt.plot(data.index, data.values, label='Real Data')  # Removed 'Close' key
        plt.title(f"""{ticker} - Real vs Simulation - {i + 1} (<= {margin_of_error}%)
        Margin of Error: {mean_error:.2f}%""")
        plt.legend()
        plt.show()


def simulate_filter_compare(ticker, margin_of_error, start_date):
    stock_data = get_stock_data(ticker)
    stock_data = calculate_log_returns(stock_data)

    drift = calculate_drift(stock_data)
    volatility = calculate_volatility(stock_data)

    print('Drift:', drift)
    print('Volatility:', volatility)

    simulations = monte_carlo_simulation(stock_data, drift, volatility, start_date)
    filtered_simulations = filter_simulations_by_error(stock_data, simulations, margin_of_error, start_date)
    print("Filtered Simulations: ", len(filtered_simulations))
    compare_simulation_with_real(stock_data, ticker, filtered_simulations, start_date, margin_of_error)


simulate_filter_compare('AMZN', 20, '2020-01-01')