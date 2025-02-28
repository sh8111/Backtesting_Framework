import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import yfinance as yf

# Function to load market data
@st.cache_data
def load_market_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Build a Backtesting Engine
# Implement the backtesting engine that simulates trades and calculates performance
class BacktestingEngine:
    def __init__(self, data):
        self.data = data
        self.portfolio = {'cash': 100000 , 'positions': {}}
        self.trades = []

    def run_backtest(self, money):
        risk_per_trade = 1 # Risk 2% of portfolio per trade
        self.portfolio = {'cash': money , 'positions': {}}

        for index, row in self.data.iterrows():
            # Implement your trading logic for each data point
            close_price = row['Close']
            signal = row['signal'].item()

            if signal == 1:
                # Buy signal
                available_cash = self.portfolio['cash']
                position_size = available_cash * risk_per_trade / close_price
                self.execute_trade(symbol=self.data.name, quantity=position_size, price=close_price)
            elif signal == -1:
                position_size = float(self.portfolio['positions'].get(self.data.name, 0))  # ✅ Convert to float
                if position_size > 0:
                    self.execute_trade(symbol=self.data.name, quantity=-position_size, price=close_price)


    def calculate_performance(self,money):
        if len(self.trades) < 2:
            return 0.0, 0.0, 0.0

        # Ensure portfolio value correctly reflects total P&L
        total_pnl = self.get_portfolio_value(self.data.iloc[-1]['Close']) - money  # Subtract initial cash

        trade_prices = np.array([float(trade['price']) for trade in self.trades])
        trade_quantities = np.array([float(trade['quantity']) for trade in self.trades])

        if len(trade_prices) < 2:
            return total_pnl, 0.0, 0.0

        valid_indices = trade_prices[:-1] != 0
        trade_returns = np.zeros(len(trade_prices) - 1)
        trade_returns[valid_indices] = np.diff(trade_prices)[valid_indices] / trade_prices[:-1][valid_indices]

        average_trade_return = np.mean(trade_returns) if len(trade_returns) > 0 else 0.0
        win_ratio = np.sum(trade_returns > 0) / len(trade_returns) if len(trade_returns) > 0 else 0.0

        return total_pnl, average_trade_return, win_ratio



    def execute_trade(self, symbol, quantity, price):
        # Update portfolio and execute trade
        self.portfolio['cash'] -= quantity * price

        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol] += quantity
        else:
            self.portfolio['positions'][symbol] = quantity

        self.trades.append({'symbol': symbol, 'quantity': quantity, 'price': price})

    def get_portfolio_value(self, price):
        # Calculate the current value of the portfolio
        positions_value = sum(self.portfolio['positions'].get(symbol, 0) * price for symbol in self.portfolio['positions'])
        return self.portfolio['cash'] + positions_value

    def get_portfolio_returns(self):
        portfolio_value = np.array([self.get_portfolio_value(row['Close']) for _, row in self.data.iterrows()]).ravel()
        if len(portfolio_value) < 2:
            return np.array([])  # Prevent division errors for short data
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        return returns

    def print_portfolio_summary(self):
        print('--- Portfolio Summary ---')
        print('Cash:', self.portfolio['cash'])
        print('Positions:')
        for symbol, quantity in self.portfolio['positions'].items():
            print(symbol + ':', quantity)

    def plot_portfolio_value(self):
        portfolio_value = [self.get_portfolio_value(row['Close']) for _, row in self.data.iterrows()]
        dates = self.data.index
        signals = self.data['signal']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot portfolio value
        ax1.plot(dates, portfolio_value, label='Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')

        # Plot buy/sell signals
        ax2 = ax1.twinx()
        ax2.plot(dates, signals, 'r-', label='Buy/Sell Signal')
        ax2.set_ylabel('Signal')
        ax2.grid(None)

        fig.tight_layout()
        st.pyplot(fig)

# Streamlit UI
st.title('Backtesting Application')

st.markdown("""
### Welcome to the Backtesting Application 🚀
This tool allows you to test different **trading strategies** on historical market data. 
Simply choose a **stock ticker**, set your **date range**, and select a **strategy** from the sidebar.

### Available Strategies:
- **SMA Crossover**: Uses short-term and long-term moving averages to determine buy/sell signals.
- **Bollinger Bands**: Identifies overbought and oversold conditions.
- **RSI**: Uses the Relative Strength Index to capture market momentum.
- **MACD**: Employs trend-following momentum indicators for trade signals.
- **Mean Reversion**: Detects price deviations from the average to capitalize on reversions.

Hit **Run Backtest** to see how the strategy performs over time! 📈
""")

with st.sidebar:
    st.title('Developed by Sumanth Bharadwaj')
    st.text("Please Enter your inputs")

    st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/hksb4602)")

    symbol = st.text_input('Enter Ticker Symbol', 'AAPL')
    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2021-01-01'))
    money = st.number_input('Cash Flow', 100000)
    strategy = st.selectbox('Select Strategy', ['SMA Crossover', 'Bollinger Bands', 'RSI', 'MACD', 'Mean Reversion'])
    run_backtest = st.button('Run Backtest')

if run_backtest:
    data = load_market_data(symbol, start_date, end_date)
    if data.empty:
        st.error('No data found for the given ticker and date range.')
    else:
        if strategy == 'SMA Crossover':
            sma_short = ta.trend.sma_indicator(data['Close'][symbol], window=20).fillna(method='bfill')
            sma_long = ta.trend.sma_indicator(data['Close'][symbol], window=50).fillna(method='bfill')
            data['signal'] = np.where(sma_short > sma_long, 1, -1)
        elif strategy == 'Bollinger Bands':
            bollinger = ta.volatility.BollingerBands(data['Close'][symbol], window=1, window_dev=2)
            data['signal'] = np.where(data['Close'][symbol] < bollinger.bollinger_lband(), 1, np.where(data['Close'][symbol] > bollinger.bollinger_hband(), -1, 0))
        elif strategy == 'RSI':
            data['rsi'] = ta.momentum.RSIIndicator(data['Close'][symbol], window=14).rsi()
            data['signal'] = np.where(data['rsi'] < 30, 1, np.where(data['rsi'] > 70, -1, 0))
        elif strategy == 'MACD':
            macd = ta.trend.MACD(data['Close'][symbol], window_slow=26, window_fast=12, window_sign=9)
            data['signal'] = np.where(macd.macd() > macd.macd_signal(), 1, np.where(macd.macd() < macd.macd_signal(), -1, 0))
        elif strategy == 'Mean Reversion':
            data['mean'] = data['Close'][symbol].rolling(window=20).mean()
            data['signal'] = np.where(data['Close'][symbol] < data['mean'] * 0.95, 1, np.where(data['Close'][symbol] > data['mean'] * 1.05, -1, 0))
    data.name = symbol  # Set the name attribute for the data DataFrame

    engine = BacktestingEngine(data)
    initial_portfolio_value = engine.get_portfolio_value(data.iloc[0]['Close'])

    engine.run_backtest(money)


    # Step 8: Evaluate Performance
    final_portfolio_value = engine.get_portfolio_value(data.iloc[-1]['Close'])
    returns = engine.get_portfolio_returns()
    # Ensure total_returns is a float
    total_returns = float((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value)
    annualized_returns = (1 + total_returns) ** (252 / len(data)) - 1  # Assuming 252 trading days in a year
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (annualized_returns - 0.02) / volatility  # Assuming risk-free rate of 2%

    # Visualize Results
    engine.plot_portfolio_value()

    total_pnl, average_trade_return, win_ratio = engine.calculate_performance(money)
    st.subheader('Performance Metrics')
    st.write(f'Total Returns: {total_returns:.2%}')
    st.write(f'Annualized Returns: {annualized_returns:.2%}')
    st.write(f'Volatility: {volatility:.2%}')
    st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    st.write(f'Total P&L: {float(total_pnl): .2f}')
    st.write(f'Average Trade Return: {average_trade_return:.2%}')
    st.write(f'Win Ratio: {win_ratio:.2%}')
