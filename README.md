# 📊 Streamlit Backtesting Application

## 🚀 Overview
This **Backtesting Application** allows traders and analysts to test different trading strategies on historical market data. Built using **Streamlit**, it provides an interactive and visual way to analyze trading performance.

## 🔧 Features
- 📈 **Multiple Trading Strategies**: Choose from SMA Crossover, Bollinger Bands, RSI, MACD, or Mean Reversion.
- 📊 **Real-time Performance Metrics**: View Total P&L, Average Trade Return, and Win Ratio.
- 📅 **Customizable Backtesting**: Select any stock ticker and date range for analysis.
- 📉 **Portfolio Value Visualization**: See how your portfolio changes over time.
- 🖥 **User-Friendly Interface**: Powered by Streamlit for easy interaction.

## 📌 How to Run the Application
### 1️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed. Then, install the required packages:
```sh
pip install streamlit pandas numpy matplotlib ta yfinance
```

### 2️⃣ Run the Application
```sh
streamlit run backtesting_app.py
```

## 📚 How It Works
1. **Enter a Stock Ticker** (e.g., `AAPL`).
2. **Select a Date Range** for backtesting.
3. **Choose a Trading Strategy**:
   - **SMA Crossover**: Uses moving average crossovers to signal trades.
   - **Bollinger Bands**: Identifies overbought/oversold conditions.
   - **RSI**: Determines market momentum using the Relative Strength Index.
   - **MACD**: Uses the Moving Average Convergence Divergence indicator.
   - **Mean Reversion**: Looks for price deviations from historical averages.
4. **Run the Backtest** and view performance metrics.
5. **Analyze Portfolio Value Trends** on a dynamic chart.

## 📎 Example Usage
- **Short-term SMA crossing above long-term SMA → Buy Signal** ✅
- **RSI above 70 → Overbought (Sell Signal)** 🔻
- **MACD line crossing below signal line → Trend Reversal (Sell Signal)** ⚠️

## 👤 Author
This application was developed by Sumanth Bharadwaj. Connect with me on **[LinkedIn](https://www.linkedin.com/in/hksb4602)**.


