# 📊 Portfolio Optimization Using Time Series Forecasting

This project implements a **data-driven portfolio optimization pipeline** using historical financial data, classical time series forecasting (ARIMA), and portfolio theory.  
The workflow is structured into **four tasks**, progressing from data exploration to forecast-based portfolio optimization.

---

## 📁 Repository Structure

```text
portfolio-optimization/
│
├── data/
│   ├── raw/                     # Original downloaded market data
│   └── processed/               # Cleaned data, forecasts, trained models
│       ├── prices_aligned.csv
│       ├── tsla_arima_forecast.csv
│       ├── bnd_arima_forecast.csv
│       ├── spy_arima_forecast.csv
│
├── notebooks/
│   ├── task1_eda.ipynb
│   ├── task2_forecasting.ipynb
│   └── task3_portfolio_analysis.ipynb
│
├── reports/
│   └── figures/
│       ├── tsla_forecast.png
│       ├── bnd_forecast.png
│       ├── spy_forecast.png
│       └── portfolio_forecast.png
│
├── src/
│   ├── __init__.py
│   ├── forecasting.py
│   └── portfolio_analysis.py
│
├── README.md
├── requirements.txt
└── .gitignore

📊 Financial Assets
Asset	Description	Volatility
🚗 TSLA	Tesla Inc.	🔥 High
🏦 BND	Vanguard Total Bond Market ETF	🟢 Low
📈 SPY	S&P 500 ETF	🟡 Medium

📅 Data Range: 2015 – 2026 (daily adjusted close prices)

🧠 Task 1 — Exploratory Data Analysis (EDA)
🎯 Objective

Understand historical price behavior, trends, volatility, and correlations.

🔍 Key Steps

Price visualization and summary statistics

Daily return computation

Volatility comparison across assets

Correlation analysis

📓 Notebook: notebooks/task1_eda.ipynb

🧠 Task 2 — Time Series Forecasting
🎯 Objective

Build and evaluate forecasting models to predict future asset prices for portfolio analysis.

🔬 Methodology
1️⃣ Train / Test Split

🏋️ Training: 2015 – 2024

🧪 Testing: 2025 – 2026

⛔ Strict chronological split (no look-ahead bias)

2️⃣ Stationarity Testing

Augmented Dickey–Fuller (ADF) tests applied

All series non-stationary → differencing required (d = 1)

3️⃣ Model Diagnostics

ACF and PACF used to guide AR and MA terms

Parameter selection via AIC minimization

🤖 Models Implemented
📘 ARIMA

Asset-specific orders (e.g., TSLA → ARIMA(3,1,2))

Interpretable and stable

Final model selected for all assets

🤖 LSTM (Experimental)

Sliding window sequence modeling

Tested but underperformed ARIMA

Not used for portfolio decisions

📊 Model Performance Summary
Asset	Model	MAE	RMSE	MAPE (%)	Best
🏦 BND	ARIMA	2.79	3.24	3.83	✅
📈 SPY	ARIMA	35.79	42.55	5.71	✅
🚗 TSLA	ARIMA	69.30	83.10	22.48	✅

📓 Notebook: notebooks/task2_forecasting.ipynb

📊 Task 3 — Forecast-Based Portfolio Analysis
🎯 Objective

Use ARIMA forecasts to estimate expected returns, risk, and portfolio behavior.

📥 Inputs

ARIMA forecast CSVs in data/processed/

Historical aligned price data

🧮 Methods

Implemented in src/portfolio_analysis.py:

Expected return estimation from forecasted prices

Annualized volatility calculation

Forecast-based trend analysis

Combined portfolio forecast construction

📈 Outputs

Asset-level forecast metrics

Combined portfolio forecast plot

Metrics ready for optimization

📓 Notebook: notebooks/task3_portfolio_analysis.ipynb
📊 Figures saved to: reports/figures/

🔮 Task 4 — Portfolio Optimization (Next)

Planned next steps:

Mean–variance optimization

Maximum Sharpe ratio portfolio

Efficient frontier visualization

Comparison with equal-weight portfolio

📍 Implemented in a separate task4-portfolio-optimization branch.

🛠️ Tech Stack

Python · Pandas · NumPy · Statsmodels · pmdarima
Scikit-learn · Matplotlib · Seaborn · TensorFlow/Keras

✅ Project Status

✔ Task 1 — EDA completed

✔ Task 2 — Forecasting completed

✔ Task 3 — Portfolio analysis completed

🔜 Task 4 — Optimization in progress