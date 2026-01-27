## 📊 Financial Assets

| Asset | Description | Volatility |
|------|------------|------------|
| 🚗 **TSLA** | Tesla Inc. | 🔥 High |
| 🏦 **BND** | Vanguard Total Bond Market ETF | 🟢 Low |
| 📈 **SPY** | S&P 500 ETF | 🟡 Medium |

📅 **Data Range:** 2015 – 2026 (daily adjusted close prices)

---

## 🧠 Task 2 — Time Series Forecasting

### 🎯 Objective
To **build, evaluate, and compare** classical and deep learning models capable of forecasting asset prices for portfolio optimization.

---

## 🔬 Methodology

### 🔹 1. Train / Test Split
- 🏋️ **Training:** 2015 – 2024  
- 🧪 **Testing:** 2025 – 2026  
- ⛔ No look-ahead bias (strict chronological split)  
- 📉 Returns used for diagnostics; **prices forecasted**

---

### 🔹 2. Stationarity Testing (ADF)
- 📊 Augmented Dickey–Fuller tests applied
- ❌ All series non-stationary
- ➗ Differencing required → `d = 1`

---

### 🔹 3. ACF & PACF Diagnostics
- 📈 Conducted on returns
- 🧩 Guides AR (p) and MA (q) selection
- ✅ Supports transparency and model explainability

---

### 🔹 4. Models Implemented

#### 📘 ARIMA (Auto-Regressive Integrated Moving Average)
- 🔍 Parameters selected via **AIC minimization**
- ⚙ Example: **TSLA → ARIMA(3,1,2)**
- 💡 Strong interpretability and stability

#### 🤖 LSTM (Long Short-Term Memory)
- 🪟 Window size: **30 days**
- 🧠 Architecture:
  - Single LSTM layer
  - Dense output layer
- ⏱ Trained for **8 epochs**
- 🎯 Captures nonlinear temporal dependencies

---

### 🔹 5. Evaluation Metrics
Models were compared using:
- 📏 **MAE** – Mean Absolute Error  
- 📐 **RMSE** – Root Mean Squared Error  
- 📊 **MAPE** – Mean Absolute Percentage Error  

---

## 🏆 Model Performance Summary

| Asset | Model | MAE | RMSE | MAPE (%) | Best |
|------|------|------|------|----------|------|
| 🏦 BND | ARIMA | 2.79 | 3.24 | 3.83 | ✅ |
| 🏦 BND | LSTM | 6.51 | 7.14 | 8.99 | ❌ |
| 📈 SPY | ARIMA | 35.79 | 42.55 | 5.71 | ✅ |
| 📈 SPY | LSTM | 179.58 | 212.15 | 27.85 | ❌ |
| 🚗 TSLA | ARIMA | 69.30 | 83.10 | 22.48 | ✅ |
| 🚗 TSLA | LSTM | 223.51 | 263.37 | 58.11 | ❌ |

---

## 🧾 Discussion of Model Selection

### 🔍 Key Findings
- ✅ **ARIMA consistently outperformed LSTM** across all assets
- 📉 Strong performance on low and medium volatility assets (BND, SPY)
- ⚠️ LSTM limited by dataset size and market noise
- 🧠 ARIMA offers better interpretability and reliability

### 🏁 Final Choice
> **ARIMA models were selected as the final forecasting approach**  
> for all assets moving into portfolio optimization.

---

## 🚀 What’s Next — Task 3

The ARIMA forecasts will be used for:
- 📅 Future price projection
- ⚖️ Risk–return estimation
- 🧮 Portfolio optimization and allocation analysis

---

## ✅ Project Status

- ✔ Task 2 completed  
- ✔ Forecasts generated and saved  
- ✔ Metrics exported  
- ✔ Visualizations produced  
- ✔ Notebook fully documented and modularized  

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Statsmodels` · `pmdarima` ·  
`TensorFlow / Keras` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

⭐ *If you find this project useful, feel free to star the repository.*
