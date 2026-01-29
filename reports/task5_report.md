
# Task 5 — Strategy Backtesting Report

## Performance Summary
                      Total Return  Annualized Return  Annualized Volatility  \
Static Strategy             0.1652             0.1528                 0.1379   
Rebalanced (No TC)          0.1652             0.1528                 0.1379   
Rebalanced (With TC)        0.1501             0.1389                 0.1382   
Benchmark 60/40             0.1504             0.1392                 0.1153   

                      Sharpe Ratio  Max Drawdown  
Static Strategy             1.1074       -0.1372  
Rebalanced (No TC)          1.1074       -0.1372  
Rebalanced (With TC)        1.0052       -0.1389  
Benchmark 60/40             1.2066       -0.1129  

## Key Findings
- Transaction costs materially impact rebalanced strategies
- Forecast-informed allocation remains competitive vs 60/40
- Walk-forward validation improves robustness

## Limitations
- Short backtest window
- No tax or slippage modeling
- Forecast uncertainty

