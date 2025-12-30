"""
Optimize pair selection and number of pairs
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')
from cift.ml.stat_arb import engle_granger_coint, half_life_mean_reversion, KalmanState

# Larger universe to find best pairs
symbols = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY',
           'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'SCHW', 'AXP',
           'HD', 'LOW', 'WMT', 'TGT', 'COST', 'TJX', 'ROST', 'DG', 'BBY',
           'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
           'MSFT', 'AAPL', 'GOOGL', 'META', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'IBM', 'AMD',
           'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'MDLZ']

end = datetime.now()
start = end - timedelta(days=3*365)
data = yf.download(symbols, start=start, end=end, progress=False, threads=False)
prices = data['Close'].dropna()
print(f'Loaded {len(prices.columns)} stocks, {len(prices)} days')

# Find ALL cointegrated pairs
print('Finding all cointegrated pairs...')
pairs = []
syms = list(prices.columns)
for i, s1 in enumerate(syms):
    for s2 in syms[i+1:]:
        try:
            p1 = prices[s1].values
            p2 = prices[s2].values
            adf, pval, hedge, intercept = engle_granger_coint(p1, p2)
            if pval < 0.05:
                spread = p1 - hedge * p2 - intercept
                hl = half_life_mean_reversion(spread)
                if 5 <= hl <= 60:
                    # Calculate Hurst
                    lags = range(2, 20)
                    tau = [np.sqrt(np.std(np.subtract(spread[lag:], spread[:-lag]))) for lag in lags]
                    tau = np.array(tau)
                    valid = tau > 0
                    if valid.any():
                        reg = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
                        hurst = reg[0]
                    else:
                        hurst = 0.5
                    
                    # Better score: prioritize low hurst (mean-reverting)
                    score = (1 - pval) * (1 - abs(hl - 20)/40) * (0.6 - hurst)
                    pairs.append({
                        's1': s1, 's2': s2, 'hedge': hedge, 
                        'hl': hl, 'hurst': hurst, 'pval': pval, 'score': score
                    })
        except:
            pass

pairs.sort(key=lambda x: x['score'], reverse=True)
print(f'Found {len(pairs)} cointegrated pairs')

print('\nTop 15 pairs by score:')
for p in pairs[:15]:
    print(f"  {p['s1']:5s}/{p['s2']:5s}: HL={p['hl']:5.1f}, Hurst={p['hurst']:.2f}, pval={p['pval']:.3f}, score={p['score']:.3f}")

# Test different numbers of pairs
print('\nOptimizing number of pairs:')
print('N_pairs  Sharpe  Annual  MaxDD   Trades')
print('-' * 45)

best_sharpe = 0
best_n = 0

for n_pairs in [5, 8, 10, 12, 15, 20]:
    entry_z, exit_z = 2.5, 0.25
    lookback = 20
    daily_returns = []
    positions = {}
    trades = 0
    trading_pairs = pairs[:n_pairs]
    
    for t in range(lookback, len(prices)):
        day_pnl = 0.0
        
        for pair in trading_pairs:
            s1, s2 = pair['s1'], pair['s2']
            p1 = prices[s1].iloc[:t+1].values
            p2 = prices[s2].iloc[:t+1].values
            
            kalman = KalmanState(beta=pair['hedge'], Q=1e-5, R=1e-3)
            for i in range(min(100, len(p1))):
                idx = max(0, len(p1) - 100) + i
                hedge = kalman.update(p2[idx], p1[idx])
            
            spread = p1 - hedge * p2
            recent = spread[-lookback:]
            zscore = (spread[-1] - np.mean(recent)) / (np.std(recent) + 1e-10)
            
            pair_key = f"{s1}/{s2}"
            pos_size = 1.0 / n_pairs
            
            if pair_key in positions:
                pos = positions[pair_key]
                ret1 = (p1[-1] - p1[-2]) / p1[-2]
                ret2 = (p2[-1] - p2[-2]) / p2[-2]
                
                if pos['dir'] == 'long':
                    spread_ret = ret1 - pos['hedge'] * ret2
                else:
                    spread_ret = -ret1 + pos['hedge'] * ret2
                
                day_pnl += spread_ret * pos_size
                
                should_exit = False
                if pos['dir'] == 'long' and zscore >= -exit_z:
                    should_exit = True
                elif pos['dir'] == 'short' and zscore <= exit_z:
                    should_exit = True
                elif abs(zscore) > 4.0:
                    should_exit = True
                
                if should_exit:
                    day_pnl -= 0.001 * pos_size
                    del positions[pair_key]
                    trades += 1
            
            elif len(positions) < n_pairs:
                if zscore <= -entry_z:
                    positions[pair_key] = {'dir': 'long', 'hedge': hedge}
                    trades += 1
                elif zscore >= entry_z:
                    positions[pair_key] = {'dir': 'short', 'hedge': hedge}
                    trades += 1
        
        daily_returns.append(day_pnl)
    
    returns = np.array(daily_returns)
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        annual = (1 + returns).prod() ** (252/len(returns)) - 1
        cum = np.cumprod(1 + returns)
        maxdd = (cum / np.maximum.accumulate(cum) - 1).min()
        print(f'{n_pairs:7}  {sharpe:6.2f}  {annual*100:6.1f}%  {maxdd*100:6.1f}%  {trades:5}')
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_n = n_pairs

print(f'\nBest: {best_n} pairs with Sharpe {best_sharpe:.2f}')
