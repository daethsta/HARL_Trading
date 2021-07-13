#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nInst=100
currentPos = np.zeros(nInst)
indices = ["ind"]
return_window = 1

trade_val = 25000
edge = 0.007
fade_rate = 0.001
rate = 0.001

def n_day_returns(data, n, remove=True):
    """Calculates returns in disjoint n day windows"""
    df = data.iloc[n:].reset_index(drop=True) / data - 1
    df.dropna(axis=0, inplace=True)
    if remove:
        df = df.iloc[::n] # Keeps every nth return to prevent overlap
    return df

def get_betas(returns, indices):
    """Computes betas for each instrument"""
    return (returns.cov() / returns.var()).iloc[:100][indices]

def clip(position, price, max_val=10000, min_val=-10000):
    """Clips dollar position between -10k and 10k"""
    return int(min(max(position * price, min_val), max_val))

def getMyPosition (prcSoFar):
    global currentPos
    
    (nins,nt) = prcSoFar.shape
    data = pd.DataFrame(prcSoFar.T)
    
    if nt == 250:
        return currentPos
     
    data["ind"] = (data.iloc[:,50:] / data.iloc[0,50:]).mean(axis=1)
    returns = n_day_returns(data, return_window)
    betas = get_betas(returns, indices)
    stds = data.std()
    avg_std = stds[50:100].mean()
    rpos = np.zeros(nInst)
    
    for i in range(50,100):
        std = stds[i]
        price = data[i].iloc[-1]
        excess_return = returns[i].iloc[-1] - betas["ind"][i] * returns["ind"].iloc[-1] 
        vol_factor = std / avg_std * rate
        fade = fade_rate * (currentPos[i] * price / trade_val) 
        
        sell_edge = max(edge - fade + vol_factor, 0)
        buy_edge = min(-edge - fade - vol_factor, 0)
        
        if excess_return > sell_edge:
            currentPos[i] -= int( (trade_val / price ) * abs( (excess_return - sell_edge) / edge) )
            
        elif excess_return < buy_edge:
            currentPos[i] += int( (trade_val / price) * abs( (buy_edge - excess_return)  / edge) )
    return currentPos

    
