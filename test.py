#!/usr/bin/env python

# RENAME THIS FILE WITH YOUR TEAM NAME.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trade_val = 0
edge = 0
i = 1
rate = 0.005
obj = {}
obj['Parameters'] = ['Trade Value', 'Edge', 'Fade Rate', 'Rate','mean(PL)','return', 'annSharpe(PL)', 'totDvolume']
                    

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

def getPosition (prcSoFar):
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
#             currentPos[i] -= int( (trade_val / price ) * abs( excess_return / sell_edge) )
#             currentPos[i] = clip(currentPos[i], price)
            
        elif excess_return < buy_edge:
            currentPos[i] += int( (trade_val / price) * abs( (buy_edge - excess_return)  / edge) )
#             currentPos[i] += int( (trade_val / price) * abs( excess_return  / buy_edge) )                    
#             currentPos[i] = clip(currentPos[i], price)
    return currentPos



def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolume0 = 0
    totDVolume1 = 0
    frac0 = 0.
    frac1 = 0.
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    for t in range(201,251):
        prcHistSoFar = prcHist[:,:t]
        newPosOrig = getPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:,-1] 
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.array([int(p) for p in np.clip(newPosOrig, -posLimits, posLimits)])
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume0 = np.sum(dvolumes[:50])
        dvolume1 = np.sum(dvolumes[50:])
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        totDVolume0 += dvolume0
        totDVolume1 += dvolume1
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
            frac0 = totDVolume0 / totDVolume
            frac1 = totDVolume1 / totDVolume
        #print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf frac0: %.4lf frac1: %.4lf" % (t,value, todayPL, totDVolume, ret, frac0, frac1))
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = 16 * plmu / plstd
    return (plmu, ret, annSharpe, totDVolume)

for trade_val in np.linspace(5000,25000,4):
    print ("=====New trade_val=====")
    print("trade_val: %.0lf" % trade_val)
    for rate in np.linspace(0.001, 0.01, 4):
        for edge in np.linspace(0.001, 0.02,4):
            for fade_rate in np.linspace(0.001, 0.05,4):
                nInst=100
                currentPos = np.zeros(nInst)
                indices = ["ind"]
                return_window = 1

                nInst = 0
                nt = 0

                # Commission rate.
                commRate = 0.0050

                # Dollar position limit (maximum absolute dollar value of any individual stock position).
                dlrPosLimit = 10000

                pricesFile="./prices250.txt"
                prcAll = loadPrices(pricesFile)
                #print ("Loaded %d instruments for %d days" % (nInst, nt))

                (meanpl, ret, sharpe, dvol) = calcPL(prcAll)
                obj['Test ' + str(i)] = [trade_val, edge, fade_rate, rate, meanpl, ret, sharpe, dvol]
                i+=1

print(obj)
df1 = pd.DataFrame(data=obj)
df1 = np.transpose(df1)
df1.to_csv("./file.csv", sep=',',index=False)
