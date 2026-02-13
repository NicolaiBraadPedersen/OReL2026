import numpy as np
import pandas as pd
#import tkinter
#import matplotlib as mpl
#mpl.use("TkAgg")
#mpl.rcParams['axes.spines.left'] = False
#mpl.rcParams['axes.spines.right'] = False
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.bottom'] = False
import matplotlib.pyplot as plt


def run_algo(mu, T, func):
    loss = np.zeros(2)
    regret = np.zeros(T)

    mu_0 = mu
    mu_1 = 1-mu
    delta = np.abs(mu_0-mu_1)
    opt = np.argmin([mu_0,mu_1])

    for t in range(T):
        a_t = func(loss, K=2, T=T, t=t)
        np.random.seed(t+42)
        x_t = np.random.binomial(1, mu)
        if x_t == a_t:
            loss[~a_t] += 1
        else:
            loss[a_t] += 1
        #print(loss)

        if a_t != opt:
            regret[t] = delta

    return np.cumsum(regret)

def FTL(loss, K, T, t):
    return np.argmin(loss)

def Hedge_T_2(loss, K, T, t):
    eta_t = np.sqrt(2*np.log(K)/T)
    p = np.exp(-eta_t * loss)/ sum(np.exp(-eta_t * loss))
    a = np.random.choice([0,1], p=p)
    return int(a)

def Hedge_T_8(loss, K, T, t):
    eta_t = np.sqrt(8*np.log(K)/T)
    p = np.exp(-eta_t * loss)/ sum(np.exp(-eta_t * loss))
    a = np.random.choice([0,1], p=p)
    return int(a)

def Hedge_t(loss, K, T, t):
    eta_t = np.sqrt(np.log(K)/t+1)
    p = np.exp(-eta_t * loss)/ sum(np.exp(-eta_t * loss))
    a = np.random.choice([0,1], p=p)
    return int(a)

def Hedge_t_2(loss, K, T, t):
    eta_t = 2*np.sqrt(np.log(K)/t+1)
    p = np.exp(-eta_t * loss)/ sum(np.exp(-eta_t * loss))
    a = np.random.choice([0,1], p=p)
    return int(a)

#print(run_algo(0.25,100,FTL))
#print(run_algo(0.25,100,Hedge))


#import matplotlib as mpl
#mpl.use("TkAgg")
#plt.ion()
#mpl.rcParams['axes.spines.left'] = False
#mpl.rcParams['axes.spines.right'] = False
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.bottom'] = False

FTL_0 = np.array([run_algo(1/2-1/4,2000,FTL) for i in range(10)]).mean(axis=0)
FTL_1 = np.array([run_algo(1/2-1/8,2000,FTL) for i in range(10)]).mean(axis=0)
FTL_2 = np.array([run_algo(1/2-1/16,2000,FTL) for i in range(10)]).mean(axis=0)

mu_0 = [1/2-1/4, 1/2-1/8, 1/2-1/16]

Hedge_0 = np.array([run_algo(1/2-1/4,2000,Hedge_t) for i in range(1000)]).mean(axis=0)

plt.plot(Hedge_0)
#plt.plot(FTL)
plt.show()