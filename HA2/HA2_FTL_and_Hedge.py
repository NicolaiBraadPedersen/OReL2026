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
        x_t = np.random.binomial(1, mu)
        if x_t == a_t:
            loss[~a_t] += 1
        else:
            loss[a_t] += 1
        #print(loss)

        if a_t != opt:
            regret[t] = delta

    return np.cumsum(regret)

def run_algo_newx(T, func):
    loss = np.zeros(2)
    regret = np.zeros(T)
    loss_realized = 0

    for t in range(T):
        a_t = func(loss, K=2, T=T, t=t)

        if t % 2 == 0:
            x_t = np.random.binomial(1, 0.5)
        else:
            x_t = np.abs(x_t-1)

        if x_t == a_t:
            loss[~a_t] += 1
        else:
            loss[a_t] += 1
            loss_realized += 1
        #print(loss)

        regret[t] = loss_realized - np.min(loss)

    return regret

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
    eta_t = np.sqrt(np.log(K)/(t+1))
    p = np.exp(-eta_t * loss)/ sum(np.exp(-eta_t * loss))
    a = np.random.choice([0,1], p=p)
    return int(a)

def Hedge_t_2(loss, K, T, t):
    eta_t = 2*np.sqrt(np.log(K)/(t+1))
    p = np.exp(-eta_t * loss)/ sum(np.exp(-eta_t * loss))
    a = np.random.choice([0,1], p=p)
    return int(a)

#print(run_algo(0.25,100,FTL))
#print(run_algo(0.25,100,Hedge))

mu_0 = [1/2-1/4, 1/2-1/8, 1/2-1/16]
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

for mu in mu_0:
    FTL_0, FTL_0_se = np.array([run_algo(mu,2000,FTL) for i in range(10)]).mean(axis=0) , np.array([run_algo(mu,2000,FTL) for i in range(10)]).std(axis=0)
    Hedge_0, Hedge_0_se = np.array([run_algo(mu,2000,Hedge_t) for i in range(10)]).mean(axis=0), np.array([run_algo(mu,2000,Hedge_t) for i in range(10)]).std(axis=0)
    Hedge_1, Hedge_1_se = np.array([run_algo(mu,2000,Hedge_t_2) for i in range(10)]).mean(axis=0), np.array([run_algo(mu,2000,Hedge_t_2) for i in range(10)]).std(axis=0)
    Hedge_2, Hedge_2_se = np.array([run_algo(mu,2000,Hedge_T_2) for i in range(10)]).mean(axis=0), np.array([run_algo(mu,2000,Hedge_T_2) for i in range(10)]).std(axis=0)
    Hedge_3, Hedge_3_se = np.array([run_algo(mu,2000,Hedge_T_8) for i in range(10)]).mean(axis=0), np.array([run_algo(mu,2000,Hedge_T_8) for i in range(10)]).std(axis=0)
    plt.plot(FTL_0, label='FTL', color = colors[0])
    plt.plot(FTL_0 + FTL_0_se, label='FTL | se.', linestyle='dashed', color = colors[0])
    plt.plot(Hedge_2, label='Hedge | eta reg.', color = colors[1])
    plt.plot(Hedge_2 + Hedge_2_se, label='Hedge | eta reg. | + se.', linestyle='dashed', color = colors[1])
    plt.plot(Hedge_3, label='Hedge | eta repar.', color = colors[2])
    plt.plot(Hedge_3 + Hedge_3_se, label='Hedge | eta repar. | + se.', linestyle='dashed', color = colors[2])
    plt.plot(Hedge_0, label='Hedge | eta 7.8', color = colors[3])
    plt.plot(Hedge_0 + Hedge_0_se, label='Hedge | eta 7.8 | + se.', linestyle='dashed', color = colors[3])
    plt.plot(Hedge_1, label='Hedge | eta 7.8 tight', color = colors[4])
    plt.plot(Hedge_1 + Hedge_1_se, label='Hedge | eta 7.8 tight | + se.', linestyle='dashed', color = colors[4])
    plt.legend()
    plt.grid(True)
    plt.xlabel('T')
    plt.ylabel('Pseudo-regret')
    plt.title(f'Hedge vs FTL | mu = {mu}')
    plt.savefig(f'C:\\Users\\nicol\\OneDrive - University of Copenhagen\\Desktop\\4 år\\OReL\\HA2\\Hedge vs FTL {mu}.png')
    plt.show()


FTL_0, FTL_0_se = np.array([run_algo_newx(2000,FTL) for i in range(10)]).mean(axis=0) , np.array([run_algo_newx(2000,FTL) for i in range(10)]).std(axis=0)
Hedge_0, Hedge_0_se = np.array([run_algo_newx(2000,Hedge_t) for i in range(10)]).mean(axis=0), np.array([run_algo_newx(2000,Hedge_t) for i in range(10)]).std(axis=0)
Hedge_1, Hedge_1_se = np.array([run_algo_newx(2000,Hedge_t_2) for i in range(10)]).mean(axis=0), np.array([run_algo_newx(2000,Hedge_t_2) for i in range(10)]).std(axis=0)
Hedge_2, Hedge_2_se = np.array([run_algo_newx(2000,Hedge_T_2) for i in range(10)]).mean(axis=0), np.array([run_algo_newx(2000,Hedge_T_2) for i in range(10)]).std(axis=0)
Hedge_3, Hedge_3_se = np.array([run_algo_newx(2000,Hedge_T_8) for i in range(10)]).mean(axis=0), np.array([run_algo_newx(2000,Hedge_T_8) for i in range(10)]).std(axis=0)
plt.plot(FTL_0, label='FTL', color = colors[0])
plt.plot(FTL_0 + FTL_0_se, label='FTL | se.', linestyle='dashed', color = colors[0])
plt.plot(Hedge_2, label='Hedge | eta reg.', color = colors[1])
plt.plot(Hedge_2 + Hedge_2_se, label='Hedge | eta reg. | + se.', linestyle='dashed', color = colors[1])
plt.plot(Hedge_3, label='Hedge | eta repar.', color = colors[2])
plt.plot(Hedge_3 + Hedge_3_se, label='Hedge | eta repar. | + se.', linestyle='dashed', color = colors[2])
plt.plot(Hedge_0, label='Hedge | eta 7.8', color = colors[3])
plt.plot(Hedge_0 + Hedge_0_se, label='Hedge | eta 7.8 | + se.', linestyle='dashed', color = colors[3])
plt.plot(Hedge_1, label='Hedge | eta 7.8 tight', color = colors[4])
plt.plot(Hedge_1 + Hedge_1_se, label='Hedge | eta 7.8 tight | + se.', linestyle='dashed', color = colors[4])
plt.legend()
plt.grid(True)
plt.xlabel('T')
plt.ylabel('Regret')
plt.title(f'Hedge vs FTL New x')
plt.savefig(
    f'C:\\Users\\nicol\\OneDrive - University of Copenhagen\\Desktop\\4 år\\OReL\\HA2\\Hedge vs FTL new_x.png')
plt.show()