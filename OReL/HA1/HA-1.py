import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_0 = 10000
delta_0 = 1/4

def reward(a, delta):
    if a == 'reg':
        return(np.random.binomial(1, 0.5-0.5*delta))
    if a == 'star':
        return(np.random.binomial(1, 0.5+0.5*delta))


Reward = np.array( [[[reward('reg', delta = delta_0) for i in range(n_0)],
                        [reward('star', delta = delta_0) for i in range(n_0)]] for j in range(20) ])

Reward1_Bandit = Reward
Reward_Bandit = Reward

R1 = np.full_like(Reward[:,0,:], 0)
R = np.full_like(Reward[:,0,:], 0)

rows = np.arange(20)

counts1 = np.ones((20, 2))
counts = np.ones((20, 2))
sums1 = np.array([Reward[:, 0, 0], Reward[:, 1, 1]]).T
sums = np.array([Reward[:, 0, 0], Reward[:, 1, 1]]).T

for t in range(2,n_0):

    UCB1 = sums1/counts1 + np.sqrt(np.log(t+1)/counts)
    UCB = sums/counts + np.sqrt(3*np.log(t+1)/(2*counts))

    a1 = UCB1.argmax(axis = 1)
    a = UCB.argmax(axis = 1)

    counts1[rows, a1] += 1
    counts[rows,a] +=1
    sums1[rows, a1] += Reward[rows, a1, t]
    sums[rows, a] += Reward[rows, a, t]

    Reward1_Bandit[rows, ~a1,t] = np.nan
    Reward_Bandit[rows, ~a, t] = np.nan

    mean_reward1 = Reward1_Bandit[:, :, :t].nanmean(axis=2)
    mean_reward = Reward_Bandit[:, :, :t].mean(axis=2)
    max_mean1 = mean_reward.max(axis=1)
    max_mean = mean_reward1.max(axis=1)

    R1[:, t] = max_mean1 - mean_reward1[rows, a1]
    R[:, t] = max_mean - mean_reward[rows, a]


    if t % 1000 == 0:
        print(f"Iteration {t}")

R1.cumsum(axis = 1)
R.cumsum(axis = 1)

R1_final = R1.mean(axis = 0)
R_final = R.mean(axis = 0)


#import matplotlib as mpl
#mpl.use("TkAgg")
#plt.ion()
#mpl.rcParams['axes.spines.left'] = False
#mpl.rcParams['axes.spines.right'] = False
#mpl.rcParams['axes.spines.top'] = False
#mpl.rcParams['axes.spines.bottom'] = False
plt.plot(R1_final)
plt.plot(R_final)
plt.show()













#if __name__ == '__main__':