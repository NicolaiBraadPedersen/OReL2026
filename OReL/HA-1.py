import numpy as np
import pandas as pd

n_0 = 100
delta_0 = 1/4

def reward(a, delta):
    if a == 'reg':
        return(np.random.binomial(1, 0.5-0.5*delta))
    if a == 'star':
        return(np.random.binomial(1, 0.5+0.5*delta))


Reward = pd.DataFrame( [[reward('reg', delta = delta_0) for i in range(n_0)],
                        [reward('star', delta = delta_0) for i in range(n_0)]])

N = pd.DataFrame(np.full_like(Reward, False)).astype('bool')
N1 = pd.DataFrame(np.full_like(Reward, False)).astype('bool')

R = pd.DataFrame(np.full_like(Reward, np.nan))
R1 = pd.DataFrame(np.full_like(Reward, np.nan))

#Initualize the first 2 choices
N.loc[0,0] = True
N1.loc[0,0] = True
N.loc[1, 1] = True
N1.loc[1, 1] = True

for t in range(2,n_0):
    UCB1 = Reward[N1].mean(axis = 1) + np.sqrt(np.log(t)/N1.sum(axis = 1))
    UCB = Reward[N].mean(axis = 1) + np.sqrt(3*np.log(t)/(2*N.sum(axis = 1)))

    N1.loc[UCB1.argmax(),t] = True
    N.loc[UCB.argmax(),t] = True

















if __name__ == '__main__':