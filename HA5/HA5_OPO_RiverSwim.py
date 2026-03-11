import numpy as np
import matplotlib.pyplot as plt
import os

# below is a personal style for plotting, will only work if the whole repo is cloned. User workaround is added
if os.environ.get('USERNAME') == 'nicol':
    from utils.plotting import use_earthy_style
    use_earthy_style()

def transition_probs(s,a):
    '''
    transition probabilitites in riverswim
    :param s:
    :param a:
    :return: next state, reward
    '''
    reward = 0
    if a == 0:
        if s == 0:
            s_prime = 0
            reward = 0.05
        else:
            s_prime = s-1
    else:
        if s == 0:
            s_prime = np.random.choice([0,1],size = 1, p = [0.6,0.4])[0]
        elif s in [1,2]:
            s_prime = np.random.choice([s-1,s,s+1],size = 1, p = [0.05,0.55,0.4])[0]
        else:
            s_prime = np.random.choice([2,3],size = 1, p = [0.4,0.6])[0]
            if s_prime == 3:
                reward = 1

    return s_prime, reward

def Q_star(T=3*10**5, gamma=0.96, off_policy=False, pi = []):
    results_off_policy = []

    number_of_states = 4
    number_of_actions = 2
    Q = np.zeros((number_of_states, number_of_actions))
    R = np.zeros((number_of_states, number_of_actions))
    P = np.zeros((number_of_states, number_of_actions, number_of_states))

    R[:, 0] = [0.05, 0 , 0, 0]
    R[:, 1] = [0, 0, 0, 1*0.6]

    P[0, 1, :] = [0.6, 0.4, 0, 0]
    P[1, 1, :] = [0.05, 0.55, 0.4, 0]
    P[2, 1, :] = [0, 0.05, 0.55, 0.4]
    P[3, 1, :] = [0, 0, 0.4, 0.6]

    P[0, 0, :] = [1, 0, 0, 0]
    P[1, 0, :] = [1, 0, 0, 0]
    P[2, 0, :] = [0, 1, 0, 0]
    P[3, 0, :] = [0, 0, 1, 0]

    Q_off = Q.copy()
    for t in range(T):
        if off_policy:
            Q_off_prev = Q_off.copy()
        Q_prev = Q.copy()

        for s in range(4):
            for a in range(2):
                Q[s, a] = R[s, a] + gamma * np.sum(P[s, a, :] * np.max(Q_prev, axis=1))
        if off_policy:
            for s in range(4):
                for a in range(2):
                    Q_off[s, a] = R[s, a] + gamma * np.sum(P[s, a, :] * Q_off_prev[np.arange(4),pi[t]])

            results_off_policy.append(Q_off[0,1])

    if off_policy:
        return results_off_policy
    else:
        return Q

def ce_opo(Q_prime):
    results = []
    results_pi = []

    epsilon = 0.15
    alpha = 0.25
    beta = 1.1
    gamma = 0.96

    Q = np.zeros([4,2])
    Q_prev = np.zeros((4, 2))
    P = np.full([4,2,4],1/4)
    r_sum = np.zeros([4,2])
    R = np.zeros([4, 2])

    N_obs_sas = np.zeros([4, 2, 4])
    N_obs_sa = np.zeros([4, 2])
    N_obs_sa_prev = np.zeros([4, 2])
    s_current = 0

    for i in range(3*10**5):
        # do the gamma greedy policy sampling
        choice = np.random.choice(['random','greedy'],size = 1, p = [epsilon,1-epsilon])[0]

        if choice == 'random':
            action = np.random.choice([0,1],size = 1, p = [0.5,0.5])[0]
        else: # choice == 'greedy'
            action = np.argmax(Q[s_current,:])

        s_obs , r_obs = transition_probs(s_current , action)

        N_obs_sas[s_current,action,s_obs] += 1
        N_obs_sa[s_current, action] += 1

        r_sum[s_current, action] += r_obs

        if (N_obs_sa > beta * N_obs_sa_prev).any():
            N_obs_sa_prev = N_obs_sa.copy()

            # R[s_current, action] = (alpha + r_sum[s_current, action]) / (alpha + N_obs_sa[s_current, action])
            # P = (alpha + N_obs_sas) / (alpha * 4 + N_obs_sa[:, :, np.newaxis])
            #
            # max_Q = np.max(Q, axis=1)
            # for s in range(4):
            #     for a in range(2):
            #         Q[s, a] = R[s, a] + gamma * np.sum(P[s, a, :] * max_Q)
            for state in range(4):
                for action in range(2):
                    P[state, action, :] = (N_obs_sas[state, action, :] + alpha) / (N_obs_sa[state, action] + alpha * 4)
                    R[state, action] = (r_sum[state, action] + alpha) / (N_obs_sa[state, action] + alpha)
                    Q[state, action] = (R[state, action] +
                                        gamma * np.sum(P[state, action, :] * np.max(Q_prev, axis=1)))
        Q_prev = Q.copy()
        s_current = s_obs

        res = np.max(np.abs(Q_prime - Q))
        res_pi = np.argmax(Q, axis = 1)
        results.append(res)
        results_pi.append(res_pi)

    return results, results_pi


Q_star_0 = Q_star()
lst, pi_0 = ce_opo(Q_star_0)

lst_1 = Q_star_0[0,1] - Q_star(pi = pi_0, off_policy=True)

plt.plot(lst)
plt.ylabel('$||Q^* - Q_t||_\infty$')
plt.xlabel('$\log_10$ transform of time')
plt.label('Error in estimated Q-value at time t')
plt.xscale('log')
plt.show()

plt.plot(lst_1)
plt.ylabel('$Q^*(1,right) - Q^{\hat\pi_t}(1,right)')
plt.xlabel('$\log_10$ transform of time')
plt.label('Q loss at left bank, for estimated policy')
plt.xscale('log')
plt.show()



