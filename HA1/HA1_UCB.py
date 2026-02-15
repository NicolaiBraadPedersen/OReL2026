import numpy as np
import matplotlib.pyplot as plt

T = 100000
runs = 20
deltas = [1 / 4, 1 / 8, 1 / 16]


def run_ucb(delta, T, modified=False):
    counts = np.zeros(2) + 1e-5
    sums = np.zeros(2)
    regret = np.zeros(T)

    for t in range(T):
        if modified:
            ucb = sums / counts + np.sqrt(3 * np.log(t + 1) / (2 * counts))
        else:
            ucb = sums / counts + np.sqrt(np.log(t + 1) / counts)

        a = np.argmax(ucb)

        if a == 0:
            r = np.random.binomial(1, 0.5 - 0.5 * delta)
            regret[t] = delta
        else:
            r = np.random.binomial(1, 0.5 + 0.5 * delta)
            regret[t] = 0

        counts[a] += 1
        sums[a] += r

    return np.cumsum(regret)

plt.figure(figsize=(10, 6))
for delta in deltas:
    R_ucb = np.zeros((runs, T))
    R_mod = np.zeros((runs, T))

    for i in range(runs):
        R_ucb[i] = run_ucb(delta, T, modified=False)
        R_mod[i] = run_ucb(delta, T, modified=True)

    mean_ucb = R_ucb.mean(axis=0)
    std_ucb = R_ucb.std(axis=0)

    mean_mod = R_mod.mean(axis=0)
    std_mod = R_mod.std(axis=0)

    plt.plot(mean_ucb, label=f'UCB1 delta={delta}')
    plt.plot(mean_ucb + std_ucb, alpha=0.2, linestyle='dashed', label=f'UCB1 delta={delta} + SE')

    plt.plot(mean_mod, label=f'New UCB delta={delta} + SE')
    plt.plot(mean_mod + std_mod, alpha=0.2, linestyle='dashed', label=f'New UCB delta={delta} + SE')

    plt.xlabel('T')
    plt.ylabel('Pseudo-regret')
    plt.title('UCB1 vs New UCB1')
    plt.legend()
    plt.show()