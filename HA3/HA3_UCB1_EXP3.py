import numpy as np
import matplotlib.pyplot as plt
import os

# below is a personal style for plotting, will only work if the whole repo is cloned. User workaround is added
if os.environ.get('USERNAME') == 'nicol':
    from utils.plotting import use_earthy_style
    use_earthy_style()

class comparison():
    def __init__(self):
        pass

    def run_ucb_exp3(self, delta, T, K):
        counts_ucb = np.zeros(K) + 1e-5

        loss_obs_ucb = np.zeros(K)
        loss_obs_exp3 = np.zeros(K)
        loss_est = np.zeros(K)

        regret_ucb = np.zeros(T)
        regret_exp3 = np.zeros(T)

        for t in range(T):
            probs = self.EXP3(loss_est, K, t)
            a_exp3 = np.random.choice(K, p=probs)

            if t < K:
                a_ucb = t
            else:
                ucb = loss_obs_ucb / counts_ucb - np.sqrt(3 * np.log(t + 1) / (2 * counts_ucb))
                a_ucb = np.argmin(ucb)

            if a_ucb == 0:
                l = np.random.binomial(1, 0.5)
                regret_ucb[t] = 0
            else:
                l = np.random.binomial(1, 0.5 + delta)
                regret_ucb[t] = delta

            counts_ucb[a_ucb] += 1
            loss_obs_ucb[a_ucb] += l

            if a_exp3 == 0:
                l = np.random.binomial(1, 0.5)
                regret_exp3[t] = 0
            else:
                l = np.random.binomial(1, 0.5 + delta)
                regret_exp3[t] = delta

            loss_obs_exp3[a_exp3] +=  l
            loss_est[a_exp3] += l / probs[a_exp3]

        return np.cumsum(regret_ucb), np.cumsum(regret_exp3)

    @staticmethod
    def EXP3(loss_est, K, t):
        eta_t = np.sqrt(np.log(K)/((t+1)*K))
        p_t = np.exp(-eta_t * loss_est)/ sum(np.exp(-eta_t * loss_est))
        return p_t

if __name__ == '__main__':

    for delta0 in [1 / 4, 1 / 8, 1 / 16]:
        i = 1
        plt.figure(figsize=[12, 12])
        for K0 in [2,4,8,16]:

            plt.subplot(2, 2, i)
            comp = comparison()
            results = np.array([comp.run_ucb_exp3(delta=delta0, T=10000, K=K0) for i in range(20)])
            mean_vals = results.mean(axis=0)
            std_vals_plus = results.mean(axis=0) + results.std(axis=0)

            line, = plt.plot(mean_vals[1, :], label='EXP3 mean')
            plt.plot(std_vals_plus[1,:], label='EXP3 mean + std', color = line.get_color(), linestyle = 'dashed', alpha = 0.6)

            line, = plt.plot(mean_vals[0, :], label='UCB1 mean')
            plt.plot(std_vals_plus[0, :], label='UCB1 mean + std', color=line.get_color(), linestyle = 'dashed', alpha = 0.6)

            plt.legend()
            plt.title(f'K = {K0}, $\delta$ = {delta0}')
            plt.ylabel('Pseudo Regret')
            plt.xlabel('T')

            i += 1
            print(i)

        plt.tight_layout()
        plt.savefig(r'C:\Users\nicol\OneDrive - University of Copenhagen\Desktop\4 år\OReL\HA3'+rf'\UCB1_EXP3_{delta0}.png')

    test = 1