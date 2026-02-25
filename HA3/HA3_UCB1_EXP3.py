import numpy as np
import matplotlib.pyplot as plt
import os

# below is a personal style for plotting, will only work if the whole repo is cloned. User workaround is added
if os.environ.get('USERNAME') == 'nicol':
    from utils.plotting import use_earthy_style
    use_earthy_style()

class comparison():
    def __init__(self):
        self.start = 0.49

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
                ucb = loss_obs_ucb / counts_ucb - np.sqrt(np.log(t + 1) / (counts_ucb))
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

    def run_ucb_exp3_regret(self, T, K):
        counts_ucb = np.zeros(K) + 1e-5

        loss_obs_ucb = np.zeros(K)
        loss_obs_exp3 = np.zeros(K)
        loss_est = np.zeros(K)

        regret_ucb = np.zeros(T)
        loss_realized_ucb = 0
        regret_exp3 = np.zeros(T)
        loss_realized_exp3 = 0

        Losses_true = self.adversarial_losses(T,K, self.start)
        Losses_best_hindsigt = Losses_true.cumsum(axis = 1).min(axis = 0)

        for t in range(T):
            probs = self.EXP3(loss_est, K, t)
            a_exp3 = np.random.choice(K, p=probs)

            if t < K:
                a_ucb = t
            else:
                ucb = loss_obs_ucb / counts_ucb - np.sqrt(np.log(t + 1) / (counts_ucb))
                a_ucb = np.argmin(ucb)

            l = Losses_true[a_ucb,t]

            counts_ucb[a_ucb] += 1
            loss_obs_ucb[a_ucb] += l
            loss_realized_ucb += l

            l = Losses_true[a_exp3,t]

            loss_obs_exp3[a_exp3] +=  l
            loss_est[a_exp3] += l / probs[a_exp3]
            loss_realized_exp3 += l

            regret_ucb[t] = loss_realized_ucb - Losses_best_hindsigt[t]
            regret_exp3[t] = loss_realized_exp3 - Losses_best_hindsigt[t]


        return regret_ucb, regret_exp3

    @staticmethod
    def EXP3(loss_est, K, t):
        eta_t = np.sqrt(np.log(K)/((t+1)*K))
        p_t = np.exp(-eta_t * loss_est)/ sum(np.exp(-eta_t * loss_est))
        return p_t

    @staticmethod
    def adversarial_losses(T, K, start):
        losses = np.ones((K, T))

        losses[0, :K] = start
        losses[1:, :K] = 0.50

        for t in range(K + 1, T):
            if t % K == 0:
                losses[0, t] = 1
                losses[1:, t] = 1
            else:
                losses[0, t] = 0
                losses[1:, t] = 1

        return losses

if __name__ == '__main__':
    comp = comparison()

    for s in [0.02,0.00001]:
        plt.figure(figsize=[12, 12])
        i = 1
        comp.start = 0.5 - s
        for K0 in [2,4,8,16]:
            plt.subplot(2, 2, i)
            results = np.array([comp.run_ucb_exp3_regret(100000,K0)])
            mean_vals = results.mean(axis=0)
            std_vals_plus = results.mean(axis=0) + results.std(axis=0)

            plt.plot(mean_vals[1, :], label=r'EXP3 $\mu$')
            plt.ylim((0,100000))
            plt.plot(mean_vals[0, :], label=r'UCB1 $\mu$')
            plt.title(f'K = {K0}')
            plt.legend()
            i += 1

        plt.suptitle(r'Break UCB1 | $\Delta=$' + f'{s}')
        plt.savefig(r'C:\Users\nicol\OneDrive - University of Copenhagen\Desktop\4 år\OReL\HA3' + rf'\Break_UCB1_{s}.png')

    i = 1
    fig, axes = plt.subplots(4, 3, figsize=[15, 20], sharey='row')
    plt.suptitle('UCB1 vs EXP3 Comparison', fontsize=16)


    k_values = [2, 4, 8, 16]
    delta_values = [1 / 4, 1 / 8, 1 / 16][::-1]
    for row_idx, K0 in enumerate(k_values):
        for col_idx, delta0 in enumerate(delta_values):

            ax = axes[row_idx, col_idx]

            comp = comparison()
            results = np.array([comp.run_ucb_exp3(delta=delta0, T=10000, K=K0) for _ in range(20)])

            mean_vals = results.mean(axis=0)
            std_vals = results.std(axis=0)

            ax.plot(mean_vals[1, :], label=r'EXP3 $\mu$')
            ax.fill_between(range(len(mean_vals[1, :])),
                            mean_vals[1, :],
                            mean_vals[1, :] + std_vals[1, :],
                            alpha=0.2, label=r'EXP3 $\mu$ + $\sigma$')

            ax.plot(mean_vals[0, :], label=r'UCB1 $\mu$')
            ax.fill_between(range(len(mean_vals[0, :])),
                            mean_vals[0, :],
                            mean_vals[0, :] + std_vals[0, :],
                            alpha=0.2, label = r'UCB1 $\mu$ + $\sigma$')

            ax.set_title(rf'K = {K0}, $\Delta$ = {delta0:.3f}')
            ax.set_xlabel('T')

            if col_idx == 0:
                ax.set_ylabel('Pseudo Regret')

            ax.legend()

    plt.tight_layout()
    plt.suptitle('UCB1 vs EXP3')
    plt.savefig(r'C:\Users\nicol\OneDrive - University of Copenhagen\Desktop\4 år\OReL\HA3'+r'\UCB1_EXP3png')

    test = 1