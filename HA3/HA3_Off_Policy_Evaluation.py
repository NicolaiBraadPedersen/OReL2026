import numpy as np
import pandas as pd

class off_policy_evaluation():
    def __init__(self):
        self.data = []
        self.gamma = 0.97


    def load_data(self, n = 0):
        df = pd.read_csv(f'Datasets_HA3/dataset{n}.csv')
        self.data = df
        self.df_N = self.compute_N()
        return df

    def compute_N(self):
        data = self.data
        S_n = max(data['state'].max(), data['next state'].max()) + 1
        self.S_n = S_n

        df_N = np.zeros((2,S_n, S_n))

        for a in [0,1]:
            for i in range(S_n):
                for j in range(S_n):
                    df_N[a,i,j] = sum((data['action'] == a) & (data['state'] == i) & (data['next state'] == j))

        #df_N[:,S_n-1,:] = 0

        return df_N

    def Bellman(self,P,r):
        return np.linalg.inv(np.eye(P.shape[0])-self.gamma*P) @ r

    @staticmethod
    def pi(action, behavior = False):
        if behavior == False:
            if action == 0:
                return 0.05
            else:
                return 0.95
        else:
            if action == 0:
                return 0.35
            else:
                return 0.65

    def algo_CE_OPE(self, alpha = 0):
        df_N = self.df_N
        S_n = self.S_n
        data = self.data

        if alpha == 0:
            alpha = 1/S_n

        P_a_s_s = np.zeros((2,S_n,S_n))
        for a in [0,1]:
            for i in range(S_n):
                for j in range(S_n):
                    P_a_s_s[a,i,j] = ((df_N[a,i,j] + alpha) / (df_N[a,i,:].sum() + alpha*S_n))

        R_a_s = np.zeros((2,S_n))
        for a in [0,1]:
            for i in range(S_n):
                # if i == S_n:
                #     R_a_s[a, i] == 0
                # else:
                indicator_mask = (data['action'] == a) & (data['state'] == i)
                R_a_s[a,i] = (alpha + data[indicator_mask]['reward'].sum()) / (alpha + indicator_mask.sum())

        policy_matrix = np.zeros((2))
        for i in [0,1]:
            policy_matrix[i] = self.pi(action = i)

        P_s_s = np.einsum('i,ijk->jk', policy_matrix, P_a_s_s)
        R_s = np.einsum('i,ik->k', policy_matrix, R_a_s)

        Value = self.Bellman(P_s_s, R_s)

        return Value[0]

    def batch_data(self):
        data = self.data
        s_max = self.S_n - 1

        splits = []
        for i in range(len(data)):
            if (data['state'][i] == s_max) & (data['next state'][i] == 0):
                splits.append(i+1)


        indices = [0] + list(splits)
        D = []
        for start, end in zip(indices[:-1], indices[1:]):
            D.append(data[start:end])

        return D

    def algo_IS(self, alpha = 0, method = ''):
        D = self.batch_data()

        def rho_and_rew(df, method = method):
            rho_x = 1
            sum_r = 0
            sum_pd_product = 0
            for i in range(len(x)):
                a = x.iloc[i]['action']
                r = x.iloc[i]['reward']

                rho_x *= self.pi(a)/self.pi(a,behavior = True)
                sum_r += r * self.gamma ** i

                if method == 'PerDecision':
                    sum_pd_product += rho_x * r * self.gamma ** i

            if method == 'PerDecision':
                return sum_pd_product
            else:
                return rho_x, sum_r

        result = 0
        sum_rho = 0
        for x in D:
            if method == 'PerDecision':
                result += rho_and_rew(x, method=method)
            else:
                result += rho_and_rew(x,method=method)[0]*rho_and_rew(x,method=method)[1]
                sum_rho += rho_and_rew(x, method=method)[0]

        if method == 'Weighted':
            return result / sum_rho
        else:
            return result / len(D)


    def true_value(self):
        policy_matrix = np.zeros((2))
        for i in [0,1]:
            policy_matrix[i] = self.pi(action = i)

        # dimension (a, s, s')
        P_a_s_s = np.array([[[1,0,0,0,0,0,0],
                     [1,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0],
                     [0,0,1,0,0,0,0],
                     [0,0,0,1,0,0,0],
                     [0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1]],
                    [[0.6, 0.4, 0, 0, 0, 0, 0],
                     [0.05, 0.55, 0.4, 0, 0, 0,0],
                     [0, 0.05, 0.55, 0.4, 0, 0,0],
                     [0, 0, 0.05, 0.55, 0.4, 0,0],
                     [0, 0, 0, 0.05, 0.55, 0.4,0],
                     [0, 0, 0, 0, 0, 0,1],
                     [0, 0, 0, 0, 0, 0,1]]])

        P_s_s = np.tensordot(policy_matrix, P_a_s_s, axes=(0, 0))

        reward_s = np.array([0.05*0.05,0,0,0,0,1,0])

        Value = self.Bellman(P_s_s, reward_s)

        return Value[0]




if __name__ == "__main__":
    ope = off_policy_evaluation()
    ope.load_data(0)
    results_i = [ope.algo_CE_OPE(), ope.algo_IS(), ope.algo_IS(method='Weighted'), ope.algo_IS(method='PerDecision')]

    V_true = ope.true_value()
    results_ii = [np.abs(V_true - x) for x in results_i]


    lst_CE_OPE = []
    lst_IS = []
    lst_WIS = []
    lst_PDIS = []
    for i in range(11):
        ope.load_data(i)
        lst_CE_OPE.append( ope.algo_CE_OPE())
        lst_IS.append( ope.algo_IS())
        lst_WIS.append( ope.algo_IS(method = 'Weighted'))
        lst_PDIS.append( ope.algo_IS(method = 'PerDecision'))

    results_iii = [np.var(lst_CE_OPE), np.var(lst_IS), np.var(lst_WIS), np.var(lst_PDIS)]

    res_1 = ", ".join(f"{x:.5f}" for x in results_i)
    res_2 = ", ".join(f"{x:.5f}" for x in results_ii)
    res_3 = ", ".join(f"{x:.5f}" for x in results_iii)
    print(f'results_i = {res_1}')
    print(f'results_ii = {res_2}')
    print(f'results_iii = {res_3}')

    test = 1
