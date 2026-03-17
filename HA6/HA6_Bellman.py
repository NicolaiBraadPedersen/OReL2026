import numpy as np
import pandas as pd

class Bellman:
    def __init__(self,):
        self.r = 10
        self.c = 5
        self.N = 10
        self.mu = 0.1
        self.start_vals = [0,0]

    def value(self, prev_val):
        prev_val = np.array(prev_val)
        results = [0, 0]

        for i, s in enumerate(['F', 'B']):
            rewards = self.reward(s)
            probs = self.probability(s)

            q_values = rewards + probs @ prev_val
            results[i] = np.max(q_values)

        return results

    def reward(self,s):
        if s == 'F':
            reward = [0, -self.r]
        else:
            reward = [-self.c, -self.r]

        return np.array(reward)

    def probability(self,s):
        if s == 'F':
            probability = [[1-self.mu, self.mu],[1, 0]]
        else:
            probability = [[0, 1],[1, 0]]

        return np.array(probability)

    def run_Bellman(self):
        results = []
        results.append(self.start_vals)
        result = self.start_vals
        for i in range(1, self.N):
            result = self.value(result)
            results.append(result)

        return results

if __name__ == '__main__':
    blm = Bellman()

    df = pd.DataFrame(np.array(blm.run_Bellman()))
    print(df.to_latex())