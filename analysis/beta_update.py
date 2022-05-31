from scipy.stats import beta
import numpy as np

class BetaUpdate():
    """Beta updating functions"""
    
    def __init__(self, prior_a, prior_b, n_choices):
        self.a = prior_a
        self.b = prior_b
        self.n = n_choices
        self.mu = self.a / (self.a + self.b)
        self.sigma = self.a * self.b / ((self.a + self.b)**2 * (self.a + self.b + 1))
        
    def simple_update(self, data):
        if data != "no":
            self.a += data 
            self.b += 1 + data * -1
        return self.a, self.b 
    
    def increment_parameters(self, a, b):
        self.a += a
        self.b += b
        return self.a, self.b 

    def weighted_update(self, data, w):
        if data == 1:
            self.a += 1 * w
        elif data == 0:
            self.b += 1 * w
        return self.a, self.b

    def discrete_beta(self):
        betacdf = beta(self.a, self.b).cdf
        beta_bins = []
        beta_bins.append(betacdf(1 / self.n))
        for beta_bin in range(2, self.n + 1):
            beta_bins.append(betacdf(beta_bin / self.n) - betacdf((beta_bin - 1) / self.n))
        return beta_bins

    def get_a(self, mu, sigma):
        return ((1 - mu) / sigma - 1 / mu) * mu**2

    def get_b(self, alpha, mu):
        return alpha * (1 / mu - 1)
    
    def get_expectation(self, observations, evidence, weights, corr_weight = 1):
        bin_probs = np.zeros((self.n))
        a_obs = observations.count(1)
        b_obs = observations.count(0)
        for e, w in zip(evidence, weights): 
            e = [int(i) for i in e if i != 'no']
            Beta = beta(self.a + a_obs + e.count(1) * corr_weight, self.b + b_obs + e.count(0) * corr_weight).cdf
            bin_count = 0
            for judgmt in np.arange(1/self.n, 1, 1/self.n):
                expct_judgmt = 0
                expct_judgmt_prev = 0
                if judgmt == 1/self.n:
                    expct_judgmt += Beta(judgmt) * w
                else:
                    expct_judgmt += Beta(judgmt) * w
                    expct_judgmt_prev   += Beta(judgmt - 1/self.n) * w
                bin_probs[bin_count] += expct_judgmt - expct_judgmt_prev
                bin_count += 1
        return bin_probs