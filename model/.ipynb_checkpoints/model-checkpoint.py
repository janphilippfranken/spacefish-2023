from scipy.stats import beta
import numpy as np
import itertools

from utils import flat

# todo: implement base class
class Beta():
    
    def __init__(self, prior_a, prior_b, n_choices):
        """Standard beta-binomial updating model with some custom functionality.
          
          Args:
          
          Returns:
        """
        self.a = prior_a
        self.b = prior_b
        self.n = n_choices
        self.mu = self.a / (self.a + self.b)
        self.sigma = self.a * self.b / ((self.a + self.b)**2 * (self.a + self.b + 1))
    
    def simple_update(self, data):
        if data == 1:
            self.a += 1
        elif data == -1:
            self.b += 1
        return self.a, self.b 
    
    def increment_parameters(self, a, b):
        self.a += a
        self.b += b
        return self.a, self.b 
    
    def discrete_beta(self):
        betacdf = beta(self.a, self.b).cdf
        beta_bins = []
        beta_bins.append(betacdf(1 / self.n))
        for beta_bin in range(2, self.n + 1):
            beta_bins.append(betacdf(beta_bin / self.n) - betacdf((beta_bin - 1) / self.n))
        return beta_bins
    
    def get_loss(self, probs):
        ground_truth = np.array(np.arange(self.n))
        response = np.array(np.arange(self.n))
        loss = np.zeros((self.n, self.n))
        exp_loss = np.zeros((self.n))
        for i in range(self.n):
            loss[:,i] = (ground_truth - i)**2
            exp_loss[i] = np.sum([l * p for l, p in zip(loss[:,i], probs)])
        return -exp_loss
    
    def softmax(self, probs, tau):
        return np.exp(np.array(probs)*tau)/np.sum(np.exp(np.array(probs)*tau))
    
    def compress_evidence(self, evidence, probs):
        e_compressed = []
        p_compressed = []
        counts = [[e.count(1), e.count(-1)] for e in evidence]
        for e in counts:
            if evidence[counts.index(e)] not in e_compressed:       # only append if not already existing
                indxs = [i for i, x in enumerate(counts) if x == e] # getting duplicates 
                p_comb = 0
                for i in indxs:
                    p_comb += probs[i]
                p_compressed.append(p_comb)
                e_compressed.append(evidence[indxs[0]])
        return {'comb': e_compressed, 'probs': p_compressed}
    
    def get_joint(self, comb, probs):
        joint_comb = comb[0]
        joint_prob = np.array(probs[0])
        for i in range(1, len(comb)):
            joint_comb = list(list(e) for e in itertools.product(joint_comb, comb[i]))
            joint_prob = np.multiply.outer(joint_prob, probs[i])
        return {'comb': [flat(e) for e in joint_comb], 'probs': joint_prob.reshape(1, np.prod(joint_prob.shape))[0]} 
    
    def get_expectation(self, observations, evidence, weights, corr_weight = 1):
        bin_probs = np.zeros((self.n))
        a_obs = observations.count(1)
        b_obs = observations.count(-1)
        for e, w in zip(evidence, weights): 
            e = [int(i) for i in e if i != 0]
            Beta = beta(1 + a_obs + e.count(1) * corr_weight, 1 + b_obs + e.count(-1) * corr_weight).cdf
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
    
    def get_marginal_inferred_obs(self, evidence, weights):
        marginal_a = 0
        marginal_b = 0
        for e, w in zip(evidence, weights): 
            acount = e.count(1)
            bcount = e.count(-1)
            marginal_a += acount * w
            marginal_b += bcount * w 
        return {'a': marginal_a, 'b': marginal_b}