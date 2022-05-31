import numpy as np
import itertools
from bayesian_learner import BayesianLearner


def amortize_parent_trial(parent_belief_t2, parent_judgmt_t1, trial_prob = {1: 1/20, 0: 1/20, 'no': 9/10}, tau = 10):
    post_trial_probs = {1: 0, 0: 0, 'no': 0} 
    for e, p in zip(parent_belief_t2['comb'], parent_belief_t2['probs']):
        for k, v in trial_prob.items():
            rev_prior = p * v
            new_obs = k
            if k == 1:
                BL = BayesianLearner(1 + (e.count(1) + 1) * rev_prior, 1 + e.count(0) * rev_prior, 7)
            elif k == 0:
                BL = BayesianLearner(1 + e.count(1) * rev_prior, 1 + (e.count(0) + 1) * rev_prior, 7)
            elif k == 'no':
                BL = BayesianLearner(1 + e.count(1) * rev_prior, 1 + e.count(0) * rev_prior, 7)
            BL.simple_update(k)
            bins = BL.discrete_beta()
            bins = BL.get_loss(bins)
            bins = BL.softmax(bins, tau)
            ll = bins[int(parent_judgmt_t1 + 3)]
            post_trial_probs[k] += ll * rev_prior    
    post_trial_probs = [prob/sum(list(post_trial_probs.values())) for prob in list(post_trial_probs.values())]
    return {'comb': [[1], [0], ['no']], 'probs': post_trial_probs}

def generate_structures(n_agents, incoming_edges=True, zero_diagonal=True): 
    outgoing_edges = list(np.array(i) for i in itertools.product([0,1], repeat=n_agents))
    structures = list(np.array(i) for i in itertools.product(outgoing_edges, repeat=n_agents))
    if incoming_edges:        
        structures = [structure for structure in structures if sum(structure[1:,0]) == n_agents-1]
    if zero_diagonal:
        structures = [structure for structure in structures if sum(np.diagonal(structure)) == 0]
    return structures

def get_expected_loss(bin_probs):
    # function returns expected lost given belief for n_choices 
    n_choices = len(bin_probs)
    ground_truth = np.array(np.arange(n_choices))
    response = np.array(np.arange(n_choices))
    belief = bin_probs 
    loss = np.zeros((n_choices, n_choices))
    expected_loss = np.zeros((n_choices))
    for i in range(n_choices):
        loss[:,i] = (ground_truth - i)**2
        expected_loss[i] = np.sum([l * p for l, p in zip(loss[:,i], belief)])
    return -expected_loss

