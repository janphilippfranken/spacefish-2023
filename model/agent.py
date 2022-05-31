import abc
import itertools
import copy
import numpy as np

import model


state = model.Beta(prior_a=1, prior_b=1, n_choices=7)

# todo: update recursive filter function
class AgentTemplate(metaclass=abc.ABCMeta):
     
    @abc.abstractmethod
    def invert_model(self) -> dict: #n arg
        """Bayesian model inversion using full set of permutations.
          Args:

          Returns:
        """
        raise NotImplementedError()
        
    @abc.abstractmethod
    def sequential_filter(self) -> dict:
        """Bayesian model inversion using filtering.
          Args:

          Returns:
        """
        raise NotImplementedError()
    
    @abc.abstractmethod # needs to be implemented properly
    def recursive_filter(self) -> dict:
        """Bayesian model inversion using recursive filtering to correct for dependencies.
          Args:

          Returns:
        """
        raise NotImplementedError()
        
    @abc.abstractmethod
    def step(self, method: str) -> state:
        """Make one step in time and update State using method.
          Args:
              method:
               - 'fullPost'
               - 'filter'
               
          Returns:
        """
        raise NotImplementedError()
        
    
class Agent(AgentTemplate): # todo: split into two classes?
    
    def __init__(self, agent_id, prior_a=1, prior_b=1, n_choices=7, possible_evidence=[1, -1, 0], evidence_probs=np.array([.25, .25, .5]), reliability=10, empirical_tau=1, sticky_weight=None, start_trial=0, fixed_obs=[]): 
        """Initialize an agent instance.
        
        Args:
        """
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.n = n_choices
        self.id = agent_id
        self.e = possible_evidence
        self.p = evidence_probs
        self.reliability = reliability
        self.emp_tau = empirical_tau
        self.sticky_weight = sticky_weight
        self.t = start_trial
        self.fixed_obs = fixed_obs
        self.obs = []
        self.coms = []
        self.com_ps = []
        self.marginal_inferred_obs = {}
        self.State = model.Beta(prior_a=self.prior_a, prior_b=self.prior_b, n_choices=self.n)

    def invert_model(self, coms: np.array):
        """See base class."""
        p_dict = dict(zip(self.e, self.p))
        inferred_obs = {}
        for com_idx in range(0, len(coms)):
            permutations = [list(i) for i in itertools.product(self.e, repeat=com_idx+1)]
            post_probs = np.zeros(len(permutations))
            for perm_idx, perm in enumerate(permutations): 
                beta_state = model.Beta(prior_a=self.prior_a, prior_b=self.prior_b, n_choices=self.n)
                prior = 1
                ll = 1
                for perm_com_idx, evidence in enumerate(perm): 
                    prior *= p_dict[evidence]
                    beta_state.simple_update(evidence)
                    com_probs = beta_state.discrete_beta()
                    com_probs = beta_state.get_loss(com_probs)
                    com_probs = beta_state.softmax(com_probs, self.reliability)
                    ll *= com_probs[int(coms[perm_com_idx])]     
                post_probs[perm_idx] = prior * ll
            post_probs /= np.sum(post_probs)
            inferred_obs[com_idx] = beta_state.compress_evidence(permutations, post_probs)  
        return inferred_obs
    
    def recursive_filter(self, coms, selfid, parents):  # this function needs to be replaced / written again before revision
        """See base class."""
        par_states = []        
        curr_par = parents[selfid]
        counter = 0 
        par_counter = 0
        forward = 1
        while counter < len(curr_par):
            # if counter == 0 and len(curr_par) == len(parents[curr_par[0]]) or counter == 0 and len(curr_par) == 1:
            #     break
            par_states.append(self.recursive_filter(coms, curr_par[counter-1], parents))
            counter += 1    
        p_dict = dict(zip(np.array([1, -1, 0]), np.array([1/3, 1/3, 1/3])))
        inferred_states = {}
        for j_idx in range(len(coms[:,selfid])):
            inferred_states[j_idx] = {}
            post_probs = np.zeros((3))
            e_idx = 0
            for e in list(p_dict.keys()):
                if j_idx == 0:
                    beta_state = model.Beta(prior_a=1, prior_b=1, n_choices=7)
                else:
                    par_a = 0
                    par_b = 0
                    for par in par_states:
                        par_a += par[j_idx-1][1]
                        par_b += par[j_idx-1][0]
                    beta_state = model.Beta(prior_a=1+inferred_states[j_idx-1][1]+par_a,prior_b=1+inferred_states[j_idx-1][0]+par_b, n_choices=7)
                ll = 1
                e_prob = p_dict[e] 
                beta_state.simple_update(e) 
                com_probs = beta_state.discrete_beta()
                com_probs = beta_state.get_loss(com_probs)
                com_probs = beta_state.softmax(com_probs, 10)
                ll *= com_probs[int(coms[:,selfid][j_idx])]     
                post_probs[e_idx] = ll * e_prob 
                e_idx += 1    
            post_probs /= np.sum(post_probs)
            if j_idx == 0:
                inferred_states[j_idx][1] = post_probs[0] 
                inferred_states[j_idx][0] = post_probs[1] 
            else:        
                inferred_states[j_idx][1] = inferred_states[j_idx - 1][1] + post_probs[0] 
                inferred_states[j_idx][0] = inferred_states[j_idx - 1][0] + post_probs[1] 
        return inferred_states
    
    def sequential_filter(self, coms: np.array):
        """See base class."""
        p_dict = dict(zip(self.e, self.p))
        inferred_states = {}
        for j_idx in range(len(coms)):
            inferred_states[j_idx] = {}
            post_probs = np.zeros((3))
            e_idx = 0
            for e in list(p_dict.keys()):
                if j_idx == 0:
                    beta_state = model.Beta(prior_a=self.prior_a, prior_b=self.prior_b, n_choices=self.n)
                else:
                    beta_state = model.Beta(prior_a=self.prior_a+inferred_states[j_idx-1][1],prior_b=self.prior_b+inferred_states[j_idx-1][0], n_choices=self.n)
                ll = 1
                e_prob = p_dict[e] 
                beta_state.simple_update(e) 
                com_probs = beta_state.discrete_beta()
                com_probs = beta_state.get_loss(com_probs)
                com_probs = beta_state.softmax(com_probs, self.reliability)
                ll *= com_probs[int(coms[j_idx])]     
                post_probs[e_idx] = ll * e_prob 
                e_idx += 1    
            post_probs /= np.sum(post_probs)
            if j_idx == 0:
                inferred_states[j_idx][1] = post_probs[0] 
                inferred_states[j_idx][0] = post_probs[1] 
            else:        
                inferred_states[j_idx][1] = inferred_states[j_idx - 1][1] + post_probs[0] 
                inferred_states[j_idx][0] = inferred_states[j_idx - 1][0] + post_probs[1] 
        return inferred_states
    
    def step(self, env, method):
        """See base class."""
        if self.fixed_obs == []:
            self.sample = np.random.choice([1, -1, 0], p=self.p)  
        else:
            self.sample = self.fixed_obs[self.t]
        self.State.simple_update(self.sample) 
        self.obs.append(self.sample)
        self.history = env.get_env(self.t, self.id)[1]        
        if method == 'fullpost':
            self.inferred_obs = {} # todo: perhaps change name, this is obs for method: 'fullPost'
            for i in range(self.history.shape[1]): 
                communications = self.history[:,i]
                self.inferred_obs[i] = self.invert_model(communications) 
            combinations = []
            probs = []
            if self.t > 0 and len(self.inferred_obs.keys()) > 0:
                for parent in self.inferred_obs.keys():
                    combinations.append(self.inferred_obs[parent][self.t-1]['comb'])
                    probs.append(self.inferred_obs[parent][self.t-1]['probs'])
                joint = self.State.get_joint(combinations, probs)
                joint = self.State.compress_evidence(joint['comb'], joint['probs'])
                communication_probs = self.State.get_expectation(self.obs, joint['comb'], joint['probs'])
                self.marginal_inferred_obs[self.t-1] =  self.State.get_marginal_inferred_obs(joint['comb'], joint['probs'])   
            else:
                communication_probs = self.State.discrete_beta()
            communication_probs = self.State.get_loss(communication_probs)
            communication_probs = self.State.softmax(communication_probs, self.emp_tau) 
            self.com_ps.append(communication_probs)
        if method == 'seqfilter':
            self.inferred_states = {} # todo: perhaps change name, this is obs for method: 'seq filter'
            for i in range(self.history.shape[1]): 
                communications = self.history[:,i]
                self.inferred_states[i] = self.sequential_filter(communications) 
            if self.t > 0 and len(self.inferred_states.keys()) > 0:
                inferred_a = 0
                inferred_b = 0
                for parent in self.inferred_states.keys():
                    inferred_a += self.inferred_states[parent][self.t-1][1]
                    inferred_b += self.inferred_states[parent][self.t-1][0]     
                state_update = model.Beta(prior_a=self.State.a+inferred_a, prior_b=self.State.b+inferred_b, n_choices=self.n)
                communication_probs = state_update.discrete_beta()   
                self.marginal_inferred_obs[self.t-1] = [state_update.a, state_update.b]
            else:
                communication_probs = self.State.discrete_beta()
            communication_probs = self.State.get_loss(communication_probs)
            communication_probs = self.State.softmax(communication_probs, self.emp_tau) 
            self.com_ps.append(communication_probs)
        if method == 'recfilter': # update this section
            self.inferred_states = {}
            communications = env.communications[:self.t,:]
            parents = env.get_env(self.t, self.id)[2]   
            for i in parents[self.id]:
                self.inferred_states[i] = self.recursive_filter(communications, i, parents)
            if self.t > 0 and len(self.inferred_states.keys()) > 0:
                inferred_a = 0
                inferred_b = 0
                for parent in self.inferred_states.keys():
                    inferred_a += self.inferred_states[parent][self.t-1][1]
                    inferred_b += self.inferred_states[parent][self.t-1][0]     
                state_update = model.Beta(prior_a=self.State.a+inferred_a, prior_b=self.State.b+inferred_b, n_choices=self.n)
                communication_probs = state_update.discrete_beta()   
                self.marginal_inferred_obs[self.t-1] = [state_update.a, state_update.b]
            else:
                communication_probs = self.State.discrete_beta()
            communication_probs = self.State.get_loss(communication_probs)
            communication_probs = self.State.softmax(communication_probs, self.emp_tau) 
            self.com_ps.append(communication_probs)  
        if self.sticky_weight is not None and self.t > 0: 
            communication_probs_sticky = []
            prev_judg = list(np.zeros(self.n)) 
            prev_judg[self.coms[self.t-1]] = 1 
            for p, j in zip(communication_probs, prev_judg):
                communication_probs_sticky.append((1 - self.sticky_weight) * p + self.sticky_weight * j)
            communication_probs = copy.deepcopy(communication_probs_sticky)
        communication = np.random.choice(np.arange(self.n), p=communication_probs)
        # communication = np.arange(self.n)[list(communication_probs).index(max(communication_probs))]
        self.coms.append(communication)
        self.t += 1
        return (self.t-1, self.id, self.sample, communication)