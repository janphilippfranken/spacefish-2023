import numpy as np

import env
import agent
import model

def simulate(n_trials, n_agents, structure, a_obs, b_obs, c_obs, emp_tau, sticky_weight, method='fullpost'):
    environment = env.Env(n_trials, n_agents, structure=structure) 
    A = agent.Agent(agent_id=0, prior_a=1, prior_b=1, n_choices=7, possible_evidence=[1, -1, 0], evidence_probs=np.array([1/3, 1/3, 1/3]), reliability=10, empirical_tau=1, sticky_weight=None, start_trial=0, fixed_obs=a_obs) 
    B = agent.Agent(agent_id=1, prior_a=1, prior_b=1, n_choices=7, possible_evidence=[1, -1, 0], evidence_probs=np.array([1/3, 1/3, 1/3]), reliability=10, empirical_tau=1, sticky_weight=None, start_trial=0, fixed_obs=b_obs) 
    C = agent.Agent(agent_id=2, prior_a=1, prior_b=1, n_choices=7, possible_evidence=[1, -1, 0], evidence_probs=np.array([1/3, 1/3, 1/3]), reliability=10, empirical_tau=1, sticky_weight=None, start_trial=0, fixed_obs=c_obs) 
    for i in range(n_trials):
        Acom = A.step(environment, method)
        Bcom = B.step(environment, method)
        Ccom = C.step(environment, method)
        environment.update_env(trial=Acom[0], agent_id=Acom[1], observation=Acom[2], communication=Acom[3])
        environment.update_env(trial=Bcom[0], agent_id=Bcom[1], observation=Bcom[2], communication=Bcom[3])
        environment.update_env(trial=Ccom[0], agent_id=Ccom[1], observation=Ccom[2], communication=Ccom[3])
    return A.marginal_inferred_obs, B.marginal_inferred_obs, C.marginal_inferred_obs, environment.communications