import itertools
import numpy as np


def generate_structures(n_agents, incoming_edges=True, zero_diagonal=True): 
    outgoing_edges = list(np.array(i) for i in itertools.product([0,1], repeat=n_agents))
    structures = list(np.array(i) for i in itertools.product(outgoing_edges, repeat=n_agents))
    if incoming_edges:        
        structures = [structure for structure in structures if sum(structure[1:,0]) == n_agents-1]
    if zero_diagonal:
        structures = [structure for structure in structures if sum(np.diagonal(structure)) == 0]
    return structures

def get_bias(a_vals, b_vals, n_sim, structures):
    bias_a = []
    bias_b = []    
    for struct in range(len(structures)):
        b_a = []
        b_b = []
        for sim in range(n_sim):
            base_a = a_vals[0][sim]
            base_b = b_vals[0][sim]
            a =  a_vals[struct][sim]
            b =  b_vals[struct][sim]
            if a < base_a:
                b_a.append(a-base_a)
            elif a >= base_a:
                b_a.append(a-base_a)
            if b < base_b:
                b_b.append(b-base_b)
            elif b >= base_b:
                b_b.append(b-base_b)
        bias_a.append(b_a)
        bias_b.append(b_b)
    return bias_a, bias_b

def flat(l):
    if l == []:
        return []
    if isinstance(l[0], list):
        return flat(l[0]) + flat(l[1:])
    else:
        return [l[0]] + flat(l[1:])