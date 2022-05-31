# GENERIC BAYESIAN LEARNER 

# imports
import numpy as np
import itertools
from beta_update import BetaUpdate

class BayesianLearner(BetaUpdate):
    
    def __init__(self, prior_a, prior_b, n_choices):
        self.a = prior_a
        self.b = prior_b
        self.n = n_choices
        
    def kl_divergence(self, p, q):
        return np.sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))
    
    def js_divergence(self, p, q, a=0.5):
        return self.kl_divergence(p, q)*a + self.kl_divergence(q, p)*a

    def softmax(self, probs, tau):
        return np.exp(np.array(probs)*tau)/np.sum(np.exp(np.array(probs)*tau))
    
    def get_loss(self, probs):
        ground_truth = np.array(np.arange(self.n))
        response = np.array(np.arange(self.n))
        loss = np.zeros((self.n, self.n))
        exp_loss = np.zeros((self.n))
        for i in range(self.n):
            loss[:,i] = (ground_truth - i)**2
            exp_loss[i] = np.sum([l * p for l, p in zip(loss[:,i], probs)])
        return -exp_loss
    
    def compress_evidence(self, evidence, probs):
        e_compressed = []
        p_compressed = []
        counts = [[e.count(1) + e.count('1'), e.count(0) + e.count('0')] for e in evidence]
        for e in counts:
            if evidence[counts.index(e)] not in e_compressed:       # only append if not already existing
                indxs = [i for i, x in enumerate(counts) if x == e] # getting duplicates 
                p_comb = 0
                for i in indxs:
                    p_comb += probs[i]
                p_compressed.append(p_comb)
                e_compressed.append(evidence[indxs[0]])
        return {'comb': e_compressed, 'probs': p_compressed}

    def infer_evidence_two_agents(self, e1, p1, e2, p2):
        p_comb = np.outer(p1, p2).reshape(len(p1)*len(p2), 1) # combined evidence probs 
        p_comb = [p for sublist in p_comb for p in sublist]
        e_comb = list(list(i) for i in itertools.product(e1, e2))
        e_comb = np.array([list(single for sublist in comb for single in sublist) for comb in e_comb])
        return {'comb': e_comb, 'probs': p_comb}
    
    def correct_dependency(self, parent_t1_e, parent_t1_p, child_t1_e, child_t1_p):
        p_comb = np.outer(parent_t1_p, child_t1_p).reshape(len(parent_t1_p)*len(child_t1_p), 1) # combined evidence probs 
        p_comb = [p for sublist in p_comb for p in sublist]
        e_comb = list(list(i) for i in itertools.product(parent_t1_e, child_t1_e))
        e_comb = np.array([list(single for sublist in comb for single in sublist) for comb in e_comb])
        return {'comb': e_comb, 'probs': p_comb}
    
    def fit_predictions(self, parameters, predictions, judgments, sticky, learner):
        """fit predictions to judgments using MLE"""
        tau = parameters[0]
        weight = parameters[1]
        log_probs = []
        for trial in range(len(judgments)): 
            prediction = predictions[str(trial)]
            expected_loss  = self.get_loss(prediction) 
            probs = self.softmax(expected_loss, tau)
            judg = list(np.zeros(7)) 
            judg[int(judgments[trial] + 3)] = 1 
            if sticky and trial >= 1: 
                prev_judg = list(np.zeros(7)) 
                prev_judg[int(judgments[trial - 1] + 3)] = 1 
                combined_probs = []
                for p, j in zip(probs, prev_judg):
                    combined_probs.append((1 - weight) * p + weight * j)
                fitted_prob = np.sum([pred * resp for pred, resp in zip(combined_probs, judg)])  
            else:
                fitted_prob = np.sum([pred * resp for pred, resp in zip(probs, judg)])  
            log_probs.append(np.log(fitted_prob))
        return -np.sum(log_probs)
    
    def sticky_baseline_prediction(self, parameters, judgments):
        """fit sticky model predictions using MLE"""
        tau = parameters[0]
        log_probs = []
        prediction = np.repeat(1/7, 7)
        for trial in range(len(judgments)):  
            judg = list(np.zeros(7)) 
            judg[int(judgments[trial] + 3)] = 1  
            if trial >= 1:
                prev_judg = list(np.zeros(7)) 
                prev_judg[int(judgments[trial - 1] + 3)] = 1 
                prediction = prev_judg
            expected_loss = self.get_loss(prediction) 
            probs = self.softmax(expected_loss, tau)
            fitted_prob = np.sum([pred * resp for pred, resp in zip(probs, judg)])  
            log_probs.append(np.log(fitted_prob))
        return -np.sum(log_probs)
    
    def get_structure_corrected_joint(self, adjacency_matrix, b_p_compressed, c_p_compressed, b_p_amortized, c_p_amortized):
        '''corrects joint considering structure'''
        corr_weight = 1
        observation = ['no']
        if adjacency_matrix[0,1] == 0 and adjacency_matrix[0,2] == 0:
            observation = [1]
        if adjacency_matrix[1,2] == 0 and adjacency_matrix[2,1] == 0:
            joint_structure = self.infer_evidence_two_agents(b_p_compressed['comb'], b_p_compressed['probs'], c_p_compressed['comb'], c_p_compressed['probs']) 
        elif adjacency_matrix[1,2] == 1 and adjacency_matrix[2,1] == 0:
            joint_structure = self.correct_dependency(b_p_amortized['comb'], b_p_amortized['probs'], c_p_compressed['comb'], c_p_compressed['probs'])
        elif adjacency_matrix[1,2] == 0 and adjacency_matrix[2,1] == 1:
            joint_structure = self.correct_dependency(c_p_amortized['comb'], c_p_amortized['probs'], b_p_compressed['comb'], b_p_compressed['probs'])   
        elif adjacency_matrix[1,2] == 1 and adjacency_matrix[2,1] == 1:
            BC = self.correct_dependency(b_p_amortized['comb'], b_p_amortized['probs'], c_p_compressed['comb'], c_p_compressed['probs'])
            CB = self.correct_dependency(c_p_amortized['comb'], c_p_amortized['probs'], b_p_compressed['comb'], b_p_compressed['probs'])   
            BCB = self.infer_evidence_two_agents(BC['comb'], BC['probs'], CB['comb'], CB['probs'])
            BCB['comb'] = [list(i) for i in BCB['comb']]
            BCB = self.compress_evidence(BCB['comb'], BCB['probs'])
            joint_structure = BCB
            corr_weight = .5
        return {'joint': joint_structure, 'observation': observation, 'corr_weight': corr_weight}
    

    def fit_predictions_all_trials(self, parameters, predictions, judgments, sticky, learner):
        """fit predictions to all judgments using MLE"""
        tau = parameters[0]
        weight = parameters[1]
        log_probs = []
        for room in range(len(predictions)):
            for trial in range(len(judgments[0])): 
                prediction = predictions[room][str(trial)]
                BL = BayesianLearner(1,1,7)
                expected_loss  = BL.get_loss(prediction) 
                probs = BL.softmax(expected_loss, tau)
                judg = list(np.zeros(7)) 
                judg[int(judgments[room][trial] + 3)] = 1       
                if sticky and trial >= 1:
                    prev_judg = list(np.zeros(7)) 
                    prev_judg[int(judgments[room][trial - 1] + 3)] = 1 
                    combined_probs = []
                    for p, j in zip(probs, prev_judg):
                        combined_probs.append((1 - weight) * p + weight * j)
                    fitted_prob = np.sum([pred * resp for pred, resp in zip(combined_probs, judg)])  
                else:
                    fitted_prob = np.sum([pred * resp for pred, resp in zip(probs, judg)])  
                log_probs.append(np.log(fitted_prob))
        return -np.sum(log_probs)
    
    def sticky_baseline_prediction_all_trials(self, parameters, judgments):
        """fit sticky model predictions using MLE"""
        tau = parameters[0]
        log_probs = []
        for room in range(len(judgments)):
            for trial in range(10): 
                prediction = np.repeat(1/7, 7)
                judg = list(np.zeros(7)) 
                judg[int(judgments[room][trial] + 3)] = 1 
                if trial >= 1:
                    prev_judg = list(np.zeros(7)) 
                    prev_judg[int(judgments[room][trial - 1] + 3)] = 1 
                    prediction = prev_judg
                expected_loss = self.get_loss(prediction) 
                probs = self.softmax(expected_loss, tau)
                fitted_prob = np.sum([pred * resp for pred, resp in zip(probs, judg)])  
                log_probs.append(np.log(fitted_prob))
        return -np.sum(log_probs)