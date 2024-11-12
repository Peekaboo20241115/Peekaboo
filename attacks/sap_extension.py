import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian


class Sapattacker:
    def __init__(self, sim_kw_trend, real_td_trend, sim_v, real_v, td_num_per_timeslot, real_doc_num_per_timeslot, sim_doc_num_per_timeslot, alpha = 0.5, countermeasure = None):
        # normlize the frequency

        self.sim_kw_trend = sim_kw_trend
        column_sum = np.sum(real_td_trend, axis=0)
        self.real_td_trend = real_td_trend / column_sum

        # normalize the volume
        self.sim_v = sim_v / sim_doc_num_per_timeslot
        self.real_v = real_v
        

        self.td_num_per_timeslot = td_num_per_timeslot
        self.real_doc_num_per_timeslot = real_doc_num_per_timeslot
        self.alpha = alpha
        self.countermeasure = countermeasure
        self.tdid_2_kwsid = {}

    def attack(self):
        Cf = self.builtCf()
        Cv = self.builtCv()
        C_matrix = self.alpha * Cf + (1 - self.alpha) * Cv
        row_ind, col_ind = hungarian(C_matrix)
        td_set = set(col_ind)
        for td, kw in zip(col_ind, row_ind):
            self.tdid_2_kwsid[td] = kw
        while len(td_set) < len(C_matrix[0]):
            for td in col_ind:
                C_matrix[:,td] = 999999999999
            row_ind, col_ind = hungarian(C_matrix)
            td_set.update(set(col_ind))
            for td, kw in zip(col_ind, row_ind):
                if td not in self.tdid_2_kwsid:
                    self.tdid_2_kwsid[td] = kw
        return self.tdid_2_kwsid

    def builtCf(self):
        log_c_matrix = np.zeros((len(self.sim_kw_trend), len(self.real_td_trend)))
        for i in range(len(self.sim_kw_trend[0])):
            probabilities = self.sim_kw_trend[:, i].copy()
            probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100
            td_num_per_timeslot = np.array([self.td_num_per_timeslot]*len(self.real_td_trend))
            temp = (td_num_per_timeslot * self.real_td_trend[:, i]) * np.log(np.array([probabilities]).T)
            log_c_matrix = log_c_matrix + temp
        return -log_c_matrix
    
    def builtCv(self):
        if self.countermeasure == None:
            log_prob_matrix = compute_log_binomial_probability_matrix(self.real_doc_num_per_timeslot, self.sim_v, self.real_v) 
        cost_vol = - log_prob_matrix
        return cost_vol

def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    This code is based on https://github.com/simon-oya/USENIX21-sap-code/blob/master/attacks.py
    It computes the C_v matrix of maltiple windows.
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities|x|observations| matrix with the log binomial probabilities
    """
    num_interval = len(ntrials)
    probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100
    log_binom_term = np.array([[_log_binomial(ntrials[t], obs / ntrials[t]) for obs in observations[:, t]] for t in range(num_interval)])  # ROW TERM 
    log_binom_term = np.sum(log_binom_term, axis = 0)
    log_matrix = log_binom_term
    for t in range(num_interval):
        column_term = np.array([np.log(probabilities[:, t]) - np.log(1 - np.array(probabilities[:, t]))]).T  # COLUMN TERM 
        last_term = np.array([ntrials[t] * np.log(1 - np.array(probabilities[:, t]))]).T  # COLUMN TERM 
        log_matrix = log_matrix + np.array(observations[:, t]) * column_term + last_term
    return log_matrix


def _log_binomial(n, a):
    #  Computes an approximation of log(binom(n, n*a)) for a < 1
    if a == 0 or a == 1:
        return 0
    elif a < 0 or a > 1:
        raise ValueError("a cannot be negative or greater than 1 ({})".format(a))
    else:
        entropy = -a * np.log(a) - (1 - a) * np.log(1 - a)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * a * (1 - a))
