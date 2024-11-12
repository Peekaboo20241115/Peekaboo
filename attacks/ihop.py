import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from tqdm import tqdm
import sys
def compute_log_binomial_probability_matrix(probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities| x |observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    if any(probabilities > 0):
        probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    else:
        probabilities += 1e-10

    tmp = 1 - np.array(probabilities)
    tmp[tmp == 0] = 1e-10
    # column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    column_term = np.array([np.log(probabilities) - np.log(tmp)]).T
    # last_term = np.array([np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([np.log(tmp)]).T
    log_matrix = np.array(observations) * column_term + last_term
    return log_matrix


# we intend to match tokens from V_last to V_keyword
def QAP(V_token, V_keyword, n_iters, p_free):
    
    def _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
        cost_vol = -compute_log_binomial_probability_matrix(np.diagonal(V_keyword)[free_keywords], np.diagonal(V_token)[free_tags])
        for tag, kw in zip(fixed_tags, fixed_keywords):
            cost_vol -= compute_log_binomial_probability_matrix(V_keyword[kw, free_keywords], V_token[tag, free_tags])
        return cost_vol
    
    def compute_coef_matrix(free_keywords, free_tags, fixed_keywords, fixed_tags):
        return _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags)
    
    n_keyword = len(V_keyword)
    n_token = len(V_token)

    ground_truth_keyword, ground_truth_token = [], []

    unknown_keyword = [i for i in range(n_keyword) if i not in ground_truth_keyword]
    unknown_token = [i for i in range(n_token) if i not in ground_truth_token]

    c_matrix_original = compute_coef_matrix(unknown_keyword, unknown_token, ground_truth_token, ground_truth_keyword)
    try:
        row_ind, col_ind = hungarian(c_matrix_original)
    except ValueError as ve:
        print(ve)
        print(c_matrix_original)
    replica_predictions_for_each_token = {token: rep for token, rep in zip(ground_truth_token, ground_truth_keyword)}
    for j, i in zip(col_ind, row_ind):
        replica_predictions_for_each_token[unknown_token[j]] = unknown_keyword[i]

    n_free = int(p_free * len(unknown_token)) # unknown_token的选择

    assert n_free > 1
    for k in range(n_iters):
        random_unknown_token = list(np.random.permutation(unknown_token))
        free_token = random_unknown_token[:n_free]
        fixed_token = random_unknown_token[n_free:] + ground_truth_token
        fixed_keyword = [replica_predictions_for_each_token[token] for token in fixed_token]
        free_keyword = [rep for rep in unknown_keyword if rep not in fixed_keyword]

        c_matrix = compute_coef_matrix(free_keyword, free_token, fixed_keyword, fixed_token)

        row_ind, col_ind = hungarian(c_matrix)
        for j, i in zip(col_ind, row_ind):
            replica_predictions_for_each_token[free_token[j]] = free_keyword[i]

    return replica_predictions_for_each_token