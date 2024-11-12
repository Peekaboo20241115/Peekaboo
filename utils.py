from sklearn import metrics
import numpy as np

def cal_ARI_single(groups_merge,query_all):
    #cal the ari and ri from the results of deduce sp single window
    labels_true = []
    kws_id_dic = {}
    for query_single_timeslot in query_all:
        for query in query_single_timeslot:
            if query not in kws_id_dic:
                kws_id_dic[query] = len(kws_id_dic)
            labels_true.append(kws_id_dic[query])
    
    labels_pred = []
    for group in groups_merge:
        labels_pred = labels_pred + [0]*len(group)
    group_count = 1
    for group in groups_merge:
        for query_id in group:
            labels_pred[query_id] = group_count
        group_count += 1
    for i in range(len(labels_pred)):
        if labels_pred[i] == 0: # not needed
            labels_pred[i] = group_count
            group_count += 1

    return metrics.adjusted_rand_score(labels_true, labels_pred),metrics.rand_score(labels_true, labels_pred)

def cal_ARI_multi(M,query_id_multi_window_list):
    #cal the ari and ri from the results of deduce sp multi window
    query_number_per_window_accumulated = [0]
    number_of_windows = len(query_id_multi_window_list)

    labels_true = []
    kws_id_dic = {}
    query_number = 0
    for query_id_single_window in query_id_multi_window_list:
        for query_id_single_timeslot in query_id_single_window:
            query_number += len(query_id_single_timeslot)
            for query in query_id_single_timeslot:
                if query not in kws_id_dic:
                    kws_id_dic[query] = len(kws_id_dic)
                labels_true.append(kws_id_dic[query])
        query_number_per_window_accumulated.append(query_number)
    
    labels_pred = [0]*query_number_per_window_accumulated[-1]
    group_count = 1
    for group_window_list in M:
        frozen_group_list = []
        window_id_list = []
        for group_window in group_window_list:
            frozen_group_list.append(group_window[0])
            window_id_list.append(group_window[1])
        # frozen_group_list = list(groups_window_dict.keys())
        # window_id_list = list(groups_window_dict.values())
        for i in range(len(frozen_group_list)):
            group = list(frozen_group_list[i])
            for q in group:
                labels_pred[q+query_number_per_window_accumulated[window_id_list[i]]] = group_count
        group_count += 1
    
    return metrics.adjusted_rand_score(labels_true, labels_pred),metrics.rand_score(labels_true, labels_pred)

def cal_accuracy(M,query_id_multi_window_list,result,observed_query_number_per_timeslot):
    #cal the accuracy
    #result maps the group in M to the guessed keyword
    total_count = 0
    correct_count = 0
    debug_count = 0
    for i in range(len(M)):
        group_window_list = M[i]
        guessed_keyword = result[i]

        group_list = []
        window_list = []
        for group_window in group_window_list:
            group_list.append(group_window[0])
            window_list.append(group_window[1])

        for j in range(len(group_list)):
            window_id = window_list[j]
            query_list = list(group_list[j])
            for query in query_list:
                debug_count += 1
                if query_id_multi_window_list[window_id][query//observed_query_number_per_timeslot][query%observed_query_number_per_timeslot]==guessed_keyword:
                    correct_count += 1
    for query_id_single_window_list in query_id_multi_window_list:
        for query_id_single_timeslot_list in query_id_single_window_list:
            total_count += len(query_id_single_timeslot_list)
    print(debug_count,total_count,correct_count)
    return correct_count/total_count


def cal_accuracy_FMA(query_id_multi_window_list, result, observed_query_number_per_timeslot, observed_timeslot_number_per_cycle):
    correct_count = 0
    total_count = 0
    for query, guessed_keyword in result.items():
        for i in range(len(observed_timeslot_number_per_cycle)):
            if query - observed_timeslot_number_per_cycle[i] * observed_query_number_per_timeslot < 0:
                window_id = i
                break
            query -= observed_timeslot_number_per_cycle[i] * observed_query_number_per_timeslot

        if query_id_multi_window_list[window_id][query // observed_query_number_per_timeslot][query % observed_query_number_per_timeslot] == guessed_keyword:
                correct_count += 1

    for query_id_single_window_list in query_id_multi_window_list:
        for query_id_single_timeslot_list in query_id_single_window_list:
            total_count += len(query_id_single_timeslot_list)

    return correct_count / total_count
