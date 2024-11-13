import numpy as np
import time
from simulate_leakage import simulate_leakage
from attacks.sap_extension import Sapattacker as Sap
from attacks.FMA import FMA
from attacks.jigsaw_extension import Jigsawattacker as Jigsaw
from attacks.infer_sp import Deduce_sp
from attacks.real_sp import Deduce_sp_truth
from utils import cal_accuracy, cal_ARI_multi, cal_accuracy_FMA

import tqdm
import matplotlib.pyplot as plt

def run_single_attack(
        leakage_params,
        dataset_params,
        attack_params,
        countermeasure_params={'name':None}
):
    
    # get leakage and auxillary data
    cycles_number = leakage_params["cycles_number"]
    observed_timeslot_number_per_cycle = leakage_params["observed_timeslot_number_per_cycle"]
    unobserved_timeslot_num_per_cycle = leakage_params["unobserved_timeslot_num_per_cycle"]
    observed_query_number_per_timeslot = leakage_params["observed_query_number_per_timeslot"]
    is_fvp = leakage_params["is_fvp"]

    dataset_name = dataset_params["dataset_name"]
    deleted_email_percent = dataset_params["deleted_email_percent"]
    storage_time_limit = dataset_params["storage_time_limit"]
    kws_universe_size = dataset_params["kws_universe_size"]
    kws_extraction = dataset_params["kws_extraction"]

    if dataset_name == "enron":
        dataset_path = './datasets/enron.pkl'
        begin_data = "1 Feb 2000 00:00:00 +0000"
    elif dataset_name == "lucene":
        dataset_path = './datasets/lucene.pkl'
        begin_data = "1 Feb 2002 00:00:00 +0000"
    else:
        assert False, "Invalid dataset_name"
    query_id_multi_window_list, query_doc_multi_window_list, F_sim_multi_window, V_sim_multi_window, C_sim_multi_window, attacker_doc_number_list = \
        simulate_leakage(observed_query_number_per_timeslot = observed_query_number_per_timeslot, \
                    observed_timeslot_number_per_cycle = observed_timeslot_number_per_cycle, \
                    unobserved_timeslot_num_per_cycle = unobserved_timeslot_num_per_cycle, \
                    cycles_number = cycles_number, deleted_email_percent = deleted_email_percent, \
                    storage_time_limit = storage_time_limit, kws_universe_size = kws_universe_size, \
                    dataset_path = dataset_path, is_fvp = is_fvp, \
                    begin_date = begin_data, kws_extraction = kws_extraction,  countermeasure_info=countermeasure_params, debug_info = False)
    
    # simulate attacks
    F_sim = []
    V_sim = []
    for F_sim_single_window in F_sim_multi_window:
        for F_sim_single_timeslot in F_sim_single_window:
            F_sim.append(F_sim_single_timeslot)
    F_sim = np.array(F_sim).T
    for V_sim_single_window in V_sim_multi_window:
        for V_sim_single_timeslot in V_sim_single_window:
            V_sim.append(V_sim_single_timeslot)
    V_sim = np.array(V_sim, dtype=float).T

    begin_time = time.time()
    attack_name = attack_params["attack_name"]
    
    need_cooccurrence = attack_params["need_cooccurrence"]
    if "need_deduce_sp" in attack_params:
        sp_deduce_time = time.time()
        need_deduce_sp = attack_params["need_deduce_sp"]
        if need_deduce_sp == True:
            deduce_sp_params = attack_params["deduce_sp_params"]
            is_truth = attack_params["is_truth"]
            if is_truth:
                deduce_sp_attacker = Deduce_sp_truth(query_id_multi_window_list, query_doc_multi_window_list)
            else:
                deduce_sp_attacker = Deduce_sp(query_doc_multi_window_list, deduce_sp_params["delta"], deduce_sp_params["p_q_threshold"],deduce_sp_params["layer_match"], is_fvp)
            deduce_sp_attacker.deduce_sp()
            M = deduce_sp_attacker.M 
            # M is a list of dicts
            # the dicts maps a frozenset of a gourp of queries to its window id
            ari_multi_window, ri_multi_window = cal_ARI_multi(M, query_id_multi_window_list)
            
            #get F_sim and V_sim
            F_real = np.zeros((len(M),sum(observed_timeslot_number_per_cycle)))
            V_real = np.zeros((len(M),sum(observed_timeslot_number_per_cycle)))
            for i in range(len(M)):
                group_window_list = M[i]
                group_list = []
                window_list = []
                for group_window in group_window_list:
                    group_list.append(group_window[0])
                    window_list.append(group_window[1])

                f_real = np.zeros((sum(observed_timeslot_number_per_cycle)))
                v_real = np.zeros((sum(observed_timeslot_number_per_cycle)))
                
                for j in range(len(group_list)):
                    group = group_list[j]
                    window_id = window_list[j]
                    
                    for query in group:
                        timeslot = query // observed_query_number_per_timeslot + sum(observed_timeslot_number_per_cycle[:window_id])
                        f_real[timeslot] = f_real[timeslot] + 1
                        volume = len(set(query_doc_multi_window_list[window_id][query // observed_query_number_per_timeslot][query % observed_query_number_per_timeslot]))
                        if v_real[timeslot] < volume:
                            v_real[timeslot] = volume 
                F_real[i] = f_real 
                V_real[i] = v_real        
        print("Deducing SP Time: ",time.time()-sp_deduce_time)

    if "need_cooccurrence" in attack_params:
        need_cooccurrence = attack_params["need_cooccurrence"]
        if need_cooccurrence == True:
            sim_M = C_sim_multi_window
            ID_real_multi_window = deduce_sp_attacker.ID_M

            groups_multi_window = deduce_sp_attacker.groups_multi_window
            groups_multi_window_frozen = []
            for groups_single_window in groups_multi_window:
                groups_single_window_frozen = []
                for group in groups_single_window:
                    groups_single_window_frozen.append(frozenset(group))
                groups_multi_window_frozen.append(groups_single_window_frozen)

            M = deduce_sp_attacker.M
            ID_real_multi_window_with_zeros = [[] for i in range(cycles_number)]
            for groups_id in range(len(M)):
                window_id_list = [group_window_id[1] for group_window_id in M[groups_id]]
                group_list = [group_window_id[0] for group_window_id in M[groups_id]]

                for i in range(cycles_number):
                    file_number = len(ID_real_multi_window[i][0])
                    if i in window_id_list:
                        frozen_group = group_list[window_id_list.index(i)]
                        index = groups_multi_window_frozen[i].index(frozen_group)
                        ID_real_multi_window_with_zeros[i].append(ID_real_multi_window[i][index])
                    else:
                        ID_real_multi_window_with_zeros[i].append([0.00000000001]*file_number)
            
            real_M = []
            for i in range(cycles_number):
                temp = np.array(ID_real_multi_window_with_zeros[i])
                real_M.append(np.matmul(temp,temp.T)/len(temp[0]))

    client_doc_number_list = []
    for query_doc_single_window in query_doc_multi_window_list:
        for query_doc_single_time_slot in query_doc_single_window:
            total_doc_set = set()
            for doc_set in query_doc_single_time_slot:
                total_doc_set.update(set(doc_set))
            client_doc_number_list.append(len(total_doc_set))
    

    
    if attack_name == "SAP+":
        alpha = 0.5
        if "alpha" in attack_params:
            alpha = attack_params["alpha"]
        
        # use search pattern that is deduced to attack
        sap = Sap(F_sim, F_real, V_sim, V_real, observed_query_number_per_timeslot, client_doc_number_list, attacker_doc_number_list ,alpha= alpha)
        result = sap.attack()
        time_overhead = time.time() - begin_time
        acc = cal_accuracy(M, query_id_multi_window_list, result, observed_query_number_per_timeslot)
        if is_truth:
            print("Sap+ with SP accuracy:", acc)
            print("Sap+ with SP time overhead:", time_overhead)
        else:
            print("Sap+ accuracy:", acc)
            print("Sap+ time overhead:", time_overhead)
    elif attack_name == 'FMA':
        
        begin_time = time.time()
        
        delta = 0.9
        if "delta" in attack_params:
            delta = attack_params["delta"]
        
        fma = FMA(query_doc_multi_window_list, F_sim, observed_query_number_per_timeslot, observed_timeslot_number_per_cycle, delta, is_fvp)
        
        result = fma.attack()
        time_overhead = time.time() - begin_time

        acc = cal_accuracy_FMA(query_id_multi_window_list, result, observed_query_number_per_timeslot, observed_timeslot_number_per_cycle)
        ari_multi_window = 0
        print("FMA accuracy", acc)
        print("FMA time overhead:", time_overhead)
    elif attack_name == 'Jigsaw+':
        alpha = 0.5
        beta = 0.8
        baseRec = 30
        confRec = 10
        if "alpha" in attack_params:
            alpha = attack_params["alpha"]
        if "beta" in attack_params:
            beta = attack_params["beta"]
        if "BaseRec" in attack_params:
            baseRec = attack_params["BaseRec"]
        if "ConfRec" in attack_params:
            confRec = attack_params["ConfRec"]
        jigsaw = Jigsaw(F_sim, F_real, V_sim, V_real, sim_M, real_M, alpha=alpha, beta=beta,baseRec=baseRec,confRec=confRec,sim_doc_num=attacker_doc_number_list,real_doc_num=client_doc_number_list,is_fvp=is_fvp)
        
        jigsaw.attack_step_1()
        jigsaw.attack_step_2()
        result = jigsaw.attack_step_3()
        time_overhead = time.time() - begin_time
        acc = cal_accuracy(M, query_id_multi_window_list, result, observed_query_number_per_timeslot)
        if is_truth:
            print("Jigsaw+ with SP accuracy:",acc)
            print("Jigsaw+ with SP time overhead:", time_overhead)
        else:
            print("Jigsaw+ accuracy:",acc)
            print("Jigsaw+ time overhead:", time_overhead)
    
    return acc, ari_multi_window, time_overhead

