import numpy as np
import time
from simulate_leakage_no_update import simulate_leakage
from attacks.sap_extension import Sapattacker as Sap
from attacks.FMA import FMA
from attacks.jigsaw_extension import Jigsawattacker as Jigsaw
from attacks.deduce_sp_multi import Deduce_sp
from attacks.deduce_sp_truth import Deduce_sp_truth
from utils import pad_zero, cal_accuracy, cal_ARI_multi, cal_accuracy_FMA, cal_accuracy_part, cal_accuracy_RSP

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
        dataset_path = './datasets/newenron-full.pkl'
        dataset_path = './datasets/newenron-full.pkl'
        begin_data = "1 Feb 2000 00:00:00 +0000"
    elif dataset_name == "lucene":
        dataset_path = './datasets/newlucene.pkl'
        dataset_path = './datasets/newlucene.pkl'
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

    # use real client information to get F and V
    # F = np.zeros((kws_universe_size, sum(observed_timeslot_number_per_cycle)))
    # V = np.zeros((kws_universe_size, sum(observed_timeslot_number_per_cycle)))
    
    # for i_window in range(len(observed_timeslot_number_per_cycle)):
    #     for j_timeslot in range(observed_timeslot_number_per_cycle[i_window]):
    #         sum_timeslot = sum(observed_timeslot_number_per_cycle[:i_window]) + j_timeslot
    #         for k_query in range(observed_query_number_per_timeslot):
    #             kw = query_id_multi_window_list[i_window][j_timeslot][k_query]
    #             F[kw][sum_timeslot] += 1

    #             volume = len(set(query_doc_multi_window_list[i_window][j_timeslot][k_query]))
    #             if V[kw][sum_timeslot] < volume:
    #                 V[kw][sum_timeslot] = volume 
    #plt.plot(V_sim[:,0])

    attack_name = attack_params["attack_name"]
    
    need_cooccurrence = attack_params["need_cooccurrence"]
    if "need_deduce_sp" in attack_params:
        need_deduce_sp = attack_params["need_deduce_sp"]
        if need_deduce_sp == True:
            deduce_sp_params = attack_params["deduce_sp_params"]
            is_truth = attack_params["is_truth"]
            if is_truth:
                deduce_sp_attacker = Deduce_sp_truth(query_id_multi_window_list, kws_universe_size, query_doc_multi_window_list, deduce_sp_params["delta"], deduce_sp_params["p_q_threshold"],deduce_sp_params["layer_match"], is_fvp)
            else:
                deduce_sp_attacker = Deduce_sp(query_doc_multi_window_list, deduce_sp_params["delta"], deduce_sp_params["p_q_threshold"],deduce_sp_params["layer_match"], is_fvp)
            deduce_sp_attacker.deduce_sp()
            M = deduce_sp_attacker.M 
            # M is a list of dicts
            # the dicts maps a frozenset of a gourp of queries to its window id
            ari_multi_window, ri_multi_window = cal_ARI_multi(M, query_id_multi_window_list)
            #print("The ARI of Deduced SP:",ari_multi_window)

            #get F_sim and V_sim
            F_real = np.zeros((len(M),sum(observed_timeslot_number_per_cycle)))
            V_real = np.zeros((len(M),sum(observed_timeslot_number_per_cycle)))
            # x = np.zeros(len(M))
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
                        
                #x[i] = query_id_multi_window_list[window_id][query//observed_query_number_per_timeslot][query%observed_query_number_per_timeslot]
                
                F_real[i] = f_real #pad_zero(f_real)
                V_real[i] = v_real #pad_zero(v_real)         
            # plt.scatter(x,V_real[:,0])
            # v_sim_re = np.array([V_sim[int(x[i])][0] for i in range(len(x))])
            # plt.scatter(x,v_sim_re)
            # plt.show()
    
    if "need_cooccurrence" in attack_params:
        need_cooccurrence = attack_params["need_cooccurrence"]
        if need_cooccurrence == True:
    ################################################################
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

                # last_ID = []
                # for i in range(cycles_number):
                #     file_number = len(ID_real_multi_window[i][0])
                #     if i in window_id_list:
                #         frozen_group = group_list[window_id_list.index(i)]
                #         index = groups_multi_window_frozen[i].index(frozen_group)
                #         last_ID = ID_real_multi_window[i][index]
                #         break
                for i in range(cycles_number):
                    file_number = len(ID_real_multi_window[i][0])
                    if i in window_id_list:
                        frozen_group = group_list[window_id_list.index(i)]
                        index = groups_multi_window_frozen[i].index(frozen_group)
                        ID_real_multi_window_with_zeros[i].append(ID_real_multi_window[i][index])
                        #last_ID = ID_real_multi_window[i][index]
                    else:
                        #TODO upgrade this
                        ID_real_multi_window_with_zeros[i].append([0.00000000001]*file_number)
            
            real_M = []
            for i in range(cycles_number):
                temp = np.array(ID_real_multi_window_with_zeros[i])
                real_M.append(np.matmul(temp,temp.T)/len(temp[0]))


    ########################################################################

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
        
        re_result = {result[i]:i for i in result.keys()}
        sorted_re = sorted(list(re_result.keys()))
        # test_v = [V_real[re_result[i]] for i in sorted_re]
        # test_v_2 = [V_sim[i] for i in sorted_re]
        # plt.scatter(sorted_re,test_v)
        # plt.scatter(sorted_re,test_v_2,c="r")
        # plt.show()
        acc = cal_accuracy(M, query_id_multi_window_list, result, observed_query_number_per_timeslot)
        # # use search pattern that isn't deduced to attack
        # sap = Sap(F_sim, F, V_sim, V, observed_query_number_per_timeslot, client_doc_number_list, attacker_doc_number_list ,alpha= alpha)
        # result = sap.attack()
        # acc_RSP = cal_accuracy_RSP(query_id_multi_window_list, F, result)
        print(acc)
    elif attack_name == 'FMA':
        delta = 0.9
        if "delta" in attack_params:
            delta = attack_params["delta"]
        # groups_multi_window = deduce_sp_attacker.groups_multi_window
        begin_time = time.time()
        fma = FMA(query_doc_multi_window_list, F_sim, observed_query_number_per_timeslot, observed_timeslot_number_per_cycle, delta, is_fvp)
        init_time = time.time()
        print("init_time:", init_time - begin_time)
        
        result = fma.attack()
        
        acc = cal_accuracy_FMA(query_id_multi_window_list, result, observed_query_number_per_timeslot, observed_timeslot_number_per_cycle)
        ari_multi_window = 0
        print(acc)
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

        # result = jigsaw.tdid_2_kwsid_step1
        # re_result = {result[i]:i for i in result.keys()}
        # sorted_re = sorted(list(re_result.keys()))
        # test_v = [V_real[re_result[i]] for i in sorted_re]
        # test_v_2 = [V_sim[i] for i in sorted_re]
        # plt.scatter(sorted_re,test_v)
        # plt.scatter(sorted_re,test_v_2,c="r")
        # plt.show()

        acc = cal_accuracy_part(M, query_id_multi_window_list, jigsaw.tdid_2_kwsid_step1, observed_query_number_per_timeslot)
        print("step1 acc:",acc)
        jigsaw.attack_step_2()
        acc = cal_accuracy_part(M, query_id_multi_window_list, jigsaw.tdid_2_kwsid_step2, observed_query_number_per_timeslot)
        print("step2 acc:",acc)
        result = jigsaw.attack_step_3()
        acc = cal_accuracy(M, query_id_multi_window_list, result, observed_query_number_per_timeslot)
        print("step3 acc:",acc)
    # print("len(M):",len(M))
    return acc, ari_multi_window





if __name__ == '__main__':
    
    Acc_Matrix = []
    Ari_Matrix = []
    cycle_number = 3 
    for beta in [0.9]:
        test_time = 2 
        for i in range(test_time):
            Acc = []
            Ari = []
            
            alpha=0.5 
            beta = beta 
                    
            leakage_params={}
            leakage_params["cycles_number"] = cycle_number
            leakage_params["observed_timeslot_number_per_cycle"]=[1]*leakage_params["cycles_number"]
            leakage_params["unobserved_timeslot_num_per_cycle"]=[16]*leakage_params["cycles_number"]
            leakage_params["observed_query_number_per_timeslot"]=500
            leakage_params["is_fvp"]=True

            dataset_params = {}
            dataset_params["dataset_name"]="enron"
            dataset_params["deleted_email_percent"]=0.1
            dataset_params["storage_time_limit"]=365
            dataset_params["kws_universe_size"]=500
            dataset_params["kws_extraction"]="sorted"

            attack_params = {}
            attack_params["attack_name"]="FMA"
            attack_params["need_deduce_sp"]=False
            attack_params["need_cooccurrence"]=False
            attack_params["is_truth"] = False
            attack_params["alpha"]= alpha
            attack_params["beta"]= beta
            deduce_sp_params={}
            deduce_sp_params["delta"]=0.95
            deduce_sp_params["p_q_threshold"]=0.95
            deduce_sp_params["layer_match"] = 5
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari = run_single_attack(leakage_params,dataset_params,attack_params)
            print("alpha:",alpha," beta:",beta," p_q_threshold:",deduce_sp_params["p_q_threshold"],"    acc:",acc,"  ari:",ari)
            Acc.append(acc)
            Ari.append(ari)
            
        print(cycle_number,":  Acc:",Acc," Ari:",Ari)
        Acc_Matrix.append(Acc)
        Ari_Matrix.append(Ari)
    print(dataset_params["dataset_name"],Acc_Matrix,Ari_Matrix)
