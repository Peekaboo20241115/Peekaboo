from run_single_attack import run_single_attack
import os
import pickle
import tqdm
import random
import numpy as np
def test_random_day(k, is_fvp, Cycles_number, dataset, test_times):
    print("---------k---------:", k)
    print("---------dataset---------:", dataset)
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/padding"):
        os.mkdir("./results/padding")
    save_dir = "./results/padding"

    for cycles_number in Cycles_number:
        leakage_params={}
        leakage_params["cycles_number"] = cycles_number
        leakage_params["observed_timeslot_number_per_cycle"]=[1 for _ in range(leakage_params["cycles_number"])]
        leakage_params["unobserved_timeslot_num_per_cycle"]=[9]*leakage_params["cycles_number"]
        leakage_params["observed_query_number_per_timeslot"]=10000
        leakage_params["is_fvp"]=is_fvp

        dataset_params = {}
        dataset_params["dataset_name"]=dataset
        dataset_params["deleted_email_percent"]=0.1
        if dataset == "lucene":
            dataset_params["storage_time_limit"]=365 * 3
        else:
            dataset_params["storage_time_limit"]=365
        dataset_params["kws_universe_size"]=500
        dataset_params["kws_extraction"]="sorted"

        countermeasure_params = {}
        countermeasure_params['name'] = 'fvp_padding'
        countermeasure_params['padding_k'] = k

        Acc_Jigsaw_plus = []
        Acc_SAP_plus = []
        Acc_FMA = []
        Acc_SP_Jigsaw_plus = []
        Acc_SP_SAP_plus = []
        for i in tqdm.tqdm(range(test_times)):
##################Jigsaw+################        

            attack_params = {}
            attack_params["attack_name"]="Jigsaw+"
            attack_params["need_deduce_sp"]=True
            attack_params["need_cooccurrence"]=True
            attack_params["is_truth"] = False
            attack_params["alpha"]= 0.5
            attack_params["beta"]= 0.7
            if is_fvp == True:
                attack_params["BaseRec"] = 15
                attack_params["ConfRec"] = 5
            deduce_sp_params={}
            deduce_sp_params["delta"]=0.95
            deduce_sp_params["p_q_threshold"]=0.85
            deduce_sp_params["layer_match"] = 5
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params, countermeasure_params)
            print("Jigsaw+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc_Jigsaw_plus.append(acc)
        
##################Sap+#######################
            attack_params = {}
            attack_params["attack_name"]="SAP+"
            attack_params["need_deduce_sp"]=True
            attack_params["need_cooccurrence"]=False
            attack_params["is_truth"] = False
            attack_params["alpha"]= 0.5
           
            deduce_sp_params={}
            deduce_sp_params["delta"]=0.95
            deduce_sp_params["p_q_threshold"]=0.85
            deduce_sp_params["layer_match"] = 5
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params, countermeasure_params)
            print("SAP+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc_SAP_plus.append(acc)
        
#################################FMA############
            attack_params = {}
            attack_params["attack_name"]="FMA"
            attack_params["delta"] = 0.95
            attack_params["need_deduce_sp"]=False
            attack_params["need_cooccurrence"]=False
            attack_params["is_truth"] = False
            deduce_sp_params={}
            deduce_sp_params["delta"]=0.95
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params, countermeasure_params)
            print("FMA, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc_FMA.append(acc)
        
#######################SP&Jigsaw+############
            attack_params = {}
            attack_params["attack_name"]="Jigsaw+"
            attack_params["need_deduce_sp"]=True
            attack_params["need_cooccurrence"]=True
            attack_params["is_truth"] = True
            attack_params["alpha"]= 0.5
            attack_params["beta"]= 0.9
            if is_fvp == True:
                attack_params["BaseRec"] = 15
                attack_params["ConfRec"] = 5
            deduce_sp_params={}
            deduce_sp_params["delta"]=0.95
            deduce_sp_params["p_q_threshold"]=0.95
            deduce_sp_params["layer_match"] = 5
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params, countermeasure_params)
            print("SP&Jigsaw+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc_SP_Jigsaw_plus.append(acc)
        
# ###################SP&SAP+######################
            attack_params = {}
            attack_params["attack_name"]="SAP+"
            attack_params["need_deduce_sp"]=True
            attack_params["need_cooccurrence"]=False
            attack_params["is_truth"] = True
            attack_params["alpha"]= 0.5
           
            deduce_sp_params={}
            deduce_sp_params["delta"]=0.95
            deduce_sp_params["p_q_threshold"]=0.95
            deduce_sp_params["layer_match"] = 5
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
            print("SPSAP+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc_SP_SAP_plus.append(acc)


        with open("./results/padding/Jigsaw_plus_"+dataset+"_padding_k_"+str(k)+".pkl","wb") as f:
            pickle.dump(Acc_Jigsaw_plus,f)
        with open("./results/padding/SAP_plus_"+dataset+"_padding_k_"+str(k)+".pkl","wb") as f:
            pickle.dump(Acc_SAP_plus,f)
        with open("./results/padding/FMA_"+dataset+"_padding_k_"+str(k)+".pkl","wb") as f:
            pickle.dump(Acc_FMA,f)
        with open("./results/padding/SPJigsaw_plus_"+dataset+"_padding_k_"+str(k)+".pkl","wb") as f:
            pickle.dump(Acc_SP_Jigsaw_plus,f)
        with open("./results/padding/SPSAP_plus_"+dataset+"_padding_k_"+str(k)+".pkl","wb") as f:
            pickle.dump(Acc_SP_SAP_plus,f)


if __name__ == "__main__":
    k_list = [1,100,200,500,1000]
    Cycles_number = [5]
    for k in k_list:
        test_random_day(k, True, Cycles_number, 'enron', 5)
        test_random_day(k, True, Cycles_number, 'lucene', 5)
