from run_single_attack_no_update import run_single_attack
import os
import pickle
import tqdm
import sys
def test_query_number(Observed_query_number_per_timeslot,is_fvp,dataset,test_times):
    
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/no_update"):
        os.mkdir("./results/no_update")
    # save_dir = "./results/query_number"

    for observed_query_number_per_timeslot in Observed_query_number_per_timeslot:
        observe_days = 1
        leakage_params={}
        leakage_params["cycles_number"] = 5
        leakage_params["observed_timeslot_number_per_cycle"]=[observe_days]*leakage_params["cycles_number"]
        leakage_params["unobserved_timeslot_num_per_cycle"]=[20-observe_days]*leakage_params["cycles_number"]
        leakage_params["observed_query_number_per_timeslot"]=observed_query_number_per_timeslot
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

###################Jigsaw+################
        # Acc = []
        # for _ in tqdm.tqdm(range(test_times)):
        #     attack_params = {}
        #     attack_params["attack_name"]="Jigsaw+"
        #     attack_params["need_deduce_sp"]=True
        #     attack_params["need_cooccurrence"]=True
        #     attack_params["is_truth"] = False
        #     attack_params["alpha"]= 0.5
        #     attack_params["beta"]= 0.9
        #     if is_fvp == True:
        #         attack_params["BaseRec"] = 15
        #         attack_params["ConfRec"] = 5
        #     deduce_sp_params={}
        #     deduce_sp_params["delta"]=0.95
        #     deduce_sp_params["p_q_threshold"]=0.95
        #     deduce_sp_params["layer_match"] = 5
        #     attack_params["deduce_sp_params"]=deduce_sp_params
        #     acc,ari = run_single_attack(leakage_params,dataset_params,attack_params)
        #     print("Jigsaw+, observe_days:",observe_days," Acc:",acc," Ari:",ari)
        #     Acc.append(acc)
        # with open("./results/no_update/Jigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_observed_query_number_per_timeslot_"+str(observed_query_number_per_timeslot)+".pkl","wb") as f:
        #     pickle.dump(Acc,f)
##################Sap+#######################
        # Acc = []
        # for _ in tqdm.tqdm(range(test_times)):
        #     attack_params = {}
        #     attack_params["attack_name"]="SAP+"
        #     attack_params["need_deduce_sp"]=True
        #     attack_params["need_cooccurrence"]=False
        #     attack_params["is_truth"] = False
        #     attack_params["alpha"]= 0.5
           
        #     deduce_sp_params={}
        #     deduce_sp_params["delta"]=0.95
        #     deduce_sp_params["p_q_threshold"]=0.95
        #     deduce_sp_params["layer_match"] = 5
        #     attack_params["deduce_sp_params"]=deduce_sp_params
        #     acc,ari = run_single_attack(leakage_params,dataset_params,attack_params)
        #     print("SAP+, observe_days:",observe_days," Acc:",acc," Ari:",ari)
        #     Acc.append(acc)
        # with open("./results/no_update/SAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_observed_query_number_per_timeslot_"+str(observed_query_number_per_timeslot)+".pkl","wb") as f:
        #     pickle.dump(Acc,f)
#################################FMA############
        # Acc = []
        # for _ in tqdm.tqdm(range(test_times)):
        #     attack_params = {}
        #     attack_params["attack_name"]="FMA"
        #     attack_params["delta"] = 0.95
        #     attack_params["need_deduce_sp"]=False
        #     attack_params["need_cooccurrence"]=False
        #     attack_params["is_truth"] = False
        #     deduce_sp_params={}
        #     deduce_sp_params["delta"]=0.95
        #     attack_params["deduce_sp_params"]=deduce_sp_params
        #     acc,ari = run_single_attack(leakage_params,dataset_params,attack_params)
        #     print("FMA, observe_days:",observe_days," Acc:",acc," Ari:",ari)
        #     Acc.append(acc)
        # with open("./results/no_update/FMA_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_observed_query_number_per_timeslot_"+str(observed_query_number_per_timeslot)+".pkl","wb") as f:
        #     pickle.dump(Acc,f)
########################SP&Jigsaw+############
        Acc = []
        for _ in tqdm.tqdm(range(test_times)):
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
            acc,ari = run_single_attack(leakage_params,dataset_params,attack_params)
            print("SP&Jigsaw+, observe_days:",observe_days," Acc:",acc," Ari:",ari)
            Acc.append(acc)
        with open("./results/no_update/SPJigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_observed_query_number_per_timeslot_"+str(observed_query_number_per_timeslot)+".pkl","wb") as f:
            pickle.dump(Acc,f)
###################SP&SAP+######################
        Acc = []
        for _ in tqdm.tqdm(range(test_times)):
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
            acc,ari = run_single_attack(leakage_params,dataset_params,attack_params)
            print("SPSAP+, observe_days:",observe_days," Acc:",acc," Ari:",ari)
            Acc.append(acc)
        with open("./results/no_update/SPSAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_observed_query_number_per_timeslot_"+str(observed_query_number_per_timeslot)+".pkl","wb") as f:
            pickle.dump(Acc,f)

if __name__ == '__main__':
#     args = sys.argv[1:]
#     params = {}
#     for arg in args:
#         key, value = arg.split('=')
#         params[key] = value
#     # number = params["number"]
#     is_fvp = params["fvp"]
#     dataset = params["dataset"]
  
#     # test_query_number([int(number)],bool(is_fvp),dataset,5)
#     test_query_number([10000,5000,2500,1000],bool(is_fvp),dataset,5)

# test_query_number([5000,2500,1000],True,"enron",5)
    # for time in range(5):
    for i in [[1000],[2500],[5000],[10000]]:
        test_query_number(i,False,"enron",5)
        test_query_number(i,False,"lucene",5)
        test_query_number(i,True,"lucene",5)
        test_query_number(i,True,"enron",5)
