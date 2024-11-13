from run_single_attack import run_single_attack
import os
import pickle
import tqdm
import random
import numpy as np
import sys
def test_random_day(unobserved_timeslot_num_per_cycle_total,lower_bound, upper_bound, Cycles_number, is_fvp, dataset, test_times):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/random_day"):
        os.mkdir("./results/random_day")
    save_dir = "./results/random_day"

    for cycles_number in Cycles_number:
        leakage_params={}
        leakage_params["cycles_number"] = cycles_number
        leakage_params["observed_timeslot_number_per_cycle"]=[1 for _ in range(leakage_params["cycles_number"])]
        
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

        print(f"dataset:{dataset}-is_fvp:{is_fvp}-upper-bound:{upper_bound}")

        Acc_Jigsaw_plus = []
        Acc_SAP_plus = []
        Acc_FMA = []
        Acc_SP_Jigsaw_plus = []
        Acc_SP_SAP_plus = []
        for i in tqdm.tqdm(range(test_times)):
           
            leakage_params["unobserved_timeslot_num_per_cycle"]=[random.randint(lower_bound,upper_bound) for _ in range(leakage_params["cycles_number"])]
##################Jigsaw+################  
            attack_params = {}
            attack_params["attack_name"]="Jigsaw+"
            attack_params["need_deduce_sp"]=True
            attack_params["need_cooccurrence"]=True
            attack_params["is_truth"] = False
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
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
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
            deduce_sp_params["p_q_threshold"]=0.95
            deduce_sp_params["layer_match"] = 5
            attack_params["deduce_sp_params"]=deduce_sp_params
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
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
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
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
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
            print("SP&Jigsaw+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc_SP_Jigsaw_plus.append(acc)
        
###################SP&SAP+######################
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


        with open("./results/random_day/Jigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_upperbound_"+str(upper_bound)+".pkl","wb") as f:
            pickle.dump(Acc_Jigsaw_plus,f)
        with open("./results/random_day/SAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_upperbound_"+str(upper_bound)+".pkl","wb") as f:
            pickle.dump(Acc_SAP_plus,f)
        with open("./results/random_day/FMA_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_upperbound_"+str(upper_bound)+".pkl","wb") as f:
            pickle.dump(Acc_FMA,f)
        with open("./results/random_day/SPJigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_upperbound_"+str(upper_bound)+".pkl","wb") as f:
            pickle.dump(Acc_SP_Jigsaw_plus,f)
        with open("./results/random_day/SPSAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_upperbound_"+str(upper_bound)+".pkl","wb") as f:
            pickle.dump(Acc_SP_SAP_plus,f)

        with open("./results/random_day/upperbound_"+ str(upper_bound)+".pkl","wb") as f:
            pickle.dump(unobserved_timeslot_num_per_cycle_total,f)


if __name__ == "__main__":

    cycles_number = 5
    # # unobserved_timeslot_num_per_cycle_total = [[random.randint(lower_bound,upper_bound) for _ in range(cycles_number)] for _ in range(5)]
    # with open("./results/random_day/upperbound_"+ str(upper_bound)+".pkl","rb") as f:
    #     unobserved_timeslot_num_per_cycle_total = pickle.load(f)
    # unobserved_timeslot_num_per_cycle_total = np.array(unobserved_timeslot_num_per_cycle_total)
    # print(unobserved_timeslot_num_per_cycle_total)
    # for i in range(4):
    #     test_random_day(unobserved_timeslot_num_per_cycle_total[i,:], lower_bound, upper_bound, [5], False, 'enron', 5)
    #     test_random_day(unobserved_timeslot_num_per_cycle_total[i,:], lower_bound, upper_bound, [5], False, 'lucene', 5)
    #     test_random_day(unobserved_timeslot_num_per_cycle_total[i,:], lower_bound, upper_bound, [5], True, 'enron', 5)
    #     test_random_day(unobserved_timeslot_num_per_cycle_total[i,:], lower_bound, upper_bound, [5], True, 'lucene', 5)

    args = sys.argv[1:]
    params = {}
    for arg in args:
        key, value = arg.split('=')
        params[key] = value
    upper_bound = params["upper_bound"]
    lower_bound = params["lower_bound"]
    dataset = params["dataset"]

    unobserved_timeslot_num_per_cycle_total = [[random.randint(lower_bound,upper_bound) for _ in range(cycles_number)] for _ in range(5)]
    test_random_day(unobserved_timeslot_num_per_cycle_total, int(lower_bound), int(upper_bound), [5], False, dataset, 5)
    test_random_day(unobserved_timeslot_num_per_cycle_total, int(lower_bound), int(upper_bound), [5], True, dataset, 5)

