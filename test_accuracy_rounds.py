from run_single_attack import run_single_attack
import os
import pickle
import tqdm
import sys
def test_comparison(Cycles_number,is_fvp,dataset,test_times):
    print(Cycles_number,is_fvp,dataset,test_times) 
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/comparison"):
        os.mkdir("./results/comparison")
    save_dir = "./results/comparison"

    for cycles_number in Cycles_number:
        leakage_params={}
        leakage_params["cycles_number"] = cycles_number
        leakage_params["observed_timeslot_number_per_cycle"]=[1]*leakage_params["cycles_number"]
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

###################Jigsaw+################
        Acc = []
        for _ in tqdm.tqdm(range(test_times)):
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
            Acc.append(acc)
        with open("./results/comparison/Jigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","wb") as f:
            pickle.dump(Acc,f)
##################Sap+#######################
        Acc = []
        for _ in tqdm.tqdm(range(test_times)):
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
            Acc.append(acc)
        with open("./results/comparison/SAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","wb") as f:
            pickle.dump(Acc,f)
#################################FMA############
        Acc = []
        for _ in tqdm.tqdm(range(test_times)):
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
            Acc.append(acc)
        with open("./results/comparison/FMA_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","wb") as f:
            pickle.dump(Acc,f)
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
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
            print("SP&Jigsaw+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc.append(acc)
        with open("./results/comparison/SPJigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","wb") as f:
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
            acc,ari,_ = run_single_attack(leakage_params,dataset_params,attack_params)
            print("SPSAP+, cycle_number:",cycles_number," Acc:",acc," Ari:",ari)
            Acc.append(acc)
        with open("./results/comparison/SPSAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","wb") as f:
            pickle.dump(Acc,f)

if __name__ == '__main__':
    args = sys.argv[1:]
    params = {}
    for arg in args:
        key, value = arg.split('=')
        params[key] = value
    rounds = params["rounds"]
    dataset = params["dataset"]
    test_comparison([int(rounds)],False,dataset,5)
    test_comparison([int(rounds)],True,dataset,5)
#test_comparison([1,2,4,8],False,"enron",5)
#test_comparison([2],False,"enron",5)
#test_comparison([1],False,"enron",5)
#test_comparison([4],False,"enron",5)
#test_comparison([8],False,"enron",5)



#test_comparison([1,2,4,8],False,"lucene",5)
#test_comparison([1],False,"lucene",5)
#test_comparison([2],False,"lucene",5)
#test_comparison([4],False,"lucene",5)
#test_comparison([8],False,"lucene",5)

#test_comparison([1],True,"enron",5)
#test_comparison([2],True,"enron",5)
#test_comparison([4],True,"enron",5)
#test_comparison([8],True,"enron",5)
#test_comparison([1,2,4,8],True,"enron",5)


#test_comparison([1],True,"lucene",5)
#test_comparison([2],True,"lucene",5)
#test_comparison([4],True,"lucene",5)
#test_comparison([8],True,"lucene",5)
#test_comparison([1,2,4,8],True,"lucene",5)
