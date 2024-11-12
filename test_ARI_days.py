from run_deduce_sp import run_peecaboo_attack
import pickle
from tqdm import tqdm
observed_query_numbers = [10000]
timeslot_numbers = [1,2,4,8]
datasets = ['lucene']
test_times = 10
is_fvp = True 
for dataset in datasets:
    if dataset == "enron":
        storage_time_limit = 365
    else:
        storage_time_limit = 365 * 3
    ARI = []
    RI = []
    for timeslot_number in timeslot_numbers:
        Ari = []
        Ri = []
        for observed_query_number in observed_query_numbers:
            unobserved_timeslot_number = 20 - timeslot_number
            ari = []
            ri = []
            for test_time in tqdm(range(test_times)):
                leakage_params={}
                leakage_params["cycles_number"] = 5
                leakage_params["observed_timeslot_number_per_cycle"]=[timeslot_number]*leakage_params["cycles_number"]
                leakage_params["unobserved_timeslot_num_per_cycle"]=[unobserved_timeslot_number]*leakage_params["cycles_number"]
                leakage_params["observed_query_number_per_timeslot"]=observed_query_number
                leakage_params["is_fvp"]=is_fvp

                dataset_params = {}
                dataset_params["dataset_name"]=dataset
                dataset_params["deleted_email_percent"]=0.1
                dataset_params["storage_time_limit"]=storage_time_limit
                dataset_params["kws_universe_size"]=500
                dataset_params["kws_extraction"]="sorted"

                attack_params = {}
                deduce_sp_params={}
                deduce_sp_params["delta"]=0.95
                deduce_sp_params["p_q_threshold"]=0.95
                deduce_sp_params["layer_match"] = 5
                attack_params["deduce_sp_params"]=deduce_sp_params
                ari_temp,ri_temp = run_peecaboo_attack(leakage_params,dataset_params,attack_params)
                print(dataset," timeslot_number:",timeslot_number," observed_query_number:",observed_query_number," ARI:",ari_temp," RI:",ri_temp)
                ari.append(ari_temp)
                ri.append(ri_temp)
            Ari.append(ari)
            Ri.append(ri)
        ARI.append(Ari)
        RI.append(Ri)
    with open("./results/peekaboo_ari_test_1_"+dataset+"_isfvp_"+str(int(is_fvp))+".pkl", "wb") as f:
        pickle.dump((ARI,RI),f)


