from run_deduce_sp import run_peecaboo_attack
import pickle
from tqdm import tqdm
if __name__ == '__main__':
    observed_query_numbers = [10000]
    Cycle_number = [1,2,4,8,16]
    datasets = ['enron','lucene']
    test_times = 10
    is_fvp = False 
    for dataset in datasets:
        ARI = []
        RI = []
        for cycle_number in Cycle_number:
            Ari = []
            Ri = []
            for observed_query_number in observed_query_numbers:
                timeslot_number = 2
                unobserved_timeslot_number = 18
                ari = []
                ri = []
                for test_time in tqdm(range(test_times)):
                    leakage_params={}
                    leakage_params["cycles_number"] = cycle_number
                    leakage_params["observed_timeslot_number_per_cycle"]=[timeslot_number]*cycle_number
                    leakage_params["unobserved_timeslot_num_per_cycle"]=[unobserved_timeslot_number]*cycle_number
                    leakage_params["observed_query_number_per_timeslot"]=observed_query_number
                    leakage_params["is_fvp"]=is_fvp

                    dataset_params = {}
                    dataset_params["dataset_name"]=dataset
                    dataset_params["deleted_email_percent"]=0.1
                    dataset_params["storage_time_limit"]=365
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
        with open("./results/peekaboo_ari_test_2_"+dataset+"_isfvp_"+str(int(is_fvp))+".pkl", "wb") as f:
            pickle.dump((ARI,RI),f)


