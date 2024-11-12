from simulate_leakage import simulate_leakage

from attacks.infer_sp import Deduce_sp

from utils import cal_ARI_multi


def run_peecaboo_attack(
        leakage_params,
        dataset_params,
        attack_params
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
        begin_data = "1 Feb 2000 00:00:00 +0000"
    elif dataset_name == "lucene":
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
                    begin_date = begin_data, kws_extraction = kws_extraction,  debug_info = False)
    
    deduce_sp_params = attack_params["deduce_sp_params"]
    deduce_sp_attacker = Deduce_sp(query_doc_multi_window_list, deduce_sp_params["delta"], deduce_sp_params["p_q_threshold"],deduce_sp_params["layer_match"], is_fvp)
    deduce_sp_attacker.deduce_sp()
    M = deduce_sp_attacker.M 
    ari_multi_window, ri_multi_window = cal_ARI_multi(M, query_id_multi_window_list)
    return ari_multi_window, ri_multi_window
   