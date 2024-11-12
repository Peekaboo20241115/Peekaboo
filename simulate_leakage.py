from datetime import datetime, timedelta
import pickle
import numpy as np
import time
from countermeasures import padding, obfuscate

def simulate_leakage(observed_query_number_per_timeslot = 300, observed_timeslot_number_per_cycle = [7,7,7,7], unobserved_timeslot_num_per_cycle = [7,7,7,7], \
                cycles_number = 4, deleted_email_percent = 0.1, storage_time_limit = 90, kws_universe_size = 1000, \
                dataset_path = './datasets/newlucene.pkl', is_fvp = False, \
                begin_date = "1 Jun 2000 00:00:00 +0000", tau_client = 0, tau_attacker = 0, kws_extraction = 'sorted', countermeasure_info = {'name':None}, debug_info = False):
    begin_running_time = time.time()
    if debug_info==True:
        print("simulate leakage begins")
        
    # For enron, add_file_begin_time is "1 Feb 2000 00:00:00 +0000", for lucene, it is "1 Feb 2002 00:00:00 +0000".
    add_file_begin_time = datetime.strptime(begin_date, "%d %b %Y %H:%M:%S %z")
    add_file_begin_time = add_file_begin_time.replace(tzinfo=None)
    
    # Read dataset
    with open(dataset_path,"rb") as f:
        dataset = pickle.load(f)

    # _ is a list of keywords, total_doc is a list of files, each file is a list of number representing the index of keywords
    # aux is dict which contains key "all_documents_size"(the list of the size of all documents), 
    # "timestamps"(the added timestamp of files), "stems_to_words"(unused) and "trends"(the query frequency of keywords).
    if dataset_path == './datasets/lucene.pkl' or dataset_path == './datasets/enron.pkl':

        keywords_list = dataset["keywords_list"]
        trend_matrix = dataset["keywords_trend"]
        kws_count = dataset["keywords_count"]
        total_doc = dataset["total_doc"]
        doc_size = dataset["doc_size"]
        doc_timestamp = dataset["doc_timestamp"]
    else:
        assert 0
    # countermeasure parameter
    is_obfuscated = False
    is_obfuscated_attacker = False

    if countermeasure_info['name'] == 'fvp_padding':
        if 'padding_k' not in countermeasure_info:
            padding_k = 500
        else:
            padding_k = countermeasure_info['padding_k']
        doc_size = padding(doc_size, padding_k)
    elif countermeasure_info['name'] == 'obfuscate':
        is_obfuscated = True
        p = countermeasure_info['p']
        q = countermeasure_info['q']
        is_obfuscated_attacker = countermeasure_info['is_obfuscated_attacker']
    

    # normalize the trend
    for i_col in range(trend_matrix.shape[1]):
        if sum(trend_matrix[:, i_col]) == 0:
            trend_matrix[:, i_col] = 1 / len(keywords_list)
        else:
            trend_matrix[:, i_col] = trend_matrix[:, i_col] / sum(trend_matrix[:, i_col])
    
    # get keyword universe
    if kws_extraction == "sorted":
        kws_count.sort(reverse=True,key = lambda x: x[1]) 
        kws_universe = kws_count[:kws_universe_size]
        kws_universe = [tmp[0] for tmp in kws_universe]
        kws_universe_dic = {tmp:i for i, tmp in enumerate(kws_universe)} 
    else:
        assert 0

    #client's dataset
    # client's doc 
    client_doc_unobfuscated = []
    # client's doc in the server 
    client_doc_server = [] 
    # client_doc[0] is a binary vector, indicates whether the keyword in kws_universe is in the doc
    # client_doc[1] is the leakage of retrieving this doc, i.e. its id or size
    # client_doc[3] is the time when this doc is added
    client_file_num = 0
    #attacker's dataset
    attacker_doc = [] # similar to client_doc, while attacker_doc[1] is the id
    attacker_doc_obfuscated = []
    attacker_doc_num = 0
    
    #before the attack, the client's dataset contains "storage_time_limit" days files.
    #get the doc added in a single day, and split them for the client and the attacker
    add_file_end_time = add_file_begin_time + timedelta(days=1)
    end_index, day_doc = read_file_day(total_doc, doc_timestamp, doc_size, add_file_begin_time, add_file_end_time, 0, kws_universe_dic, kws_universe_size)
    client_file_num, attacker_doc_num = file_allocation(client_doc_unobfuscated, client_doc_server, attacker_doc, attacker_doc_obfuscated, client_file_num, attacker_doc_num, day_doc, deleted_email_percent, is_fvp)
    for day in range(1, storage_time_limit):
        add_file_end_time = add_file_end_time + timedelta(days=1)
        end_index, day_doc = read_file_day(total_doc, doc_timestamp, doc_size, add_file_begin_time, add_file_end_time, end_index, kws_universe_dic, kws_universe_size)
        client_file_num, attacker_doc_num = file_allocation(client_doc_unobfuscated, client_doc_server, attacker_doc, attacker_doc_obfuscated, client_file_num, attacker_doc_num, day_doc, deleted_email_percent, is_fvp)

    # simulate the updates and leakages in each cycle
    query_id_multi_window_list = [] # plaintext queries
    query_doc_multi_window_list=[] # leakages of queries
    # the attacker updates its own dataset and record the leakage in each time slot
    V_sim_multi_window = [] #volume information of the attacker
    C_sim_multi_window = [] #co-occurrence information of the attacker in different window
    Doc_number_sim_per_window = []

    time_offset_client = tau_client

    
    for i in range(cycles_number):
        # obfuscate
        if is_obfuscated:
            # client tries to obfuscate
            pre_obfuscate = [doc[0] for doc in client_doc_unobfuscated]
            post_obfuscate = obfuscate(pre_obfuscate, p, q)
            client_doc_server = [[post_obfuscate[i_doc], client_doc_unobfuscated[i_doc][1], client_doc_unobfuscated[i_doc][2]] for i_doc in range(len(client_doc_unobfuscated))]

        else:
            client_doc_server = client_doc_unobfuscated    
        if is_obfuscated_attacker:
            pre_obfuscate = [doc[0] for doc in attacker_doc]
            post_obfuscate = obfuscate(pre_obfuscate, p, q)
            attacker_doc_obfuscated = [[post_obfuscate[i_doc], attacker_doc[i_doc][1], attacker_doc[i_doc][2]] for i_doc in range(len(attacker_doc))]
        else:
            attacker_doc_obfuscated = attacker_doc
                
        # client
        query_id_single_window_list = [] # plaintext queries
        query_doc_single_window_list=[] # leakages of queries
        V_sim_single_window = [] # the volume information of attacker's dataset
        C_sim_single_window = []
        ID_sim_single_window = []
        attacker_doc_set = set()
        # begin simulate a window
        for _ in range(observed_timeslot_number_per_cycle[i]):
    
            time_offset_client = time_offset_client + 1
            f = []
            for kw in kws_universe:
                f.append(trend_matrix[kw, time_offset_client])
            #f = np.array([1]*len(f))
            f = f / sum(f)

            #simulate queries
            query_client = np.random.choice(kws_universe, observed_query_number_per_timeslot, replace = True, p = f)

            time_offset_client += unobserved_timeslot_num_per_cycle[i]

            #calculate the leakage of queries
            doc_query_ID = np.array([doc[0] for doc in client_doc_server]).T
            doc_leakage = np.array([doc[1] for doc in client_doc_server])
            query_id_single_timeslot = [] #queries list in a single time slot
            query_doc_single_timeslot = [] #doc leakage list in a single timeslot
            for j_query in range(observed_query_number_per_timeslot):
                leakage_temp = doc_query_ID[query_client[j_query]] * doc_leakage
                query_doc_single_timeslot.append(leakage_temp[leakage_temp!=0].tolist())
                query_id_single_timeslot.append(query_client[j_query])

            query_doc_single_window_list.append(query_doc_single_timeslot)
            query_id_single_window_list.append(query_id_single_timeslot)

            # at the sametime, get the attacker's doc number
            Doc_number_sim_per_window.append(len(attacker_doc_obfuscated))

            #the volume information of attacker's dataset
            doc_keyword_ID = np.array([doc[0] for doc in attacker_doc_obfuscated]).T
            doc_id = np.array([doc[1] for doc in attacker_doc_obfuscated])
            keyword_volume_single_timeslot = []
            for keyword in kws_universe:
                doc_id_temp = doc_keyword_ID[keyword] * doc_id
                keyword_volume_single_timeslot.append(len(set(doc_id_temp[doc_id_temp!=0].tolist())))
            V_sim_single_window.append(keyword_volume_single_timeslot)
            #the ID matrix to calculate the co-occurrence of attacker's dataset
            doc_keyword_ID = doc_keyword_ID.T
            doc_id_set = set(doc_id.tolist())
            new_doc_id_set = doc_id_set - attacker_doc_set
            for doc in attacker_doc_obfuscated:
                if doc[1] in new_doc_id_set:
                    ID_sim_single_window.append(doc[0])
            attacker_doc_set.update(new_doc_id_set)
            
            # simulate updates in a day
            add_file_end_time = add_file_end_time + timedelta(days=1)
            end_index, day_doc = read_file_day(total_doc, doc_timestamp,doc_size, add_file_begin_time, add_file_end_time, end_index, kws_universe_dic, kws_universe_size)
            client_file_num, attacker_doc_num = file_allocation(client_doc_unobfuscated, client_doc_server, attacker_doc, attacker_doc_obfuscated, client_file_num, attacker_doc_num, day_doc, deleted_email_percent, is_fvp)

            # simulate deleting the outdated files in the client's and the attacker's dataset
            file_deleted_time = add_file_end_time + timedelta(days=-storage_time_limit)
            client_doc_unobfuscated = [doc for doc in client_doc_unobfuscated if doc[2] >= file_deleted_time]
            client_doc_server = [doc for doc in client_doc_server if doc[2] >= file_deleted_time]
            attacker_doc = [doc for doc in attacker_doc if doc[2] >= file_deleted_time]
            attacker_doc_obfuscated = [doc for doc in attacker_doc_obfuscated if doc[2] >= file_deleted_time]
        
           

        #simulate updates during unobserved period
        for _ in range(unobserved_timeslot_num_per_cycle[i]):

            add_file_end_time = add_file_end_time + timedelta(days=1)
            end_index, day_doc = read_file_day(total_doc, doc_timestamp, doc_size, add_file_begin_time, add_file_end_time, end_index, kws_universe_dic, kws_universe_size)
            client_file_num, attacker_doc_num = file_allocation(client_doc_unobfuscated, client_doc_server, attacker_doc, attacker_doc_obfuscated, client_file_num, attacker_doc_num, day_doc, deleted_email_percent, is_fvp)
            
            file_deleted_time = add_file_end_time + timedelta(days=-storage_time_limit)
            client_doc_unobfuscated = [doc for doc in client_doc_unobfuscated if doc[2] >= file_deleted_time]
            attacker_doc = [doc for doc in attacker_doc if doc[2] >= file_deleted_time]
            attacker_doc_obfuscated = [doc for doc in attacker_doc_obfuscated if doc[2] >= file_deleted_time]
        
        query_id_multi_window_list.append(query_id_single_window_list)
        query_doc_multi_window_list.append(query_doc_single_window_list)
        V_sim_multi_window.append(V_sim_single_window)
        ID_sim_single_window = np.array(ID_sim_single_window)
        C_sim_single_window = np.matmul(ID_sim_single_window.T, ID_sim_single_window)
        C_sim_single_window = C_sim_single_window / len(attacker_doc_set)
        C_sim_multi_window.append(C_sim_single_window)

    # the attacker's frequency information
    F_sim_multi_window = []
    time_offset_attacker = tau_attacker
    for i in range(cycles_number):
        F_sim_single_window = []
        for j in range(observed_timeslot_number_per_cycle[i]):
            f_sim = [trend_matrix[kw, time_offset_attacker] for kw in kws_universe]
            f_sim = f_sim / sum(f_sim)
            F_sim_single_window.append(f_sim)
            time_offset_attacker += 1
        F_sim_multi_window.append(F_sim_single_window)
        time_offset_attacker += unobserved_timeslot_num_per_cycle[i]

    if debug_info==True:
        print("simulate leakage finished, runtime:", time.time() - begin_running_time, "seconds")
    return query_id_multi_window_list, query_doc_multi_window_list, F_sim_multi_window, V_sim_multi_window, C_sim_multi_window, Doc_number_sim_per_window



def read_file_day(total_doc, doc_timestamp, doc_size, add_file_begin_time, add_file_end_time, begin_index, kws_universe_dic, kws_universe_size):
    # read files with timestamp between "add_file_begin_time" and "add_file_end_time form dataset"
    day_doc = []
    for j_timestamp in range(begin_index, len(doc_timestamp)):
        timestamp = doc_timestamp[j_timestamp]
        dt = datetime.fromtimestamp(timestamp)
        if add_file_begin_time <= dt < add_file_end_time: #
            doc = [0] * kws_universe_size
            for kw in total_doc[j_timestamp]:
                if kw in kws_universe_dic:
                    doc[kws_universe_dic[kw]] = 1
            day_doc.append([doc, doc_size[j_timestamp], dt])
        elif dt >= add_file_end_time:
            end_index = j_timestamp
            break
    return end_index, day_doc

def file_allocation(client_doc_unobfuscated, client_doc_server, attacker_doc, attacker_doc_obfuscated, total_file_number, total_file_number_similar, needed_doc, deleted_email_percent, is_fvp,attacker_del=False):
    #split the files, half of the files is client's file and another half is the attacker's file
    #the client will randomly remove some file from its whole dataset (remove number = deleted_email_percent * added files number)
    num_client = len(needed_doc) // 2
    num_attacker = len(needed_doc) - num_client

    np.random.shuffle(needed_doc)

    for i in range(num_client):
        client_doc_unobfuscated.append([needed_doc[i][0], needed_doc[i][1] if is_fvp else total_file_number, needed_doc[i][2]])# file size or file id
        client_doc_server.append([needed_doc[i][0], needed_doc[i][1] if is_fvp else total_file_number, needed_doc[i][2]])
        total_file_number += 1

    for i in range(num_client, len(needed_doc)):
        #attacker_doc.append([needed_doc[i][0], needed_doc[i][1] if is_fvp else total_file_number_similar, needed_doc[i][2]])
        attacker_doc.append([needed_doc[i][0], total_file_number_similar, needed_doc[i][2]])
        attacker_doc_obfuscated.append([needed_doc[i][0], total_file_number_similar, needed_doc[i][2]])
        total_file_number_similar += 1

    if round(deleted_email_percent * num_client) != 0:
        indices = list(range(len(client_doc_unobfuscated)))
        np.random.shuffle(indices)
        indices = indices[: -round(deleted_email_percent * num_client)]

        client_doc_unobfuscated = [client_doc_unobfuscated[i] for i in indices]
        client_doc_server = [client_doc_server[i] for i in indices]

    if attacker_del == True:
        if round(deleted_email_percent * num_attacker) != 0:
            indices = list(range(len(client_doc_unobfuscated)))
            np.random.shuffle(indices)
            indices = indices[: -round(deleted_email_percent * num_client)]

            attacker_doc = [attacker_doc[i] for i in indices]
            attacker_doc_obfuscated = [attacker_doc_obfuscated[i] for i in indices]

    return total_file_number, total_file_number_similar

if __name__ == '__main__':
    simulate_leakage(debug_info=True)