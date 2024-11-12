from collections import Counter
import numpy as np
import time
from tqdm import tqdm
class FMA:
    def __init__(self, query_doc_multi_window_list, F_sim, observed_query_number_per_timeslot, observed_timeslot_number_per_cycle, delta = 0.6, is_fvp = False):
        self.F = F_sim
        self.delta = delta
        self.MM = {}
        self.tdid_2_kwsid = {} 
        self.is_fvp = is_fvp
        self.query_sequence = []
        self.response_sequence = []
        self.groups_list = []
        self.q_remained = []
        self.query_doc_multi_window_list = query_doc_multi_window_list
        self.observed_query_number_per_timeslot = observed_query_number_per_timeslot
        self.observed_timeslot_number_per_cycle = observed_timeslot_number_per_cycle

    def attack(self):
        
        begin_time = time.time()
        self.candidate_gen()
        print("step 1:", time.time() - begin_time)
        
        for i in range(len(self.groups_list)):
            for group in self.groups_list[i]:
                for query_id in group:
                    if len(self.MM[query_id]) == 1:
                        self.tdid_2_kwsid[query_id] = self.MM[query_id][0]
                    else:
                        self.q_remained.append(query_id)

        q_recovered = set()
        for q in self.q_remained:
            if q in q_recovered:
                continue
            q_recover_together = [q] 
            q_to_kwid_tmp = set(self.MM[q])
            for q_prime in self.q_remained:
                if q_prime in q_recovered:
                    continue
               
                if self.qeq(self.response_sequence[q], self.response_sequence[q_prime]) == 1:
                    q_to_kwid_tmp = q_to_kwid_tmp & set(self.MM[q_prime])
                    q_recover_together.append(q_prime)
            if len(q_to_kwid_tmp) == 1:
                for query_id in q_recover_together:
                    self.tdid_2_kwsid[query_id] = list(q_to_kwid_tmp)[0]
                    q_recovered.add(query_id)

        return self.tdid_2_kwsid

    def candidate_gen(self):
        query_sequence_timeslot_list = []
        num_query_total = 0
        for i_multi_window in range(len(self.query_doc_multi_window_list)):
            for query_doc_dic_single_timeslot in self.query_doc_multi_window_list[i_multi_window]:
                query_sequence_timeslot = []
                response_sequence_timeslot = []
                query_number = len(query_doc_dic_single_timeslot)

                query_sequence_timeslot.extend([num_query_total + j for j in range(query_number)])
                response_sequence_timeslot.extend(query_doc_dic_single_timeslot)

                self.response_sequence.extend(response_sequence_timeslot)
                query_sequence_timeslot_list.append(query_sequence_timeslot)

                num_query_total += self.observed_query_number_per_timeslot 
                
        for query_sequence_timeslot in query_sequence_timeslot_list:
            groups = self.deduce_sp(query_sequence_timeslot, self.response_sequence)
            self.groups_list.append(groups)

        for i in range(len(self.groups_list)):
            groups = self.groups_list[i]
            p = np.array(self.F[:,i])
            num = sum(len(group) for group in groups)
            for group in groups:
                tmp = np.abs(len(group) - p * num)
                min_value = np.min(tmp)
                index = np.where(tmp == min_value)[0]
                for query_id in group:
                    self.MM[query_id] = index

        
    def deduce_sp(self, query_sequence, response_sequence):
        groups = []

        temp_group = [query_sequence[0]]
        groups.append(temp_group)

        query_id_2_group_id = {}
        query_id_2_group_id[query_sequence[0]] = 0

        for k in tqdm(range(1, len(query_sequence))):
            q = query_sequence[k]
            cand = []
            for i_query_id in range(k):
                ap1, ap2 = response_sequence[query_sequence[i_query_id]], response_sequence[q]
                if self.is_fvp:
                    counter1, counter2 = Counter(ap1), Counter(ap2)
                    inter = counter1 & counter2
                    union = counter1 | counter2
                    if sum(union.values()) == 0:
                        rsp = 1
                    else:
                        rsp = sum(inter.values()) / sum(union.values())
                else:
                    counter1, counter2 = set(ap1), set(ap2)
                    inter = counter1 & counter2
                    union = counter1 | counter2
                    if len(union)==0 :
                        rsp = 1
                    else:
                        rsp = len(inter) / len(union)
                if rsp >= self.delta:
                    cand.append((rsp, query_id_2_group_id[query_sequence[i_query_id]]))
            if not cand:
                temp_group = [q]
                query_id_2_group_id[q] = len(groups)
                groups.append(temp_group)
            else:
                cand.sort(reverse=True, key= lambda x: x[0])
                query_id_2_group_id[q] = cand[0][1]
                groups[cand[0][1]].append(q)
        return groups
    
    def qeq(self, q, q_prime):
        if self.is_fvp:
            counter1, counter2 = Counter(q), Counter(q_prime)
            inter = counter1 & counter2
            union = counter1 | counter2
            if sum(union.values()) == 0:
                rsp = 1
            else:
                rsp = sum(inter.values()) / sum(union.values())
        else:
            counter1, counter2 = set(q), set(q_prime)
            inter = counter1 & counter2
            union = counter1 | counter2
            if len(union)==0 :
                rsp = 1
            else:
                rsp = len(inter) / len(union)

        if rsp >= self.delta:
            return 1
        else:
            return 0