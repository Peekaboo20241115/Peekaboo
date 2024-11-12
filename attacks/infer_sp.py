from collections import Counter
from tqdm import tqdm
import concurrent.futures
from functools import partial
import time
import numpy as np
from .ihop import QAP
import math

class Deduce_sp:
    def __init__(self, query_doc_multi_window_list, delta = 0.9,p_q_threshold = 0.9,layer_match=2, is_fvp = False) -> None:
        self.query_doc_multi_window_list = query_doc_multi_window_list
        self.delta = delta
        self.p_q_threshold = p_q_threshold
        self.layer_match = layer_match
        self.is_fvp = is_fvp


        self.groups_multi_window = [None] * len(self.query_doc_multi_window_list) #results of deduce_sp_single
        self.id_matrix_list = []
        self.M = {}
        self.ID_M = []
    def deduce_sp(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
            future_to_task = {executor.submit(self.deduce_sp_single, self.query_doc_multi_window_list[i]): i for i in range(len(self.query_doc_multi_window_list))}
            for future in concurrent.futures.as_completed(future_to_task):
                task_num = future_to_task[future]
                try:
                    result = future.result()
                    self.groups_multi_window[task_num] = result
                except Exception as exc:
                    print(f"Task {task_num} generated an exception: {exc}")
        self.M = self.deduce_sp_multi()
        
    def match_last(self, leakage1, leakage2, index):
       
        if not self.is_fvp:
            ap1, ap2 = leakage1, leakage2
            counter1, counter2 = set(ap1), set(ap2)
            inter = counter1 & counter2
            union = counter1 | counter2
            if len(union)==0 :
                rsp = 1
            else:
                rsp = len(inter) / len(union)
            
        else:
            fvp1, fvp2 = leakage1, leakage2
            counter1, counter2 = Counter(fvp1), Counter(fvp2)
            inter = counter1 & counter2
            union = counter1 | counter2
            if sum(union.values()) == 0:
                rsp = 1
            else:
                rsp = sum(inter.values()) / sum(union.values())
        
        if rsp >= self.delta:
            return rsp, index
        else: 
            return None
        
    def deduce_sp_single(self, query_doc_single_window_list):
        query_sequence = []
        for query_doc_dic_single_timeslot_list in query_doc_single_window_list:
            query_number = len(query_doc_dic_single_timeslot_list)
            query_sequence.extend([i for i in range(len(query_sequence), len(query_sequence) + query_number)])

        response_sequence = []
        for query_doc_single_timeslot in query_doc_single_window_list:
            response_sequence.extend(query_doc_single_timeslot)
        

        groups = []
        groups_last = [] # stores the last query of groups and the group id

        temp_group = [query_sequence[0]]
        groups.append(temp_group)
        groups_last.append([query_sequence[0], 0])
        # multi-thread
        for k in tqdm(range(1, len(query_sequence))):
            q = query_sequence[k]
            cand = []

            for group_last in groups_last:
                if not self.is_fvp:
                    ap1, ap2 = response_sequence[group_last[0]], response_sequence[q]
                    counter1, counter2 = set(ap1), set(ap2)
                    inter = counter1 & counter2
                    union = counter1 | counter2
                    if len(union)==0 :
                        rsp = 1
                    else:
                        rsp = len(inter) / len(union)
                    if rsp >= self.delta:
                        cand.append((rsp, group_last[1]))
                else:
                    fvp1, fvp2 = response_sequence[group_last[0]], response_sequence[q]
                    counter1, counter2 = Counter(fvp1), Counter(fvp2)
                    inter = counter1 & counter2
                    union = counter1 | counter2
                    if sum(union.values()) == 0:
                        rsp = 1
                    else:
                        rsp = sum(inter.values()) / sum(union.values())
                    if rsp >= self.delta:
                        cand.append((rsp, group_last[1]))
            if not cand:
                temp_group = [q]
                groups.append(temp_group)
                groups_last.append([q, len(groups_last)])
            else:
                cand.sort(reverse=True, key= lambda x: x[0])
                groups[cand[0][1]].append(q)
                groups_last[cand[0][1]][0] = q
        return groups


    def deduce_sp_multi(self):

        # get IDH, IDT, and ID
        ID_M,IDH_M,IDT_M = [],[],[]
        for i in range(len(self.query_doc_multi_window_list)):
            query_doc_single_window = [query_doc for query_doc_single_timeslot in self.query_doc_multi_window_list[i] for query_doc in query_doc_single_timeslot]

            # get the observed doc list
            doc_list = list(set([doc for query_doc in query_doc_single_window for doc in query_doc]))
            doc_index_dic = dict(zip(doc_list, range(len(doc_list))))

            groups = self.groups_multi_window[i]

            IDH = np.zeros((len(groups),len(doc_list)))
            IDT = np.zeros((len(groups),len(doc_list)))
            ID = np.zeros((len(groups),len(doc_list)))
            for j in range(len(groups)):
                first_query = groups[j][0]
                end_query = groups[j][-1]
                first_query_doc = query_doc_single_window[first_query]
                end_query_doc = query_doc_single_window[end_query]
                for doc in first_query_doc:
                    IDH[j][doc_index_dic[doc]] = 1
                for doc in end_query_doc:
                    IDT[j][doc_index_dic[doc]] = 1
                showed_doc = set()
                for query in groups[j]:
                    showed_doc.update(set(query_doc_single_window[query]))
                for doc in showed_doc:
                    ID[j][doc_index_dic[doc]] = 1
            ID_M.append(ID)
            IDH_M.append(IDH)
            IDT_M.append(IDT)
        self.ID_M = ID_M
        #match between windows
        M = [] #M maps forzen set
        for i in range(len(self.groups_multi_window)):
            for group in self.groups_multi_window[i]:
                M.append([(frozenset(set(group)),i)])
        if self.layer_match > len(self.groups_multi_window):
            layer_match = len(self.groups_multi_window)
        else:
            layer_match = self.layer_match
        
        for i in range(1,layer_match):
            for j in range(len(self.groups_multi_window)-i):
                Matches_new = {}
                #print("Match two between ",len(self.groups_multi_window[j+i])," and ",len(self.groups_multi_window[j])," groups.")
                if len(self.groups_multi_window[j+i])>len(self.groups_multi_window[j]):
                    Matches_new = self.match_two_v2(IDT_M[j],IDH_M[j+i])
                else:
                    Matches_new_inverse = self.match_two_v2(IDH_M[j+i],IDT_M[j])
                    Matches_new = {value:key for key,value in Matches_new_inverse.items()}
                for group1_id in Matches_new.keys():
                    group1 = self.groups_multi_window[j][group1_id]
                    group_list_1,windows1,index1 = self.get_merged_groups(j,group1,M)
                    group2_id = Matches_new[group1_id]
                    group2 = self.groups_multi_window[j+i][group2_id]
                    group_list_2,windows2,index2 = self.get_merged_groups(j+i,group2,M)

                    if len(set(windows1) & set(windows2))==0:
                        new_group_list = []
                        for k in range(len(group_list_1)):
                            new_group_list.append((group_list_1[k],windows1[k]))
                        for k in range(len(group_list_2)):
                            new_group_list.append((group_list_2[k],windows2[k]))
                        M.append(new_group_list)
                        del M[index1]
                        if index1>index2:
                            del M[index2]
                        else:
                            del M[index2-1]

        return M
    
    def get_merged_groups(self,window,group,M):
        for i in range(len(M)):
            group_window_list = M[i]
            if (frozenset(group),window) in group_window_list:
                group_list = []
                window_list = []
                for group_window in group_window_list:
                    group_list.append(group_window[0])
                    window_list.append(group_window[1])
                return group_list, window_list, i
        print("Here at get_merged_groups")
        return [],[],-1
    
    def match_two_v2(self, id_1, id_2):
        co_occurrence_1, co_occurrence_2 = (id_1 @ id_1.T) / id_1.shape[1], (id_2 @ id_2.T) / id_2.shape[1]
        
   
        p_q = QAP(co_occurrence_1, co_occurrence_2, 100, 0.5)
        p_q = self.rm_unmatch(co_occurrence_1, co_occurrence_2, p_q, self.p_q_threshold)
        return p_q

  

    def rm_unmatch(self, matrix_1, matrix_2, p, threshold):
        co_occurrence_1 = matrix_1.astype(float)
        co_occurrence_2 = matrix_2.astype(float)

        values = list(p.values())
        co_occurrence_2_prime = co_occurrence_2[:, values][values, :]
        keys = list(p.keys())
        co_occurrence_1_prime = co_occurrence_1[:, keys][keys, :]

        assert co_occurrence_1_prime.shape[0] == co_occurrence_2_prime.shape[0]
        tmp = []
     
        for i, (k, v) in enumerate(p.items()):
            if sum(co_occurrence_1_prime[i, :]) != 0:
                co_occurrence_1_prime[i, :] = co_occurrence_1_prime[i, :] / sum(co_occurrence_1_prime[i, :])
            else:
                continue
            if sum(co_occurrence_2_prime[i, :]) != 0:
                co_occurrence_2_prime[i, :] = co_occurrence_2_prime[i, :] / sum(co_occurrence_2_prime[i, :])
            else:
                continue
            Euclidean_distance = np.linalg.norm(co_occurrence_1_prime[i] - co_occurrence_2_prime[i])
            tmp.append([Euclidean_distance, k, v])
    
        tmp = sorted(tmp, key= lambda x :x[0])
        tmp = tmp[:round(len(tmp) * threshold)]

        p_adjusted = {x[1]:x[2] for x in tmp}
        return p_adjusted