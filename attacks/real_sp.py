from collections import Counter
from tqdm import tqdm
import concurrent.futures
from functools import partial
import time
import numpy as np
from .ihop import QAP
import math

class Deduce_sp_truth:
    def __init__(self, query_id_multi_window_list, query_doc_multi_window_list,layer_match=1000) -> None:
        self.query_doc_multi_window_list = query_doc_multi_window_list
        self.query_id_multi_window_list = query_id_multi_window_list
        self.layer_match = layer_match
        self.groups_multi_window = [] #results of deduce_sp_single
        self.id_matrix_list = []
        self.M = {}
        self.ID_M = []
    def deduce_sp(self):
        for i in range(len(self.query_id_multi_window_list)):
            groups_single_window = self.deduce_sp_single(self.query_id_multi_window_list[i])
            self.groups_multi_window.append(groups_single_window)
        
        self.M = self.deduce_sp_multi()

    def deduce_sp_single(self, query_id_single_window_list):
        query_sequence = []
        for query_doc_dic_single_timeslot_list in query_id_single_window_list:
            query_number = len(query_doc_dic_single_timeslot_list)
            query_sequence.extend([i for i in range(len(query_sequence), len(query_sequence) + query_number)])

        kws_sequence = []
        for query_id_single_timeslot in query_id_single_window_list:
            kws_sequence.extend(query_id_single_timeslot)

        groups = {i:[] for i in range(self.kws_universe_size)}

        for query in query_sequence:
            groups[kws_sequence[query]].append(query)
        
        groups = [group for _, group in groups.items() if len(group) > 0]
        return groups

    def deduce_sp_multi(self):

        # get IDH, IDT, and ID
        ID_M,IDH_M,IDT_M = [],[],[]
        for i in range(len(self.query_id_multi_window_list)):
            kws_single_window = [kw for query_id_single_timeslot in self.query_id_multi_window_list[i] for kw in query_id_single_timeslot] # 一维

            query_doc_single_window = [query_doc for query_doc_single_timeslot in self.query_doc_multi_window_list[i] for query_doc in query_doc_single_timeslot]

            # get the observed doc list
            doc_list = list(set([doc for query_doc in query_doc_single_window for doc in query_doc]))
            doc_index_dic = dict(zip(doc_list, range(len(doc_list))))
            groups = self.groups_multi_window[i]

            IDH = []
            IDT = []
            ID = np.zeros((len(groups),len(doc_list)))
            for j in range(len(groups)):
                kw = kws_single_window[groups[j][0]]
                IDH.append(kw)
                IDT.append(kw)
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
            for j in range(len(self.groups_multi_window) - i):
                Matches_new = {}

                if len(self.groups_multi_window[j + i]) > len(self.groups_multi_window[j]):
                    Matches_new = self.match_two_v2(IDT_M[j],IDH_M[j + i])
                else:
                    Matches_new_inverse = self.match_two_v2(IDH_M[j + i],IDT_M[j])
                    Matches_new = {value:key for key,value in Matches_new_inverse.items()}
                for group1_id in Matches_new.keys():
                    group1 = self.groups_multi_window[j][group1_id]
                    group_list_1, windows1, index1 = self.get_merged_groups(j, group1, M)
                    group2_id = Matches_new[group1_id]
                    group2 = self.groups_multi_window[j + i][group2_id]
                    group_list_2, windows2, index2 = self.get_merged_groups(j + i, group2, M)

                    if len(set(windows1) & set(windows2))==0:
                        new_group_list = []
                        for k in range(len(group_list_1)):
                            new_group_list.append((group_list_1[k], windows1[k]))
                        for k in range(len(group_list_2)):
                            new_group_list.append((group_list_2[k], windows2[k]))
                        M.append(new_group_list)
                        del M[index1]
                        if index1>index2:
                            del M[index2]
                        else:
                            del M[index2-1]
        return M

    def get_merged_groups(self, window, group, M):
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
    
    def match_two_v2(self, kws_1, kws_2):
        p_q = {}
        for i, value in enumerate(kws_1):
            if value in kws_2:
                j = kws_2.index(value)
                if value == kws_2[j]:
                    p_q[i] = j
        return p_q

   

  

