import numpy as np
class Jigsawattacker:

    def __init__(self, sim_F=None, real_F=None, sim_V=None, real_V=None, sim_M=None, real_M=None,
                 sim_doc_num=None, real_doc_num=None,
                 no_F=False, baseRec=30, confRec=10, refinespeed=15,
                 alpha=0.5, beta=0.8, countermeasure_params={"alg": None}, refinespeed_exp=None,is_fvp=False):
    
        self.sim_doc_num = sim_doc_num
        self.real_doc_num = real_doc_num

        self.num_interval = len(real_F[0])
        self.cycle_number = len(sim_M)
        self.sim_F = np.array(sim_F,dtype=float)
        self.real_F = np.array(real_F,dtype=float)
        self.sim_V = np.array(sim_V,dtype=float)
        self.real_V = np.array(real_V,dtype=float)
        self.sim_M = np.array(sim_M,dtype=float)
        self.real_M = np.array(real_M,dtype=float)

 
        # normalizations
        if is_fvp==False:
            for i in range(self.num_interval):
                self.sim_F[:, i] = self.sim_F[:,i]/np.sum(self.sim_F[:,i])
                self.real_F[:, i] = self.real_F[:,i]/np.sum(self.real_F[:,i])
                self.sim_V[:, i] = self.sim_V[:,i]/self.sim_doc_num[i]
                self.real_V[:, i] = self.real_V[:,i]/self.real_doc_num[i]
        else:
            for i in range(self.num_interval):
                self.sim_V[:, i] = self.sim_V[:,i]/np.mean(self.sim_V[:,i])
                self.real_V[:, i] = self.real_V[:,i]/np.mean(self.real_V[:,i])
                self.sim_F[:, i] = self.sim_F[:,i]/np.mean(self.sim_F[:,i])
                self.real_F[:, i] = self.real_F[:,i]/np.mean(self.real_F[:,i])

        for i in range(self.cycle_number):
            min_non_zero = np.min(self.sim_M[i][self.sim_M[i] > 0])/10000
            self.sim_M[i][self.sim_M[i] == 0] = min_non_zero

        for i in range(self.cycle_number):
            min_non_zero = np.min(self.real_M[i][self.real_M[i] > 0])/10000
            self.real_M[i][self.real_M[i] == 0] = min_non_zero

            
        
        
        self.no_F = no_F
        self.BaseRec = baseRec
        self.ConfRec = confRec
        self.refinespeed = refinespeed
        self.refinespeed_exp = refinespeed_exp
        self.alpha = alpha
        self.beta = beta
    
        # del sim_kw_d
        self.tdid_2_kwsid = {}
        self.tdid_2_kwsid_step1 = {}
        self.tdid_2_kwsid_step2 = {}
        self.tdid_2_kwsid_step3 = {}
        self.unrec_td_set = set([i for i in range(len(self.real_M[0]))])
        self.id_known_kws = None
        self.id_queried_kws = None

    def attack_step_1(self):
        # Jigsaw Step1:Locating and recovering the distinctive queries by Volume and/or Frequency
        D_FV = self.calculate_dVF()
        id_Diff = []
        for i in range(len(D_FV)):
            id_Diff.append([i, D_FV[i]])
        id_Diff.sort(key=lambda x: x[1], reverse=True)
        top = id_Diff[:self.BaseRec]
        top_td_list = [i[0] for i in top]

        tdid_2_kwsid = self.recover_by_VF(top_td_list)
        self.tdid_2_kwsid_step1 = tdid_2_kwsid

    def attack_step_2(self):
        # Jigsaw Step2:Verify by co-occurance
        tdid_2_kwsid = self.verify_by_M()
        self.tdid_2_kwsid.update(tdid_2_kwsid)
        self.tdid_2_kwsid_step2 = tdid_2_kwsid
        self.unrec_td_set = self.unrec_td_set - set(tdid_2_kwsid.keys())

    def attack_step_3(self):
        #Jigsaw Step3:Using co-occurance to recover remaining queries
        while (len(self.unrec_td_set) > 0):
            paired_td = list(self.tdid_2_kwsid.keys())
            paired_kw = [self.tdid_2_kwsid[i] for i in paired_td]
            unpaired_kw = list(set([i for i in range(len(self.sim_M[0]))]) - set(paired_kw))
            un_td_list = list(self.unrec_td_set)

            M = []
            M_ = []
            for t in range(self.cycle_number):
                Mtmp = self.real_M[t][un_td_list][:, paired_td]
                M_tmp = self.sim_M[t][unpaired_kw][:, paired_kw]
                M.append(Mtmp / Mtmp.sum(axis=1).reshape((len(Mtmp), 1)))
                M_.append(M_tmp / M_tmp.sum(axis=1).reshape((len(M_tmp), 1)))

            sim_F = self.sim_F[unpaired_kw, :]
            real_F = self.real_F[un_td_list, :]
            sim_V = self.sim_V[unpaired_kw, :]
            real_V = self.real_V[un_td_list, :]

            Certainty = []
            for i in range(len(M[0])):
                score = np.zeros(len(sim_F), dtype=float)
                S1 = np.zeros(len(sim_F), dtype=float)
                S2 = np.zeros(len(sim_F), dtype=float)
                for t in range(self.num_interval):
                    S1 = S1 + self.alpha * np.abs(real_V[i][t] - sim_V[:, t]) + (1 - self.alpha) * (
                        np.abs(real_F[i][t] - sim_F[:, t]))
                for t in range(self.cycle_number):
                    S2 = S2 + np.linalg.norm(M[t][i, :] - M_[t], axis=1)
                    
                score = score - np.log(self.beta/self.cycle_number * S2 + (1 - self.beta)/self.num_interval * (S1))
                candidate_w = np.argmax(score)
                score = sorted(score, reverse=True)
                certainty = score[0] - score[1]
                Certainty.append([i, certainty, candidate_w])
            Certainty.sort(key=lambda x: x[1], reverse=True)
            if len(Certainty) < self.refinespeed:
                top_td = [Certainty[i][0] for i in range(len(Certainty))]
                Matched_w = [Certainty[i][2] for i in range(len(Certainty))]
            else:
                top_td = [Certainty[i][0] for i in range(int(self.refinespeed))]
                Matched_w = [Certainty[i][2] for i in range(int(self.refinespeed))]
            
            tdid_2_kwsid = {}
            for i in range(len(top_td)):
                tdid_2_kwsid[un_td_list[top_td[i]]] = unpaired_kw[Matched_w[i]]

            self.tdid_2_kwsid.update(tdid_2_kwsid)
            self.tdid_2_kwsid_step3.update(tdid_2_kwsid)
            self.unrec_td_set = self.unrec_td_set - set(tdid_2_kwsid.keys())
            if self.refinespeed_exp:  
                self.refinespeed = self.refinespeed * 1.1
          
        return self.tdid_2_kwsid

    def calculate_dVF(self):

        td_nb = len(self.real_M[0])
        D_FV = np.zeros(td_nb)
        for i in range(td_nb):
            d_F = np.zeros(len(self.real_F), dtype=float)
            d_V = np.zeros(len(self.real_V), dtype=float)
            for t in range(self.num_interval):
                d_F += np.abs(self.real_F[:, t] - self.real_F[i][t])
                d_V += np.abs(self.real_V[:, t] - self.real_V[i][t])
            if not self.no_F:
                d_FV = self.alpha * d_V + (1 - self.alpha) * d_F
            else:
                d_FV = d_V
            d_FV[i] = float("inf")
            D_FV[i] = np.min(d_FV)
        return D_FV

    def recover_by_VF(self, td_list):

        # using volume and frequency to recover queries in td_list
    
        tmp_pair = {}
        for i in range(len(td_list)):
            s1 = np.zeros(len(self.sim_V), dtype=float)
            s2 = np.zeros(len(self.sim_V), dtype=float)
            for t in range(self.num_interval):
                s1 += np.abs(self.sim_V[:, t] - self.real_V[td_list[i], t])
            if not self.no_F:
                for t in range(self.num_interval):
                    s2 += np.abs(self.sim_F[:, t] - self.real_F[td_list[i], t])
                s = self.alpha * s1 + (1 - self.alpha) * s2
            else:
                s = s1
            tmp_pair[td_list[i]] = np.argmin(s)
        return tmp_pair

    def verify_by_M(self):
        # using co-occurance matrix to verify paired queries
        tmp_pair = self.tdid_2_kwsid_step1.copy()
        nb = len(tmp_pair)
        pair = tmp_pair.copy()
        pair_list = []
        for i in tmp_pair.keys():
            pair_list.append([i, tmp_pair[i]])
        td = [i[0] for i in pair_list]
        kw = [i[1] for i in pair_list]
        Dis = [0] * nb
        for t in range(self.cycle_number):
            td_M = self.real_M[t][td][:, td]
            kw_M = self.sim_M[t][kw][:, kw]
       
            td_M = td_M / td_M.sum(axis=1).reshape((nb, 1))
            kw_M = kw_M / kw_M.sum(axis=1).reshape((nb, 1))

            for i in range(nb):
                Dis[i] += np.linalg.norm(td_M[i] - kw_M[i])

        flag = sorted(Dis, reverse=True)[self.BaseRec - self.ConfRec]
        for i in range(len(Dis)):
            if Dis[i] > flag:
                del (pair[pair_list[i][0]])
        return pair
