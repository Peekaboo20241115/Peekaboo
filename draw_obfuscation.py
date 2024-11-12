import pickle
import numpy as np
import matplotlib.pyplot as plt
def draw_obfuscate(p, q_list,is_fvp, dataset,test_times):

    plt.rcParams.update({
    'figure.figsize': '6, 3.5',  # set figure size
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":14,
    "lines.markersize":10})
    plt.style.context(['science', 'no-latex'])
    ax = plt.axes()


    ACC_Jigsaw_plus = []
    ACC_Sap_plus = []
    ACC_FMA = []
    ACC_sp_Jigsaw_plus = []
    ACC_sp_Sap_plus = []
    ACC_Jigsaw_plus_average = []
    ACC_Sap_plus_average = []
    ACC_FMA_average = []
    ACC_sp_Jigsaw_plus_average = []
    ACC_sp_Sap_plus_average = []
    for q in q_list:
        with open("./results/obfuscate/Jigsaw_plus_"+"is_fvp_"+str(int(is_fvp))+"_"+dataset+"_p_"+str(p)+"q_"+str(q)+".pkl","rb") as f:
            Acc_Jigsaw_plus = pickle.load(f)
            ACC_Jigsaw_plus.append(Acc_Jigsaw_plus)
            ACC_Jigsaw_plus_average.append(np.average(Acc_Jigsaw_plus))
        with open("./results/obfuscate/SAP_plus_"+"is_fvp_"+str(int(is_fvp))+"_"+dataset+"p_"+str(p)+"q_"+str(q)+".pkl","rb") as f:
            Acc_Sap_plus = pickle.load(f)
            ACC_Sap_plus.append(Acc_Sap_plus)
            ACC_Sap_plus_average.append(np.average(Acc_Sap_plus))

        with open("./results/obfuscate/FMA_"+"is_fvp_"+str(int(is_fvp))+"_"+dataset+"_p_"+str(p)+"_q_"+str(q)+".pkl","rb") as f:
            Acc_FMA = pickle.load(f)
            ACC_FMA.append(Acc_FMA)
            print(q,ACC_FMA)
            ACC_FMA_average.append(np.average(Acc_FMA))

        with open("./results/obfuscate/SPJigsaw_plus_"+"is_fvp_"+str(int(is_fvp))+"_"+dataset+"p_"+str(p)+"q_"+str(q)+".pkl","rb") as f:
            Acc_sp_Jigsaw_plus = pickle.load(f)
            ACC_sp_Jigsaw_plus.append(Acc_sp_Jigsaw_plus)
            ACC_sp_Jigsaw_plus_average.append(np.average(Acc_sp_Jigsaw_plus))

        with open("./results/obfuscate/SPSAP_plus_"+"is_fvp_"+str(int(is_fvp))+"_"+dataset+"p_"+str(p)+"q_"+str(q)+".pkl","rb") as f:
            Acc_sp_Sap_plus = pickle.load(f)
            ACC_sp_Sap_plus.append(Acc_sp_Sap_plus)
            ACC_sp_Sap_plus_average.append(np.average(Acc_sp_Sap_plus))
    

    Position_1=np.array([0,1,2,3])
    Position_2=Position_1
  

    AP_lineprops = dict(linewidth=1.5,color='darkorange')
    Bp_Jigsaw_plus = ax.boxplot(ACC_Jigsaw_plus,positions=Position_1-0.2,\
            widths=0.09,patch_artist=True,boxprops=dict(facecolor="wheat",edgecolor= "darkorange"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    AP_lineprops = dict(linewidth=1.5,color='darkgreen')
    Bp_sp_Jigsaw_plus = ax.boxplot(ACC_sp_Jigsaw_plus,positions=Position_2-0.1,\
            widths=0.09,patch_artist=True,boxprops=dict(facecolor="green",edgecolor= "darkgreen"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    AP_lineprops = dict(linewidth=1.5,color='blueviolet')
    Bp_Sap_plus = ax.boxplot(ACC_Sap_plus,positions=Position_1,\
            widths=0.09,patch_artist=True,boxprops=dict(facecolor="violet",edgecolor= "blueviolet"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    AP_lineprops = dict(linewidth=1.5,color='dodgerblue')
    Bp_sp_Sap_plus = ax.boxplot(ACC_sp_Sap_plus,positions=Position_2+0.1,\
            widths=0.09,patch_artist=True,boxprops=dict(facecolor="lightskyblue",edgecolor= "dodgerblue"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    AP_lineprops = dict(linewidth=1.5,color='red')
    Bp_FMA = ax.boxplot(ACC_FMA,positions=Position_2+0.2,\
            widths=0.09,patch_artist=True,boxprops=dict(facecolor="salmon",edgecolor= "red"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    
    plt.ylim((0,1.01))
    plt.xlabel("$FPR$")
    plt.ylabel("Accuracy")
    plt.xticks([0,1,2,3],["No Obfuscation","0.01","0.02","0.05"])
    if dataset =="enron" and not is_fvp:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
                ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
                loc=1,bbox_to_anchor=(1, 0.45),\
                fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    elif dataset == "lucene" and not is_fvp:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
                ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
                loc=1, bbox_to_anchor=(1, 0.3), \
                fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    elif dataset == "enron" and is_fvp:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
                ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
                loc=1,bbox_to_anchor=(1, 1),\
                fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    else:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
        ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
        loc=1, bbox_to_anchor=(1, 1), \
        fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    #plt.show()
    plt.savefig("./results/obfuscate/test_obfuscate_"+"is_fvp_"+str(int(is_fvp))+"_"+dataset+".pdf",bbox_inches='tight')


draw_obfuscate(0.999, [0,0.01,0.02,0.05],False, "enron",5)
draw_obfuscate(0.999, [0,0.01,0.02,0.05],False, "lucene",5)
