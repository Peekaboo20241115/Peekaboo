import pickle
import numpy as np
import matplotlib.pyplot as plt
def draw_comparison(Cycles_number,is_fvp,dataset,test_times):

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
    for cycles_number in Cycles_number:
        with open("./results/comparison/Jigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","rb") as f:
            Acc_Jigsaw_plus = pickle.load(f)
            ACC_Jigsaw_plus.append(Acc_Jigsaw_plus)
            ACC_Jigsaw_plus_average.append(np.average(Acc_Jigsaw_plus))
        with open("./results/comparison/SAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","rb") as f:
            Acc_Sap_plus = pickle.load(f)
            ACC_Sap_plus.append(Acc_Sap_plus)
            ACC_Sap_plus_average.append(np.average(Acc_Sap_plus))

        with open("./results/comparison/FMA_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","rb") as f:
            Acc_FMA = pickle.load(f)
            ACC_FMA.append(Acc_FMA)
            ACC_FMA_average.append(np.average(Acc_FMA))

        with open("./results/comparison/SPJigsaw_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","rb") as f:
            Acc_sp_Jigsaw_plus = pickle.load(f)
            ACC_sp_Jigsaw_plus.append(Acc_sp_Jigsaw_plus)
            ACC_sp_Jigsaw_plus_average.append(np.average(Acc_sp_Jigsaw_plus))

        with open("./results/comparison/SPSAP_plus_"+dataset+"_is_fvp_"+str(int(is_fvp))+"_cycles_numer_"+str(cycles_number)+".pkl","rb") as f:
            Acc_sp_Sap_plus = pickle.load(f)
            ACC_sp_Sap_plus.append(Acc_sp_Sap_plus)
            ACC_sp_Sap_plus_average.append(np.average(Acc_sp_Sap_plus))
    

    Position_1=np.array([1,2,3,4])
    Position_2=np.array([1,2,3,4])

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
    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy")
    plt.xticks([1,2,3,4],[1,2,4,8])
    if dataset=="Enron" and is_fvp == False:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
                ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
                loc=1, bbox_to_anchor=(1,0.5),\
                fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    elif dataset=="Lucene" and is_fvp == False:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
                ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
                loc=4,\
                fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    else:
        plt.legend([Bp_Jigsaw_plus["boxes"][0], Bp_sp_Jigsaw_plus["boxes"][0],Bp_Sap_plus["boxes"][0], Bp_sp_Sap_plus["boxes"][0],Bp_FMA["boxes"][0]],\
                ["Jigsaw+","Jigsaw+ with SP","Sap+","Sap+ with SP","FMA"],\
                loc=1,\
                fancybox=True, shadow=True,ncol=3,prop={'size': 11.5},columnspacing=0.3,handletextpad=0.3)
    
 

    #plt.show() 
    
    plt.savefig("./results/comparison/test_comparison_"+dataset+"_isfvp_"+str(int(is_fvp))+".pdf",bbox_inches='tight')

draw_comparison([1,2,4,8],False,"Enron",5)
draw_comparison([1,2,4,8],True,"Enron",5)
draw_comparison([1,2,4,8],False,"Lucene",5)
draw_comparison([1,2,4,8],True,"Lucene",5)