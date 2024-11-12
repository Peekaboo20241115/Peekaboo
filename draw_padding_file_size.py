import pickle
import numpy as np
import matplotlib.pyplot as plt
def draw_padding(k_list,dataset,test_times):

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
    for k in k_list:
        with open("./results/padding/Jigsaw_plus_"+dataset+"_padding_k_"+str(k)+".pkl","rb") as f:
            Acc_Jigsaw_plus = pickle.load(f)
            ACC_Jigsaw_plus.append(Acc_Jigsaw_plus)
            ACC_Jigsaw_plus_average.append(np.average(Acc_Jigsaw_plus))
        with open("./results/padding/SAP_plus_"+dataset+"_padding_k_"+str(k)+".pkl","rb") as f:
            Acc_Sap_plus = pickle.load(f)
            ACC_Sap_plus.append(Acc_Sap_plus)
            ACC_Sap_plus_average.append(np.average(Acc_Sap_plus))

        with open("./results/padding/FMA_"+dataset+"_padding_k_"+str(k)+".pkl","rb") as f:
            Acc_FMA = pickle.load(f)
            ACC_FMA.append(Acc_FMA)
            print(k,ACC_FMA)
            ACC_FMA_average.append(np.average(Acc_FMA))

        with open("./results/padding/SPJigsaw_plus_"+dataset+"_padding_k_"+str(k)+".pkl","rb") as f:
            Acc_sp_Jigsaw_plus = pickle.load(f)
            ACC_sp_Jigsaw_plus.append(Acc_sp_Jigsaw_plus)
            ACC_sp_Jigsaw_plus_average.append(np.average(Acc_sp_Jigsaw_plus))

        with open("./results/padding/SPSAP_plus_"+dataset+"_padding_k_"+str(k)+".pkl","rb") as f:
            Acc_sp_Sap_plus = pickle.load(f)
            ACC_sp_Sap_plus.append(Acc_sp_Sap_plus)
            ACC_sp_Sap_plus_average.append(np.average(Acc_sp_Sap_plus))
    

    Position_1=np.array([1,2,3,4,5])
    Position_2=np.array([1,2,3,4,5])
  

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
    plt.xlabel("$k$")
    plt.ylabel("Accuracy")
    plt.xticks([1,2,3,4,5],["No Padding","100","200","500","1000"])
    if dataset =="enron":
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
    plt.savefig("./results/padding/test_padding_"+dataset+".pdf",bbox_inches='tight')


draw_padding([1,100,200,500,1000],"lucene",5)
draw_padding([1,100,200,500,1000],"enron",5)