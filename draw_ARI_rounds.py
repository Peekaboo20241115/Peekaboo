import pickle
import numpy as np
import matplotlib.pyplot as plt
def draw(dataset):
    with open("results/peekaboo_ari_test_2_"+dataset+"_isfvp_"+str(int(True))+".pkl", "rb") as f:
        ARI_fvp,RI_fvp = pickle.load(f)
    with open("results/peekaboo_ari_test_2_"+dataset+"_isfvp_"+str(int(False))+".pkl", "rb") as f:
        ARI_ap,RI_ap= pickle.load(f)
    
    plt.rcParams.update({
    'figure.figsize': '6, 3',  # set figure size
    "text.usetex": True,
    "font.family": "stix",
    "font.serif": ["Times"],
    "font.size":25,
    "lines.markersize":20})
    plt.style.context(['science', 'no-latex'])
    ax = plt.axes()

    ARI_ap_reshape = [ari_temp[0] for ari_temp in ARI_ap]
    ARI_fvp_reshape = [ari_temp[0] for ari_temp in ARI_fvp]

    timeslot_numbers = [1,2,4,8]
    Positions_AP = np.array([1,2,3,4,5])-0.175
    AP_lineprops = dict(linewidth=1.5,color='darkgreen')
    Bp_AP = ax.boxplot(ARI_ap_reshape,positions=Positions_AP,\
            widths=0.3,patch_artist=True,boxprops=dict(facecolor="green",edgecolor= "darkgreen"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    Positions_FVP = np.array([1,2,3,4,5])+0.175
    AP_lineprops = dict(linewidth=1.5,color='darkorange')
    Bp_FVP = ax.boxplot(ARI_fvp_reshape,positions=Positions_FVP,\
            widths=0.3,patch_artist=True,boxprops=dict(facecolor="wheat",edgecolor= "darkorange"),whiskerprops=AP_lineprops,
            medianprops=AP_lineprops,capprops=AP_lineprops,showfliers=True)
    
    plt.xticks([1,2,3,4,5],[1,2,4,8,16])
    plt.ylim((0.72,1.01))
    
    plt.ylabel("ARI")
    plt.xlabel("Number of Rounds")
   
    plt.legend([Bp_AP["boxes"][0], Bp_FVP["boxes"][0]],\
            ["AP","FVP"],\
            #loc="center left", bbox_to_anchor=(-0.015, 0.5),\
            fancybox=True, shadow=True)
    #plt.show()
    plt.savefig("./results/test_peekaboo_2_"+dataset+".pdf",bbox_inches='tight')


draw("lucene")
draw("enron")
    