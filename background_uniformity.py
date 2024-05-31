# Background uniformity in the ROI
# LEGEND data: p03/p04 (taup dataset)
# 01 August 2023, Elisabetta Bossio modified by Toby Dixon m
import matplotlib.pyplot as plt
import legendstyles

# use the LEGEND Matplotlib style
plt.style.use(legendstyles.LEGEND)

import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
import uproot
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages

from legendmeta import LegendMetadata


def plot_uniformity(
    fig,
    axes,
    total_bi:float,
    i_cts:np.ndarray,
    i_exp:np.ndarray,
    i_ids:np.ndarray,
    ylabel:str="Counts [cts/kg/yr]",
    figsize=(6, 4),
    title="a title",
    fontsize=8,
    maxo=0
):
    """
    Produces a plot of the background uniformity
    Parameters
    ----------
    total_bi
        float of the total bkg index
    i_cts
        the number of observed counts per category
    i_exp
        the exposure per category
    i_ids
        the labels for each category
    ylabel, figsize,title
        parameters for plotting
    """
  
    if (isinstance(i_cts,list)):
        i_cts= np.array(i_cts)
    if (isinstance(i_exp,list)):
        i_exp= np.array(i_exp)
    if (isinstance(i_ids,list)):
        i_ids= np.array(i_ids)

    expected_cts = np.multiply(i_exp, total_bi)
    i_exp = np.where(i_exp>0,i_exp,1)
    one_sigma = stats.poisson.interval(0.68, expected_cts)
    two_sigma = stats.poisson.interval(0.95, expected_cts)
    three_sigma = stats.poisson.interval(0.999, expected_cts)
    if (len(three_sigma[1])>0 and len(i_cts)>0):
        maxi = max(np.max(three_sigma[1]/(i_exp)),np.max(i_cts/(i_exp)),maxo,0.0001)*1.4
    else:
        maxi=0.01
    i_x = np.arange(0, len(i_cts), 1)
    axes.set_title(title)

    axes.fill_between(
        i_x,
        three_sigma[0] / i_exp,
        three_sigma[1] / i_exp,
        color="red",
        alpha=0.3,
        label="99.9%",
        linewidth=0,
    )
    axes.fill_between(
        i_x, two_sigma[0] / i_exp, two_sigma[1] / i_exp, color="gold", label="95%",alpha=0.3,linewidth=0,
    )
    axes.fill_between(
        i_x, one_sigma[0] / i_exp, one_sigma[1] / i_exp, color="green", label="68%",alpha=0.3,linewidth=0,
    )
    axes.errorbar(
        i_x,
        i_cts / i_exp,
        0,
        linestyle="None",
        marker="o",
        markersize=4,
        color="black",
        label="Data",
    )

    axes.plot(i_x, expected_cts / i_exp, color="black", label="Expectation")
    axes.set_ylabel(ylabel)
   
    axes.set_xticks(i_x)
    axes.set_ylim(0,maxi)
   
    axes.set_xticklabels(i_ids, rotation=90,fontsize=fontsize)
    

    return maxi/1.4

# define test
def test(n_vector:np.ndarray, mu_vector:np.ndarray)->float:
    """
    Compute test statistic
    Parameters
    ----------
    n_vector
        list of observed counts
    mu_vector:
        expectation vector
    """
    test_num = np.sum(stats.poisson.logpmf(k=n_vector, mu=mu_vector),axis=1)
    test_den = np.sum(stats.poisson.logpmf(k=n_vector, mu=n_vector),axis=1)
    return -2 * test_num + 2 * test_den


def run_test(expected_cts:np.ndarray, observed_cts:np.ndarray, plotting:bool=False,n:int=100000):
    """
    Run the tests / toy MC
    Parameters
    ----------
    expected_cts  
        array of expected counts per group
    observed_cts
        array of observed counts per group
    """
    # define structures for toy MC
    mu_vector = expected_cts
    test_statistic_values = []

    # run toy MC
    n_vector = np.random.poisson(np.tile(mu_vector,(n,1)))
    mu_vector_tile = np.tile(mu_vector,(n,1))
    test_statistic_values = test(n_vector=n_vector, mu_vector=mu_vector_tile)
    


    # compute test value for observed dataset
    dataset_test = test([observed_cts], [mu_vector])

    # test statistic distribution
    hh, bins = np.histogram(
        test_statistic_values, range=(0, 300), bins=1000, density=True
    )

    # observed pvalue
    cum = 1 - np.cumsum(hh * np.diff(bins))  # multiply for bin size

    if plotting == True:
        # plot hist with
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all")
        legendstyles.add_preliminary(ax1, color="red")
        legendstyles.legend_watermark(ax2, logo_suffix="-200")
        ax1.set_ylabel("pdf")
        ax1.grid(True)
        ax1.stairs(hh, bins)
        ax1.axvline(x=dataset_test)
        ax1.set_xlim(bins[0], bins[-1])

        ax2.set_xlabel("test")
        ax2.set_ylabel("cdf")
        ax2.grid(True)
        ax2.stairs(cum, bins)
        ax2.axvline(x=dataset_test)
        ax2.set_xlim(bins[0], bins[-1])
        plt.tight_layout()


    try:
        pvalue = cum[np.where(bins >= dataset_test)[0][0]]
        return pvalue
    except:
        return 0

def run_analysis():
    path = "/home/tdixon/LEGEND/BackgroundModel/hmixfit/inputs/data/datasets/l200a_neutrino24_v2.1.0.root"
    spec="mul_surv"
    metadb = LegendMetadata("../LEGEND/legend-metadata/")

    regions={
        #"ROI":[[1930,2014],[2064,2099],[2109,2114],[2124,2190]],
        #"two_nu":[[1000,1300]],
        #"alpha":[[3000,6000]],
        #"ar_39":[[200,500]],
        "bi_1764":[[1760,1770]],
        "tl_2615":[[2610,2620]]
    }
    
    for reg in regions:
        for mode in ["types","string","floor","chan"]:
            
            if (mode=="chan"):
                font = 5
            elif (mode=="string"):
                font=14
            else:
                font =10
            
            for spec in ["mul_surv","lar_surv","psd_surv"]:
                if (spec!="psd_surv"):
                    psd_use=False
                else:
                    psd_use =True
                pdf = PdfPages(f"plots/{spec}_{reg}_{mode}_uniformity.pdf")
                
                fig,axes =   plt.subplots(
                1, 1, figsize=(6,3), sharex=True, gridspec_kw={"hspace": 0}
                        )
                legendstyles.add_preliminary(axes, color="red")
                legendstyles.legend_watermark(axes, logo_suffix="-200")
                for sel_mode in ["all","by_type"]:
                    
                    if (sel_mode=="all"):
                        fig,axes =   plt.subplots(
                        1, 1, figsize=(8,3.5), sharex=True, gridspec_kw={"hspace": 0}
                        )
                        legendstyles.add_preliminary(axes, color="red")
                        legendstyles.legend_watermark(axes, logo_suffix="-200")

                        list_type =["all"]
                    else:
                        fig,axes =   plt.subplots(
                        1,4, figsize=(8,3.7), sharex=False, gridspec_kw={"hspace": 0,"wspace":0},sharey=True
                        )
                        legendstyles.add_preliminary(axes[0], color="red")
                        legendstyles.legend_watermark(axes[3], logo_suffix="-200")

                        list_type=["icpc","bege","ppc","coax"]
                    maxo=0
                    for id,det_type in enumerate(list_type):
                        
                        if (id==0):
                            yl= "rate [cts/kg/keV/yr]"
                        else:
                            yl=""
                        if (det_type=="all"):
                            axes_tmp = axes
                        else:
                            axes_tmp=axes[id]

                        if (det_type=="all"):
                            groups_type,_,ns = utils.get_det_types(mode,psd_usable=psd_use)
                        else:
                            groups_type,_,ns = utils.get_det_types(mode,det_type_sel=det_type,psd_usable=psd_use)
                
                        range_roi=0
                        for sub in regions[reg]:
                            range_roi +=sub[1]-sub[0]
                        # open the histo
                        file = uproot.open(path)
                        counts_tot = utils.get_data_counts_total(spec,groups_type,regions,file,key_list=list(regions.keys()))
                        
                        cats=[]
                        exps=[]
                        counts=[]

                        for idx,cat in enumerate(counts_tot):
                            if (groups_type[cat]["exposure"]<=0):
                                continue
                            exps.append(groups_type[cat]["exposure"])

                            counts.append(counts_tot[cat][reg])
                            if (mode=="chan"):
                                cats.append(utils.number2name(metadb,cat))
                            else:
                                cats.append(cat)
                        cats=np.array(cats)
                        counts=np.array(counts)
                        exps=np.array(exps)*range_roi
                        total_bi = np.sum(counts)/np.sum(exps)

                        expectation = total_bi * exps
                        
                        p = run_test(expectation,counts)
                        print(f"For {spec} {mode} p = {p*100:.3f} %")
                        if (sel_mode=="all"):
                            maxi=plot_uniformity(fig,axes_tmp,total_bi,counts,exps,cats,title=f"BI by {mode} for {spec} ({det_type} dets) (p  = {p*100:.1f} %)",fontsize=font)
                        else:
                            maxo=plot_uniformity(fig,axes_tmp,total_bi,counts,exps,cats,title=f"{det_type}: p  = {p*100:.1f} %)",fontsize=font,ylabel=yl,maxo=maxo)
                        if ((id==0)and sel_mode=="all") or (id==3 and sel_mode=="by_type"):
                            axes_tmp.legend(loc="upper right",fontsize=10)
                        
                    if (sel_mode!="all"):
                        fig.suptitle(f"Rate for {mode} for {spec}")
                    plt.tight_layout()
                    pdf.savefig()
                pdf.close()

if __name__ == "__main__":
    run_analysis()
