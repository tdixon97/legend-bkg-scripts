"""
remove_ob.py
Author: Toby Dixon
A script to give the sensitivity to removing the IB

Statistical formulation
----------------------------------------------
1) Lets suppose the counting rate in some region (gamma line, ROI etc) is composed of 2 sources:
    eg A and B with rates, muA, muB, (for example A could be OB and B all others) (counts/kg-yr unit)
2) Suppose we measured N1 counts in an exposure of MT1 in one configuation with both OB and other sources,
and that we measured N2 counts in exposure MT2 in another without OB.
3) The likelihood of this is:
    L = P(N1|muA+muB)*P(N2|muB) = Pois(N1/MT1|muA+muB)*Pois(N2/MT2|muB)
4) We invert with Bayes Thereom to get the 2D posterior on muA,muB
    P(muA,muB)=  Pois(N1/MT1|muA+muB)*Pois(N2/MT2|muB)*P(muA)*P(muB)/N
    this could be obtained numerically (MCMC) but can also just be plotted.
5) From this we can obtained marginalised probability distributions of P(muA) and P(muB)
"""


from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')

from collections import OrderedDict
import uproot
from hist import Hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
import hist
import argparse
import re
import json
import time
import warnings
import sys
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import truncnorm
from scipy.stats import poisson
from scipy.stats import gamma

vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
lset = tc.tol_cset('light')

cmap=tc.tol_cmap('iridescent')
cmap.set_under('w',1)
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))
 
import os

def likelihood(muA:float,muB:float,N1:float,N2:float,MT1,MT2,NB1=0,NB2=0):
    """
    The on-off analysis likelihood
    """
   
    

    #return poisson.pmf(N1,(muA+muB)*MT1)*poisson.pmf(N2,muB*MT2)
    return gamma.pdf((muA+muB)*MT1+NB1,N1+1+NB1)*gamma.pdf(muB*MT2+NB2,N2+1+NB2)

def likelihood2(mu:float,N:float,MT,NB=0):
    """
    The on-off analysis likelihood
    """

    return gamma.pdf(mu*MT+NB,N+1+NB)

style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}
### first a quick test

def get_quantiles(counts,bins):
    cumulative_counts = np.cumsum(counts)
    # Calculate total number of data points
    total_count = np.sum(counts)

    # Calculate quantile percentages
    quantile_percentages = [16,50,84]

    # Calculate quantile bin indices
    quantile_bins = [np.argmax(cumulative_counts >= q / 100 * total_count) for q in quantile_percentages]
    mode = bins[np.argmax(counts)]

    ### get the smallest credible interval
    integral = counts[np.argmax(counts)]
    bin_id_l = np.argmax(counts)
    bin_id_u = np.argmax(counts)

    integral_tot = np.sum(counts)
    while integral<0.683*integral_tot:

        ### get left bin
        if (bin_id_l>0 and bin_id_l<len(counts)-1):
            c_l =counts[bin_id_l-1]
        else:
            c_l =0

        if (bin_id_u>0 and bin_id_u<len(counts)-1):
            c_u =counts[bin_id_u+1]
        else:
            c_u =0

        if (c_l>c_u):
            integral+=c_l
            bin_id_l-=1
        else:
            integral+=c_u
            bin_id_u+=1
        
    low_quant = bins[bin_id_l]
    high_quant=bins[bin_id_u]
        

    #quantiles = [bins[idx] for idx in quantile_bins]
    return [low_quant,mode,high_quant]

def run_example(N1,frac,MT2,MT1=44,label="Tl",make_plots=True,pdf=None):
    
    MT1 = 44
    mu1 = N1/MT1


    mu2 = mu1*frac
    N2 = mu2*MT2

    histo = ( Hist.new.Reg(200, 0, 1.1*mu1).Reg(200,0,1.1*mu1).Double())

    w,x,y = histo.to_numpy()

    x_2d = np.tile(x, (len(y), 1))
    y_2d = np.tile(y, (len(x), 1)).T

    w=likelihood(x_2d,y_2d,N1,N2,MT1,MT2)
    maxi = np.max(w)

    w_x = np.sum(w,axis=1)
    w_y=np.sum(w,axis=0)
    qx=get_quantiles(w_x,x)   
    qy=get_quantiles(w_y,y)
    exm = qx[1]-qx[0]
    exp=qx[2]-qx[1]
    eym = qy[1]-qy[0]
    eyp=qy[2]-qy[1]
    
    if (make_plots):
        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})

        mesh = axes.pcolormesh(x, y, w.T/maxi, cmap="PuRd")
        cbar =fig.colorbar(mesh)
        cbar.set_label("Probability")
        axes.set_xlabel("$\mu_{other}$ [cts/day]")
        axes.set_ylabel("$\mu_{OB}$ [cts/day]")

        ## set limits
        
        axes.set_xlim(qx[1]-3*exm,qx[1]+3*exp)
       
        axes.set_ylim(qy[1]-3*eym,qy[1]+3*eyp)

        axes.set_title("{:.0f}% reduction, {} kg-yr, {}".format(100-100*frac,MT2,label),fontsize=8)
        histo_x = ( Hist.new.Reg(200, 0, 1.1*mu1).Double())
        histo_y = ( Hist.new.Reg(200, 0, 1.1*mu1).Double())

        for i in range(histo_x.size-2):
            histo_x[i]=w_x[i]
            histo_y[i]=w_y[i]

        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
        axes.set_xlabel("$\mu$ [cts/kg-yr]")
        axes.set_ylabel("Prob [arb units]")
    
        axes.axvline(x=mu1, color='b', linestyle='--',label="Current rate")
        histo_x.plot(ax=axes,**style,histtype="fill",alpha=0.5,label="Other sources")
        histo_y.plot(ax=axes,**style,histtype="fill",alpha=0.5,label="OB")

        axes.set_title("{:.0f}% reduction, {} kg-yr,  {}".format(100-100*frac,MT2,label),fontsize=8)
        plt.legend(loc="upper left",frameon=True,facecolor="white")

        axes.set_xlim(0,1.2*max(qx[0]+4*exp,qy[0]+4*eyp,mu1))

        axes.set_ylim(0,1.5*max(np.max(w_x),np.max(w_y)))
        if (pdf is None):
            plt.show()
        else:
            pdf.savefig()
    plt.close()
    return (get_quantiles(w_x,x)[1],exm,exp),(get_quantiles(w_y,y)[1],eym,eyp)

def run_example_bkg(mu,muB,MT,label="Tl",make_plots=True,pdf=None):
    
    N=mu*MT
    histo = ( Hist.new.Reg(200, 0, 2*(N+5)/MT).Double())

    w,x = histo.to_numpy()

    

    w=likelihood2(x,N,MT,NB=muB*MT)
    maxi = np.max(w)

    for i in range(histo.size-2):
        histo[i]=w[i]
    qx=get_quantiles(w,x)   
    exm = qx[1]-qx[0]
    exp=qx[2]-qx[1]
    
    
    if (make_plots):
       

        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
        axes.set_xlabel("$\mu$ [cts/kg-yr]")
        axes.set_ylabel("Prob [arb units]")
    
        axes.axvline(x=mu, color='b', linestyle='--',label="Current rate")
        histo.plot(ax=axes,**style,histtype="fill",alpha=0.5)
       

        axes.set_title("{:.0f}% reduction, {} kg-yr,  {}".format(100-100*frac,MT,label),fontsize=8)
        plt.legend(loc="upper left",frameon=True,facecolor="white")

        axes.set_xlim(0,1.2*max(qx[0]+4*exp,mu))

        if (pdf is None):
            plt.show()
        else:
            pdf.savefig()
    plt.close()
    return (get_quantiles(w,x)[1],exm,exp)


def run_example_bkg_2D(mu1,mu2,muB,MT,MT2,label="Tl",make_plots=True,pdf=None):
    
    N1=mu1*MT
    N2=mu2*MT2
    histo = ( Hist.new.Reg(200, 0, 1.1*mu1).Reg(200,0,1.1*mu1).Double())

    w,x,y = histo.to_numpy()

    x_2d = np.tile(x, (len(y), 1))
    y_2d = np.tile(y, (len(x), 1)).T

    w=likelihood(x_2d,y_2d,N1,N2,MT,MT2,NB1=0,NB2=MT2*muB)
    maxi = np.max(w)

    w_x = np.sum(w,axis=1)
    w_y=np.sum(w,axis=0)
    qx=get_quantiles(w_x,x)   
    qy=get_quantiles(w_y,y)
    exm = qx[1]-qx[0]
    exp=qx[2]-qx[1]
    eym = qy[1]-qy[0]
    eyp=qy[2]-qy[1]
    
    
    
    
    return (get_quantiles(w_x,x)[1],exm,exp),(get_quantiles(w_y,y)[1],eym,eyp)

def run_example_1D(N1,frac,MT2,NB=0,MT1=44,label="Tl",make_plots=True,pdf=None):
    
    mu1 = N1/MT1


    mu2 = mu1*frac
    N2 = mu2*MT2
    histo = ( Hist.new.Reg(200, 0, 2*(N2+5)/MT2).Double())

    w,x = histo.to_numpy()

    

    w=likelihood2(x,N2,MT2)
    maxi = np.max(w)

    for i in range(histo.size-2):
        histo[i]=w[i]
    qx=get_quantiles(w,x)   
    exm = qx[1]-qx[0]
    exp=qx[2]-qx[1]
    
    
    if (make_plots):
       

        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
        axes.set_xlabel("$\mu$ [cts/kg-yr]")
        axes.set_ylabel("Prob [arb units]")
    
        axes.axvline(x=mu1, color='b', linestyle='--',label="Current rate")
        histo.plot(ax=axes,**style,histtype="fill",alpha=0.5)
       

        axes.set_title("{:.0f}% reduction, {} kg-yr,  {}".format(100-100*frac,MT2,label),fontsize=8)
        plt.legend(loc="upper left",frameon=True,facecolor="white")

        axes.set_xlim(0,1.2*max(qx[0]+4*exp,mu1))

        if (pdf is None):
            plt.show()
        else:
            pdf.savefig()
    plt.close()
    return (get_quantiles(w,x)[1],exm,exp)

def effect_of_bkg(r_sig,r_bkg,ref_p38,ref_p10,max_exp,label,frac,save_name):
    """
    Function to plot the uncertainity on a counting rate vs time and bkg
    Parameters:
    - r_sig: The rate of signal events in cts/kg-yr
    - r_bkg: The rate of background in cts/kg-yr
    - ref:   Reference to compare to 
    - max_exp: length of time (days)
    """

    med=[]
    low=[]
    high=[]
    exp=np.linspace(0.1,max_exp,2000)    

    for exp_tmp in exp:
        n_ot=run_example_bkg(r_sig,r_bkg,exp_tmp,label=label,make_plots=False)


        med.append(n_ot[0]/ref_p38[0]/365)
        low.append(n_ot[1]/ref_p38[0]/365)
        high.append(n_ot[2]/ref_p38[0]/365)

    fig, axes = lps.subplots(1, 1, figsize=(4,3), sharex=True, gridspec_kw = {'hspace': 0})

    axes.fill_between(exp,-np.array(low)+np.array(med),np.array(high)+np.array(med), color=vset.cyan, alpha=0.4,label="Expected p11 ",linewidth=0)
    axes.axhline(y=1,linestyle="--",color=vset.red)
    axes.fill_between(exp,1-ref_p38[1]/ref_p38[0],1+ref_p38[2]/ref_p38[0], color=vset.red, alpha=0.4,label="p3-8",linewidth=0)
    axes.axhline(y=ref_p10[0]/ref_p38[0],linestyle="--",color="green")
    axes.fill_between(exp,ref_p10[0]/ref_p38[0]-ref_p10[1]/ref_p38[0],ref_p10[0]/ref_p38[0]+ref_p10[2]/ref_p38[0], color=vset.teal, alpha=0.4,label="p10",linewidth=0)

    axes.plot(exp,med,color=vset.blue,linestyle="--")
    axes.set_xlabel("Exposure [kg-yr]")
    axes.set_ylabel("Normalised rate")
    axes.set_xlim(1,max_exp)
    axes.set_ylim(0,1.5)
    axes.set_xscale("linear")
    axes.set_title("{:.0f}% of counts from MS,  {}".format(100*frac,label),fontsize=8)
    axes.legend(loc="best")
    plt.savefig(f"plots/{save_name}")


def effect_of_bkg_2D(mu1,mu2,MT1s,muB,max_exp,label,frac,ref):
    """
    Function to plot the uncertainity on a counting rate vs time and bkg
    Parameters:
    - r_sig: The rate of signal events in cts/kg-yr
    - r_bkg: The rate of background in cts/kg-yr
    - ref:   Reference to compare to 
    - max_exp: length of time (days)
    """
    fig, axes = lps.subplots(1, 1, figsize=(4,3), sharex=True, gridspec_kw = {'hspace': 0})
    colors=[vset.red,vset.magenta,vset.blue]
    for MT1,c in zip(MT1s,colors):
        med=[]
        low=[]
        high=[]
        exp=np.linspace(0.1,max_exp,2000)    

        for exp_tmp in exp:
            n_ot,n_ms=run_example_bkg_2D(mu1,mu2,muB,MT=MT1,MT2=exp_tmp,label=label,make_plots=False)


            med.append(n_ms[0]/ref)
            low.append(n_ms[1]/ref)
            high.append(n_ms[2]/ref)


        axes.fill_between(exp,-np.array(low)+np.array(med),np.array(high)+np.array(med), alpha=0.4,label=f"{MT1} kg-yr p10",linewidth=0,color=c)
    
        axes.plot(exp,med,linestyle="--",color=c)
        axes.set_xlabel("Exposure p11 [kg-yr]")
        axes.set_ylabel("Normalised rate from NMS")
        axes.set_xlim(1,max_exp)
        axes.set_ylim(0,10)
        axes.set_xscale("linear")
        axes.set_title("{:.0f}% of counts from MS,  {}".format(100*frac,label),fontsize=8)
        axes.legend(loc="best")
    plt.show()




def run_effect_of_bkg_2D(mode,muB,label,frac,exps):
    if (mode=="1764"):
        ref_p38= (0.0097,0.0007,0.0007)
        ref_p10 =(0.006,0.002,0.002)
        mu=0.0097*365

        mu_pred = (ref_p10[0]/ref_p38[0]-frac)*mu

        effect_of_bkg_2D(ref_p10[0]*365,mu_pred,exps,muB,100,label,frac,ref=ref_p38[0]*365)
    elif (mode=="2615"):
        ref_p38= (0.0176, 0.0009 , 0.0009)
        ref_p10 =(0.017 ,0.003 ,0.003 )
        mu=0.0176*365
        mu_pred = (ref_p10[0]/ref_p38[0]-frac)*mu

        effect_of_bkg_2D(ref_p10[0]*365,mu_pred,exps,muB,100,label,frac,ref=ref_p38[0]*365)

def run_effect_of_bkg(mode,muB,label,frac,save_name):

    if (mode=="1764"):

        ref_p38= (0.0097,0.0007,0.0007)
        ref_p10 =(0.006,0.002,0.002)
        mu=0.0097*365
        mu_pred = (ref_p10[0]/ref_p38[0]-frac)*mu
        print(mu_pred)
        effect_of_bkg(mu_pred,muB,ref_p38,ref_p10,30,"$^{214}$Bi 1764 keV "+"{}".format(label),frac,save_name)
    elif (mode=="2615"):
        ref_p38= (0.0176, 0.0009 , 0.0009)
        ref_p10 =(0.017 ,0.003 ,0.003 )
        mu=0.0176*365
        mu_pred = (ref_p10[0]/ref_p38[0]-frac)*mu
        print(mu_pred)
        effect_of_bkg(mu_pred,muB,ref_p38,ref_p10,30,"$^{208}$Tl 2615 keV "+"{}".format(label),frac,save_name)

def main():
    do=True
    colors=[vset.orange,vset.blue,vset.magenta,vset.teal,"grey"]

    if (do):
        N_Tl=297
        N_Bi=171
        for type_analysis in ["Tl","Bi"]:

            ### save the plots
            with PdfPages("plots/remove_ob_{}_gamma.pdf".format(type_analysis)) as pdf:

                ### run the exampls
                if (type_analysis=="Tl"):
                    run_example(N_Tl,0.1,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)
                    run_example(N_Tl,0.2,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)
                    run_example(N_Tl,0.3,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)
                    run_example(N_Tl,0.5,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)
                    run_example(N_Tl,0.7,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)
                    run_example(N_Tl,0.9,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)
                    run_example(N_Tl,1,10,label="$^{208}$Tl (2615 keV)",pdf=pdf)

                    label="$^{208}$Tl (2615 keV)"
                    N=N_Tl

                elif (type_analysis=="Bi"):

                    run_example(N_Bi,0.1,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)
                    run_example(N_Bi,0.2,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)

                    run_example(N_Bi,0.3,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)
                    run_example(N_Bi,0.5,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)
                    run_example(N_Bi,0.7,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)
                    run_example(N_Bi,0.9,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)
                    run_example(N_Bi,1,10,label="$^{214}$Bi (1764 keV)",pdf=pdf)

                    label="$^{214}$Bi (1764 keV)"
                    N=N_Bi

                exp=np.linspace(0.1,30,2000)    
                res_ob={}
                res_ot={}
                for frac in [0,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1]:
                    med=[]
                    low=[]
                    high=[]

                    med_ot=[]
                    low_ot=[]
                    high_ot=[]
                    for exp_tmp in exp:
                        if (type_analysis=="Tl"):
                            n_ot,n_ob=run_example(N_Tl,frac,exp_tmp,label="$^{208}$Tl (2615 keV)",make_plots=False)
                        else:
                            n_ot,n_ob=run_example(N_Bi,frac,exp_tmp,label="$^{214}$Bi (1764 keV)",make_plots=False)

                        med.append(n_ob[0])
                        low.append(n_ob[1])
                        high.append(n_ob[2])


                        med_ot.append(n_ot[0])
                        low_ot.append(n_ot[1])
                        high_ot.append(n_ot[2])

                    res_ob[frac]=[med,low,high]
                    res_ot[frac]=[med_ot,low_ot,high_ot]

                    fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})

                    axes.fill_between(exp,-np.array(low)+np.array(med),np.array(high)+np.array(med), color=vset.cyan, alpha=0.4,label="OB",linewidth=0)
                    axes.axhline(y=N/44,color="b",label="Current rate",linestyle="--")
                    axes.plot(exp,med,color=vset.blue)
                    axes.set_xlabel("Exposure [kg-yr]")
                    axes.set_ylabel("$\mu$ [kg-yr]")
                    axes.set_title("{:.0f}% reduction,  {}".format(100-100*frac,label),fontsize=8)

                    axes.set_ylim(0,1.6*N/44)
                    axes.set_xlim(0.1,20)
                    axes.fill_between(exp,-np.array(low_ot)+np.array(med_ot),np.array(high_ot)+np.array(med_ot), color=vset.orange, alpha=0.4,label="other",linewidth=0)
                    axes.plot(exp,med_ot,color=vset.red)
                    plt.legend(frameon=True,facecolor="white")
                    pdf.savefig()

                fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
                colors=[vset.orange,vset.blue,vset.magenta,vset.teal,"grey"]
                
                for col,frac in zip(colors,[0.1,0.3,0.7,0.9]):
                    low = res_ob[frac][1]
                    med =res_ob[frac][0]
                    high=res_ob[frac][2]
                    axes.fill_between(exp,-np.array(low)+np.array(med),np.array(high)+np.array(med),color=col, alpha=0.3,label="{:.0f}% reduce".format(100-100*frac),linewidth=0)
                    axes.plot(exp,med,color=col)
                    axes.set_xlabel("Exposure [kg-yr]")
                    axes.set_ylabel("$\mu_{ob}$ [kg-yr]")
                    axes.set_title("OB rate {}".format(label))
                    axes.set_ylim(0,1.2*N/44)
                    axes.set_xlim(0.1,20)
                    axes.axhline(y=N/44,color="b",linestyle="--")
                    axes.legend(loc='upper right',edgecolor="black",frameon=True, facecolor='white',framealpha=1,ncol=1,fontsize=6
                                                    )
                pdf.savefig()    
                fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
                colors=[vset.orange,vset.blue,vset.magenta,vset.teal,"grey"]
                
                for col,frac in zip(colors,[0.1,0.3,0.7,0.9]):
                    low = res_ot[frac][1]
                    med =res_ot[frac][0]
                    high=res_ot[frac][2]
                    axes.fill_between(exp,-np.array(low)+np.array(med),np.array(high)+np.array(med),color=col, alpha=0.3,label="{:.0f}% reduce".format(100-100*frac),linewidth=0)
                    axes.plot(exp,med,color=col)
                    axes.set_xlabel("Exposure [kg-yr]")
                    axes.set_ylabel("$\mu_{ot}$ [kg-yr]")
                    axes.set_title("Other rate {}".format(label))
                    axes.set_ylim(0,1.2*N/44)
                    axes.set_xlim(0.1,20)
                    axes.axhline(y=N/44,color="b",linestyle="--")
                    axes.legend(loc='upper right',edgecolor="black",frameon=True, facecolor='white',framealpha=1,ncol=1,fontsize=6)
                                
                pdf.savefig()



    #### look
    type_analysis="bkg"
    exp=np.linspace(0.1,500,1000)    
    res_ob={}
    res_ot={}
    for bi in [8E-4,6E-4,2E-4]:
        med=[]
        low=[]
        high=[]

        med_ot=[]
        low_ot=[]
        high_ot=[]
        frac = bi/(5/31/190)
        for exp_tmp in exp:
            if (type_analysis=="bkg"):
            
                n_ot=run_example_1D(5,frac,exp_tmp,label="$^{208}$Tl (2615 keV)",make_plots=False,MT1=31)
            med_ot.append(n_ot[0])
            low_ot.append(n_ot[1])
            high_ot.append(n_ot[2])
    
    
    
        res_ot[bi]=[med_ot,low_ot,high_ot]

    fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
    ### plot the bkg index
    ###
    for col,bi in zip(colors,res_ot.keys()):

        med_ot= np.array(res_ot[bi][0])/190
        low_ot= np.array(res_ot[bi][1])/190
        high_ot= np.array(res_ot[bi][2])/190


        
        axes.set_xlabel("Exposure [kg-yr]")
        axes.set_ylabel("BI  [cts/keV/kg/yr]")

        axes.set_ylim(0,1.6*5/31/190)
        axes.set_xlim(0.1,500)

        axes.fill_between(exp,-np.array(low_ot)+np.array(med_ot),np.array(high_ot)+np.array(med_ot),  alpha=0.4,label="{:.3E}".format(bi),color=col,linewidth=0)
        axes.plot(exp,med_ot,color=col)
        plt.legend(loc="best",frameon=True,facecolor="white")
    plt.show()

