from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')

import glob
import time
import os
import awkward as ak
import numpy as np
from pathlib import Path
from legendmeta import LegendMetadata
from tqdm import tqdm
import json
import copy
import sys
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
from hist import Hist
import hist
from datetime import datetime
from scipy.stats import poisson

def get_run_times(metadb:LegendMetadata,analysis_runs:dict,verbose:bool=True,ac=[],off=[])->dict:
    """ Get the livetime from the metadata
    Parameters:
        -metadb: the LegendMetadata object
        - analysis_runs (dict) a dictonary of the analysis runs
        -verbose: (bool), default True a bool to say if the livetime is printed to the screen
    Returns:
        -dict of the format
        period :
            {
                run: [start_time,stop_time,mass],
                ...
            }
    
    """
    runinfo = metadb.dataprod.runinfo
    output={}
    first_time =None
    ### loop over periods
    for period in runinfo.keys():
        if (period in analysis_runs.keys()):
            output[period]={}
        livetime_tot =0

        ## loop over runs
        for run in runinfo[period].keys():
            
            ## skip 'bad' runs
            if (period in analysis_runs.keys() and run in analysis_runs[period]):

                if "phy" in runinfo[period][run].keys() and "livetime_in_s" in runinfo[period][run]["phy"].keys():
                    timestamp = datetime.strptime(runinfo[period][run]["phy"]["start_key"], '%Y%m%dT%H%M%SZ')
                    
                    ch = metadb.channelmap(metadb.dataprod.runinfo[period][run]["phy"]["start_key"])

                    geds_list= [ (_name,_dict["daq"]["rawid"]) for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and 
                                ch[_name]["analysis"]["usability"] in ["on","no_psd"]]

                    start_time = int(timestamp.timestamp())
                    if (first_time is None):
                        first_time=start_time
                    
                    start_time-=first_time
                    time = runinfo[period][run]["phy"]["livetime_in_s"]
                    end_time =start_time+time
                    mass=0
                    for det in geds_list:
                   
                        if (det[1] not in ac) and (det[1] not in off):
                            mass += ch[det[0]].production.mass_in_g/1000
                    output[period][run]=[start_time,end_time,mass]

    return output




def get_smallest_ci(N,x,y):
    integral = y[np.argmax(y)]
    bin_id_l = np.argmax(y)
    bin_id_u = np.argmax(y)

    integral_tot = np.sum(y)
    while integral<0.683*integral_tot:

        ### get left bin
        if (bin_id_l>0 and bin_id_l<len(y)):
            c_l =y[bin_id_l-1]
        else:
            c_l =0

        if (bin_id_u>0 and bin_id_u<len(y)):
            c_u =y[bin_id_u+1]
        else:
            c_u =0
        
        if (c_l>c_u):
            integral+=c_l
            bin_id_l-=1
        else:
            integral+=c_u
            bin_id_u+=1
        
    low_quant = x[bin_id_l]
    high_quant=x[bin_id_u]
    return N-low_quant,high_quant-N

def get_error_bar(N:float):
    """
    A poisson error-bar for N observed counts.
    """

    x= np.linspace(0,5+2*N,5000)
    y=poisson.pmf(N,x)
    return get_smallest_ci(N,x,y)


def get_hist(obj,range:tuple=(132,4195),bins:int=10):
    """                                                                                                                                                                                                    
    Extract the histogram (hist package object) from the uproot histogram                                                                                                                                  
    Parameters:                                                                                                                                                                                            
        - obj: the uproot histogram                                                                                                                                                                        
        - range: (tuple): the range of bins to select (in keV)                                                                                                                                             
        - bins (int): the (constant) rebinning to apply                                                                                                                                                    
    Returns:                                                                                                                                                                                               
        - hist                                                                                                                                                                                             
    """
    return obj.to_hist()[range[0]:range[1]][hist.rebin(bins)]


def normalise_histo(hist,factor=1):
    """ Normalise a histogram into units of counts/keV"""

    widths= np.diff(hist.axes.edges[0])

    for i in range(hist.size-2):
        hist[i]/=widths[i]
        hist[i]*=factor
    return hist

def integrate_hist(hist,low,high):
    """ Integrate the histogram"""

    bin_centers= hist.axes.centers[0]

    values = hist.values()
    lower_index = np.searchsorted(bin_centers, low, side="right")
    upper_index = np.searchsorted(bin_centers, high, side="left")
    bin_contents_range =values[lower_index:upper_index]
    bin_centers_range=bin_centers[lower_index:upper_index]

    return np.sum(bin_contents_range)



def sideband_counting(hist,low,center_low,center_high,high,pdf=None,name=""):
    """
    A counting analysis using a bayesian approach,
    the data is modelled as a flat bkg and a central peak,
    the posterior is evaliuated on a grid then the marginalised posterior on
    the signal counts is computed.

    Parameters:
        hist: histogram object
        low, center_low,center_high,high: edges of the bins
    Returns:
        low,med,high
    """

    ## get the counts per bin
    N1 = integrate_hist(hist,low,center_low)
    N2 = integrate_hist(hist,center_low,center_high)
    N3 = integrate_hist(hist,center_high,high)

    max_S = 10+N2*2
    max_B = 10+N2*2
    ## get bin widths
    w1= center_low-low
    w2=center_high-center_low
    w3=high-center_high

    def likelihood(S,B):
        """
        Likelihood function based on "S" signal counts, "B" background
        """
        return poisson.pmf(N1,B*w1/w2)*poisson.pmf(N2,B+S)*poisson.pmf(N3,B*w3/w2)

    histo = ( Hist.new.Reg(200, 0, max_S).Reg(200,0,max_B).Double())

    w,x,y = histo.to_numpy()

    x_2d = np.tile(x, (len(y), 1))
    y_2d = np.tile(y, (len(x), 1)).T

    l = likelihood(x_2d,y_2d)
    maxi = np.max(l)

    best_fit = x_2d.flatten()[np.argmax(l.flatten())],y_2d.flatten()[np.argmax(l.flatten())]
    w_x = np.sum(l,axis=0)
    w_y=np.sum(l,axis=1)

    if (pdf is not None):
        style = {
            "yerr": False,
            "flow": None,
            "lw": 0.6,
            }
        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})

        
        histo_x = ( Hist.new.Reg(200, 0, max_S).Double())
        histo_y = ( Hist.new.Reg(200, 0, max_B).Double())

        for i in range(histo_x.size-2):
            histo_x[i]=w_x[i]
            histo_y[i]=w_y[i]

        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
        axes.set_xlabel("Counts")
        axes.set_ylabel("Prob [arb units]")
    
        histo_x.plot(ax=axes,**style,histtype="fill",alpha=0.5,label="Signal")
        histo_y.plot(ax=axes,**style,histtype="fill",alpha=0.5,label="Bakground")
        axes.set_title(name)
        axes.set_xlim(0,max_S)
        plt.legend(loc="best",frameon=True,facecolor="white")

        pdf.savefig()
        plt.close("all")

        hist_fit =copy.deepcopy(hist)
        bw =np.diff(hist.axes.centers[0])[0]
        maxi=0
        for i in range(hist.size-2):
            xt= hist.axes.centers[0][i]

            B1 = best_fit[1]*w1/w2
            B3 = best_fit[1]*w3/w2

            if (xt<low):
                hist_fit[i]=0
            elif(xt<center_low):
                hist_fit[i]=bw*B1/w1
            elif (xt<center_high):
                hist_fit[i]=bw*(best_fit[1]+best_fit[0])/w2
            elif (xt<high):
                hist_fit[i]=bw*B3/w3
            else:
                hist_fit[i]=0

            if (xt>low and xt<high and hist[i]>maxi):
                maxi=hist[i]

        fig, axes = lps.subplots(1, 1, figsize=(3,3), sharex=True, gridspec_kw = {'hspace': 0})
        hist.plot(ax=axes,**style,histtype="fill",alpha=0.5,label="Data")
        hist_fit.plot(ax=axes,**style,color="black",alpha=0.5,label="Fit")

        axes.set_title(name)
        axes.set_xlim(low-5,high+5)
        axes.set_ylim(0,maxi+5)
        axes.set_xlabel("Energy [keV]")
        axes.legend(loc="best")
        pdf.savefig()
        plt.close("all")
    
    return (x[np.argmax(w_x)],get_smallest_ci(x[np.argmax(w_x)],x,w_x)[0],get_smallest_ci(x[np.argmax(w_x)],x,w_x)[1]),histo_x



def sample_hist(hist,N):
    """
    Generate samples from a histogram
    """

    edges = hist.axes[0].edges
    int = sum(hist.values())

    counts = hist.view()

    bin_widths = np.diff(edges)
    probabilities = counts / np.sum(counts )

    sample_indices = np.random.choice(range(len(edges) - 1), size=N, p=probabilities)

    sampled_values = np.random.uniform(edges[sample_indices], edges[sample_indices + 1])
    
    return sampled_values