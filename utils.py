import glob
import time
import os
import awkward as ak
import numpy as np
from pathlib import Path
from lgdo import lh5
from legendmeta import LegendMetadata
from tqdm import tqdm
import json
import sys
import ROOT
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
from hist import Hist
import hist
from datetime import datetime
from scipy.stats import poisson

def get_run_times(metadb:LegendMetadata,analysis_runs:dict,verbose:bool=True)->dict:
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
    runinfo_p10 = json.load(open('/data1/users/calgaro/legend-metadata/dataprod/runinfo.json'))
    runinfo["p10"]={}
    for key,item in runinfo_p10["p10"].items():
        runinfo["p10"][key]=item

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

                if "phy" in runinfo[period][run].keys():
                    timestamp = datetime.strptime(runinfo[period][run]["phy"]["start_key"], '%Y%m%dT%H%M%SZ')
                    
                    ch = metadb.channelmap(metadb.dataprod.runinfo[period][run]["phy"]["start_key"])

                    geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and 
                                ch[_name]["analysis"]["usability"] in ["on","no_psd"]]

                    start_time = int(timestamp.timestamp())
                    if (first_time is None):
                        first_time=start_time
                    
                    start_time-=first_time
                    time = runinfo[period][run]["phy"]["livetime_in_s"]
                    end_time =start_time+time
                    mass=0
                    for det in geds_list:
                        mass += ch[det].production.mass_in_g/1000
                    output[period][run]=[start_time,end_time,mass]

    return output






def get_error_bar(N:float):
    """
    A poisson error-bar for N observed counts.
    """

    x= np.linspace(0,5+2*N,5000)
    y=poisson.pmf(N,x)
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

