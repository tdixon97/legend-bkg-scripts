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
    runinfo_p10 = json.load(open('/data1/users/calgaro/runinfo_new_p10.json'))
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
    print(low_quant,high_quant)
    return N-low_quant,high_quant-N