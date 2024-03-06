"""
time-analysis.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk) 
"""
import shutil
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
import subprocess
from collections import OrderedDict
import uproot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import hist
import argparse
from datetime import datetime, timezone

import os
import sys
import re
import utils
import json
from legendmeta import LegendMetadata

from matplotlib.backends.backend_pdf import PdfPages
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))


style = {
  "yerr":False,
    "flow": None,
    "lw": 0.6,
}

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


    output={}
    first_time =None
    ### loop over periods
    for period in metadb.dataprod.runinfo.keys():
        if (period in analysis_runs.keys()):
            output[period]={}
        livetime_tot =0

        ## loop over runs
        for run in metadb.dataprod.runinfo[period].keys():
            
            ## skip 'bad' runs
            if (period in analysis_runs.keys() and run in analysis_runs[period]):
                
                if "phy" in metadb.dataprod.runinfo[period][run].keys():
                    if (run=="r006" and period=="p06"):
                        continue
                    ch = metadb.channelmap(metadb.dataprod.runinfo[period][run]["phy"]["start_key"])

                    geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and 
                                ch[_name]["analysis"]["usability"] in ["on","no_psd"]]

                    timestamp = datetime.strptime(metadb.dataprod.runinfo[period][run]["phy"]["start_key"], '%Y%m%dT%H%M%SZ')
                
                    start_time = int(timestamp.timestamp())
                    if (first_time is None):
                        first_time=start_time
                    
                    start_time-=first_time
                    
                    time = metadb.dataprod.runinfo[period][run]["phy"]["livetime_in_s"]
                    end_time =start_time+time
                    mass=0
                    for det in geds_list:
                        mass += ch[det].production.mass_in_g/1000
                    output[period][run]=[start_time,end_time,mass]

                    
                    

    return output



### load the meta-data

metadb = LegendMetadata()
chmap = metadb.channelmap(datetime.now())
runs=metadb.dataprod.config.analysis_runs
run_times=get_run_times(metadb,runs,verbose=1)

### load the data
print(json.dumps(run_times,indent=1))

path="~/Downloads/l200a-p34678-dataset-v1.0.root"
low=4000
high=6000
out_name =f"test_{low}_{high}.root"

hists={}
periods=["p03","p04","p06","p07","p08"]
bins=[]
with uproot.open(path) as f:
    for period in periods:
        run_list=runs[period]
        hists[period]={}
        for run in run_list:
            if (run not in run_times[period]):
                continue
            tstart,tstop,mass = run_times[period][run]
            bins.append(tstart/60/60/24)
            bins.append(tstop/60/60/24)

            print(tstart,tstop,mass)
            if (f"mul_surv/{period}_{run};1" in f.keys()) and mass>0:
                hists[period][run]=utils.get_hist(f[f"mul_surv/{period}_{run}"],(0,6000),1)
counts={}

### get counts
for period in periods:
    run_list=runs[period]

    counts[period]={}
    for run in run_list:
        if run in hists[period].keys():
            counts[period][run]=utils.integrate_hist(hists[period][run],low,high)
print(json.dumps(counts,indent=1))
print(bins)
histo_time =( Hist.new.Variable(bins).Double())
histo_mass =( Hist.new.Variable(bins).Double())

### fill the mass histo
for period in periods:
    run_list=runs[period]
    hists[period]={}
    for run in run_list:
        if (run not in run_times[period]):
            continue
        tstart,tstop,mass = run_times[period][run]

        histo_mass[(tstart/60/60/24+tstop/60/60/24)*0.5j]=mass
        histo_time[(tstart/60/60/24+tstop/60/60/24)*0.5j]=counts[period][run]

with uproot.recreate(out_name) as output_file:
    output_file["counts"]=histo_time
    output_file["mass"]=histo_mass

histo_time=utils.normalise_histo(histo_time)
fig, axes_full = lps.subplots(1, 1, figsize=(6,4), sharex=True)

#histo_mass.plot(ax=axes_full,**style,color="black")
for i in range(histo_mass.size-2):
    if (histo_mass[i]>0):
        histo_time[i]/=histo_mass[i]
histo_time.plot(ax=axes_full,**style,color=vset.blue,histtype="fill",alpha=0.5)
axes_full.set_xlabel("Time [days]")
axes_full.set_ylabel("Counts / kg -day")

plt.show()

