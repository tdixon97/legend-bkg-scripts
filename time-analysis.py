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
import utils
import os
import sys
import re
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




### load the meta-data

metadb = LegendMetadata()
chmap = metadb.channelmap(datetime.now())
runs=metadb.dataprod.config.analysis_runs
runs['p10']= ['r000']
run_times=utils.get_run_times(metadb,runs,verbose=1)

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
                hists[period][run]=get_hist(f[f"mul_surv/{period}_{run}"],(0,6000),1)
counts={}

### get counts
for period in periods:
    run_list=runs[period]

    counts[period]={}
    for run in run_list:
        if run in hists[period].keys():
            counts[period][run]=integrate_hist(hists[period][run],low,high)
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

histo_time=normalise_histo(histo_time)
fig, axes_full = lps.subplots(1, 1, figsize=(6,4), sharex=True)

#histo_mass.plot(ax=axes_full,**style,color="black")
for i in range(histo_mass.size-2):
    if (histo_mass[i]>0):
        histo_time[i]/=histo_mass[i]
histo_time.plot(ax=axes_full,**style,color=vset.blue,histtype="fill",alpha=0.5)
axes_full.set_xlabel("Time [days]")
axes_full.set_ylabel("Counts / kg -day")

plt.show()

