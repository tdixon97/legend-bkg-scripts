"""
time-analysis.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk) 
"""
from legend_plot_style import LEGENDPlotStyle as lps

import shutil
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
import copy
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

print(json.dumps(run_times,indent=1))
### load the data
print(json.dumps(run_times,indent=1))

path="outputs/l200a-p34678-dataset-v1.0.root"
path_p10 ="outputs/l200a-p10-r000-dataset-tmp-auto.root"
low=3000
high=6000
out_name =f"test_{low}_{high}.root"

hists={}
hists_p10={}

periods=["p03","p04","p06","p07","p08","p10"]
bins=[]
with uproot.open(path) as f:
    with uproot.open(path_p10) as f2:
        for period in periods:
            run_list=runs[period]
            hists[period]={}
            hists_p10[period]={}

            for run in run_list:
                if (run not in run_times[period]):
                    continue
                tstart,tstop,mass = run_times[period][run]
                bins.append(tstart/60/60/24)
                bins.append(tstop/60/60/24)

                print(tstart,tstop,mass)
                if (period!="p10" and f"mul_surv/{period}_{run};1" in f.keys()) and mass>0:
                    hists[period][run]=get_hist(f[f"mul_surv/{period}_{run}"],(0,6000),1)
                elif (period=="p10" and f"mul_surv/{period}_{run};1" in f2.keys()) and mass>0:
                    hists_p10[period][run]=get_hist(f2[f"mul_surv/{period}_{run}"],(0,6000),1)
                
            
counts={}
counts_p10={}
### get counts
for period in periods:
    run_list=runs[period]

    if (period!="p10"):
        counts[period]={}
    else:
        counts_p10[period]={}
    for run in run_list:
        if run in hists[period].keys():
            counts[period][run]=integrate_hist(hists[period][run],low,high)
        if run in hists_p10[period].keys():
            counts_p10[period][run]=integrate_hist(hists_p10[period][run],low,high)


histo_time =( Hist.new.Variable(bins).Double())
histo_mass =( Hist.new.Variable(bins).Double())
histo_time_p10 =( Hist.new.Variable(bins).Double())
histo_mass_p10 =( Hist.new.Variable(bins).Double())

### fill the mass histo
for period in periods:
    run_list=runs[period]
    for run in run_list:
        if (run not in run_times[period]):
            continue
        tstart,tstop,mass = run_times[period][run]

        if (period!="p10"):
            histo_mass[(tstart/60/60/24+tstop/60/60/24)*0.5j]=mass
            histo_time[(tstart/60/60/24+tstop/60/60/24)*0.5j]=counts[period][run]
        else:
            histo_mass_p10[(tstart/60/60/24+tstop/60/60/24)*0.5j]=mass
            histo_time_p10[(tstart/60/60/24+tstop/60/60/24)*0.5j]=counts_p10[period][run]

with uproot.recreate(out_name) as output_file:
    output_file["counts"]=histo_time
    output_file["mass"]=histo_mass
    output_file["counts_p10"]=histo_time_p10
    output_file["mass_p10"]=histo_mass_p10

plot_hist=False
if (plot_hist==True):
    histo_time_plot=normalise_histo(histo_time)
else:
    histo_time_plot=copy.deepcopy(histo_time)

if (plot_hist==True):
    histo_time_plot_p10=normalise_histo(histo_time_p10)
else:
    histo_time_plot_p10=copy.deepcopy(histo_time_p10)
fig, axes_full = lps.subplots(1, 1, figsize=(4,3), sharex=True)

#histo_mass.plot(ax=axes_full,**style,color="black")
x=[]
y=[]
ey_low=[]
ey_high=[]
x_p10=[]
y_p10=[]
ey_low_p10=[]
ey_high_p10=[]
widths= np.diff(histo_mass.axes.edges[0])
centers=histo_mass.axes.edges[0]

for i in range(histo_mass.size-2):
    if (histo_mass[i]>0 and widths[i]>1):
        
        histo_time_plot[i]/=histo_mass[i]

        norm=(widths[i]*histo_mass[i])
        x.append(centers[i])
        y.append(histo_time[i]/norm)
        print(histo_time[i])
        el,eh =utils.get_error_bar(histo_time[i])
        ey_low.append(el/norm)  
        ey_high.append(eh/norm)


for i in range(histo_mass_p10.size-2):
    if (histo_mass_p10[i]>0 and widths[i]>1):
        
        histo_time_plot_p10[i]/=histo_mass_p10[i]

        norm=(widths[i]*histo_mass_p10[i])
        x_p10.append(centers[i])
        y_p10.append(histo_time_p10[i]/norm)
        print(histo_time[i])
        el,eh =utils.get_error_bar(histo_time_p10[i])
        ey_low_p10.append(el/norm)  
        ey_high_p10.append(eh/norm)

print(histo_time_p10)
if (plot_hist):
    histo_time.plot(ax=axes_full,**style,color=vset.blue,histtype="fill",alpha=0.5,label="WIth OB")
    histo_time_p10.plot(ax=axes_full,**style,color=vset.orange,histtype="fill",alpha=0.5,label="NO OB")

else:
    axes_full.errorbar(x=x,y=y,yerr=[np.abs(ey_low),np.abs(ey_high)],color=vset.blue,fmt="o",ecolor="grey",label="With OB")
    axes_full.errorbar(x=x_p10,y=y_p10,yerr=[np.abs(ey_low_p10),np.abs(ey_high_p10)],color=vset.red,fmt="o",ecolor=vset.orange,label="No OB")

axes_full.set_xlabel("Time [days]")
axes_full.set_ylabel("Counts / kg -day")
axes_full.legend(loc="upper right")
plt.savefig(out_name[0:-5]+".pdf")
