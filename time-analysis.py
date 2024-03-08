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


### looad the arguments
parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
parser.add_argument("--output", "-o",type=str,help="Name of output root file, eg l200a-vancouver_full_alpha will be appended with energy range",default="l200a-vancouver_full_alpha_3000_6000_keV")
parser.add_argument("--input", "-i",type=str,help="Name of input root file",default = "outputs/l200a-vancouver_full-dataset-v0.0.0.root")
parser.add_argument("--input_p10", "-I",type=str,help="Name of input root file for p10",default =None)
parser.add_argument("--energy", "-e",type=str,help="comma seperate energy range",default="4000,6000")

args = parser.parse_args()
output =args.output
input = args.input
input_p10 =args.input_p10
include_p10=True

if (input_p10 is None):
    include_p10 =False
    
energy=args.energy

energy_low = float(energy.split(",")[0])
energy_high = float(energy.split(",")[1])


### load the meta-data
### ----------------------


metadb = LegendMetadata()

chmap = metadb.channelmap(datetime.now())
runs=metadb.dataprod.config.analysis_runs

## hardcoded for now
if (include_p10):
    runs["p10"]=["r001"]

run_times=utils.get_run_times(metadb,runs,verbose=1)

### load the data

out_name =f"{output}_{energy_low}_{energy_high}.root"

hists={}
hists_p10={}


### extract the data and start / stop times
### ----------------------------------------

periods=["p03","p04","p06","p07","p08"]
if (include_p10):
    periods.append("p10")

bins=[]
print(input)
with uproot.open(input) as f:
    
    for period in periods:
        run_list=runs[period]
        hists[period]={}

        for run in run_list:
            if (run not in run_times[period]):
                continue
            tstart,tstop,mass = run_times[period][run]
            bins.append(tstart/60/60/24)
            bins.append(tstop/60/60/24)

            if (period!="p10" and f"mul_surv/{period}_{run};1" in f.keys()) and mass>0:
                hists[period][run]=utils.get_hist(f[f"mul_surv/{period}_{run}"],(0,6000),1)


### now also p10   
if (include_p10):
    with uproot.open(input_p10) as f:
    
        for period in periods:
            hists_p10[period]={}

            for run in run_list:
                if (run not in run_times[period]):
                    continue

                if (period=="p10" and f"mul_surv/{period}_{run};1" in f.keys()) and mass>0:
                    hists_p10[period][run]=utils.get_hist(f[f"mul_surv/{period}_{run}"],(0,6000),1)
          


### get counts in each run
### -------------------------------------------------
                    
counts={}
counts_p10={}


for period in periods:
    run_list=runs[period]

    if (period!="p10"):
        counts[period]={}
    elif include_p10:
        counts_p10[period]={}

    for run in run_list:
        if run in hists[period].keys():
            counts[period][run]=utils.integrate_hist(hists[period][run],energy_low,energy_high)

        if include_p10:
            if run in hists_p10[period].keys():
                counts_p10[period][run]=utils.integrate_hist(hists_p10[period][run],energy_low,energy_high)


### fill the histograms
### -------------------------------------------------

histo_time =( Hist.new.Variable(bins).Double())
histo_mass =( Hist.new.Variable(bins).Double())

if (include_p10):
    histo_time_p10 =( Hist.new.Variable(bins).Double())
    histo_mass_p10 =( Hist.new.Variable(bins).Double())


for period in periods:
    run_list=runs[period]
    for run in run_list:
        if (run not in run_times[period]):
            continue
        tstart,tstop,mass = run_times[period][run]

        if (period!="p10"):
            histo_mass[(tstart/60/60/24+tstop/60/60/24)*0.5j]=mass
            histo_time[(tstart/60/60/24+tstop/60/60/24)*0.5j]=counts[period][run]
        elif (include_p10):
            histo_mass_p10[(tstart/60/60/24+tstop/60/60/24)*0.5j]=mass
            histo_time_p10[(tstart/60/60/24+tstop/60/60/24)*0.5j]=counts_p10[period][run]

with uproot.recreate(out_name) as output_file:
    output_file["counts"]=histo_time
    output_file["mass"]=histo_mass
    if (include_p10):
        output_file["counts_p10"]=histo_time_p10
        output_file["mass_p10"]=histo_mass_p10

plot_hist=True
if (plot_hist==True):
    histo_time_plot=utils.normalise_histo(histo_time)
else:
    histo_time_plot=copy.deepcopy(histo_time)

if (include_p10):
    if (plot_hist==True):
        histo_time_plot_p10=utils.normalise_histo(histo_time_p10)
    else:
        histo_time_plot_p10=copy.deepcopy(histo_time_p10)



### Normalise the histos and save graph (errorbar) for plotting
### ----------------------------------------------------------

x=[]
y=[]
ey_low=[]
ey_high=[]
if (include_p10):
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
        el,eh =utils.get_error_bar(histo_time[i])
        ey_low.append(el/norm)  
        ey_high.append(eh/norm)
    else:
        histo_time_plot[i]=0

if (include_p10):
    for i in range(histo_mass_p10.size-2):
        if (histo_mass_p10[i]>0 and widths[i]>1):
            
            histo_time_plot_p10[i]/=histo_mass_p10[i]

            norm=(widths[i]*histo_mass_p10[i])
            x_p10.append(centers[i])
            y_p10.append(histo_time_p10[i]/norm)
            el,eh =utils.get_error_bar(histo_time_p10[i])
            ey_low_p10.append(el/norm)  
            ey_high_p10.append(eh/norm)
        else:
            histo_time_plot_p10[i]=0
            

### Make plots
### --------------------------------------------------------

fig, axes_full = lps.subplots(1, 1, figsize=(4,3), sharex=True)

if (plot_hist):
    histo_time_plot.plot(ax=axes_full,**style,color=vset.blue,histtype="fill",alpha=0.5,label="WIth OB")
    if (include_p10):
        histo_time_plot_p10.plot(ax=axes_full,**style,color=vset.orange,histtype="fill",alpha=0.5,label="NO OB")

else:
    axes_full.errorbar(x=x,y=y,yerr=[np.abs(ey_low),np.abs(ey_high)],color=vset.blue,fmt="o",ecolor="grey",label="With OB")
    if (include_p10):
        axes_full.errorbar(x=x_p10,y=y_p10,yerr=[np.abs(ey_low_p10),np.abs(ey_high_p10)],color=vset.red,fmt="o",ecolor=vset.orange,label="No OB")

axes_full.set_xlabel("Time [days]")
axes_full.set_ylabel("Counts / kg -day")
if (include_p10):
    axes_full.legend(loc="upper right")
plt.savefig(out_name[0:-5]+".pdf")
