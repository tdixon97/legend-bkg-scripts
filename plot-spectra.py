"""
plot-spectra.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk) 
"""
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
import shutil
import subprocess
from collections import OrderedDict
import uproot
from pathlib import Path

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
plt.rcParams.update({'font.size': 24}) 

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

parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
parser.add_argument("--output", "-o",type=str,help="Name of output pdf file",default="test")
parser.add_argument("--input", "-i",type=str,help="Name of input root file",default = "/data1/users/tdixon/build_pdf/outputs/l200a-p34678-dataset-v1.0.root")
parser.add_argument("--input_p10", "-I",type=str,help="Name of input root file for p10",default ="outputs/l200a-p10-r000-r001-dataset-tmp-auto.root")
parser.add_argument("--energy", "-e",type=str,help="Energy range to plot",default="0,4000")
parser.add_argument("--binning", "-b",type=int,help="Binning",default=5)
parser.add_argument("--spectrum","-s",type=str,help="Spectrum to plot",default="mul_surv")
parser.add_argument("--dataset","-d",type=str,help="Which group of detectors to plot",default="all")
parser.add_argument("--variable","-V",type=str,help="Variable binning, argument is the path to the cfg file defualt 'None' and flat binning is used",default=None)

args = parser.parse_args()

path =args.input_p10
path_all = args.input
output =args.output
binning=args.binning
spectrum =args.spectrum
energy=args.energy
dataset=args.dataset
variable = args.variable
energy_low = int(energy.split(",")[0])
energy_high = int(energy.split(",")[1])
os.makedirs("plots",exist_ok=True)


metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")
#usability_path = "/data1/users/tdixon/legend-bkg-scripts/cfg/usability_changes.json" #not used

chmap = metadb.channelmap(datetime.now())
runs=metadb.dataprod.config.analysis_runs
runs['p10']=['r000','r001','r002','r003']

run_times=utils.get_run_times(metadb,runs,verbose=1)

## get exposure

exp_nu=0
exp_10=0
for p, _dict in run_times.items():
    for r, times in _dict.items():
        
        if (p=="p10"):
            exp_10+=(times[1]-times[0])*times[2]/(60*60*24*365)
        elif (p in ["p03","p04","p05","p06","p07","p08"]):
            exp_nu+=(times[1]-times[0])*times[2]/(60*60*24*365)

print("Total exposure in p10:", exp_10, "kg-yr")
print("Total exposure in other periods:", exp_nu, "kg-yr")

## get binning edges for now the ones for ICPC, this can be also repalced with any other binning
edges =None
if (variable is not None):
    with open(variable, 'r') as json_file:
        edges=np.unique(utils.string_to_edges(json.load(json_file)["icpc"]))

with uproot.open(path_all) as f2:
    
    h1=utils.get_hist(f2[f"{spectrum}/{dataset}"],(energy_low,energy_high),binning,edges)

with uproot.open(path) as f2:
    
    h2=utils.get_hist(f2[f"{spectrum}/{dataset}"],(energy_low,energy_high),binning,edges)

for i in range(h1.size-2):
    h1[i]/=exp_nu
for i in range(h2.size-2):
    h2[i]/=exp_10

fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)

h2.plot(ax=axes_full, **style,color=vset.blue,label=f"p10 {runs['p10']}")
h1.plot(ax=axes_full,**style,color=vset.orange,label="p3-8")
axes_full.set_xlabel("Energy [keV]")
if (variable is None):
    axes_full.set_ylabel(f"counts/({binning} keV kg yr)")
else:
    axes_full.set_ylabel(f"counts/(keV kg yr)")

axes_full.set_yscale("linear")
axes_full.set_title(f"{spectrum} - {dataset}")
axes_full.set_xlim(energy_low,energy_high)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("plots/"+output+".pdf")

