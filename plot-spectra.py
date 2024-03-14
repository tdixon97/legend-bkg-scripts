"""
plot-spectra.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk) 
"""
import shutil
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend')
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
metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")
usability_path = "/data1/users/tdixon/legend-bkg-scripts/cfg/usability_changes.json"

chmap = metadb.channelmap(datetime.now())
runs=metadb.dataprod.config.analysis_runs
#runs['p10']=['r000']
runs['p10']=['r000','r001']
with Path(usability_path).open() as f:
    usability = json.load(f)
off_list = usability["ac_to_off"]
ac_list=usability["ac"]
run_times=utils.get_run_times(metadb,runs,verbose=1,ac=ac_list,off=off_list)
print(json.dumps(run_times,indent=1))

## get exposure

exp_nu=0
exp_10=0
for p, _dict in run_times.items():
    for r, times in _dict.items():

        if (p=="p10"):
            exp_10+=(times[1]-times[0])*times[2]/(60*60*24*365)
        elif (p in ["p03","p04","p05","p06","p07","p08"]):
            exp_nu+=(times[1]-times[0])*times[2]/(60*60*24*365)

print(exp_nu)
print(exp_10)
path = "../build_pdf/outputs/l200a-vancouver23-dataset-v1.0.root"

path_all = "outputs/l200a-p10-r000-dataset-tmp-auto.root"
with uproot.open(path_all) as f2:
    
    h2=utils.get_hist(f2["mul_surv/all"],(0,3000),1)

with uproot.open(path) as f2:
    
    h1=utils.get_hist(f2["mul_surv/all"],(0,3000),1)

for i in range(h1.size-2):
    h1[i]*=exp_10/exp_nu
fig, axes_full = lps.subplots(1, 1, figsize=(6,4), sharex=True)

h2.plot(ax=axes_full, **style,color=vset.blue,label="p10 r000")
h1.plot(ax=axes_full,**style,color=vset.orange,label="p3-8")
axes_full.set_xlabel("Energy [keV]")
axes_full.set_ylabel("counts/5 keV")
axes_full.set_yscale("linear")
axes_full.set_xlim(2600,2630)
axes_full.set_ylim(0,10)

plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("test.pdf")

