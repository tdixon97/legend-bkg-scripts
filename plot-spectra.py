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
