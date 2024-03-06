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
from datetime import datetime

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
