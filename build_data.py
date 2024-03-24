"""
Script to load the data for the LEGEND-200 background model
Main Authors: Sofia Calgaro, Toby Dixon, Luigi Pertoldi based on a script from William Quinn
"""
import glob
import time
import os
import awkward as ak
import numpy as np
from pathlib import Path
import argparse
from lgdo import lh5
from legendmeta import LegendMetadata
from tqdm import tqdm
import json
import sys
import ROOT
import uproot
import logging
from datetime import datetime

# -----------------------------------------------------------
# LOGGER SETTINGS 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
stream_handler.setFormatter(formatter)
# -----------------------------------------------------------

def get_vectorised_converter(mapping):
    def channel2other(channel):
        """ Extract which string a given channel is in"""

        return mapping[f"ch{channel}"]

    return np.vectorize(channel2other)

def get_string_row_diff(channel_array,channel2string,channel2position):
    """
    Get the categories for the m2 data based on 3 categories (should be in the cfg)
    1) Same string vertical neighbour
    2) Same string not vertical neighbor
    3) Different string
    Parameters:
        channel_array: 2D numpy array of channels
        channel2string: vectorised numpy function to convert channel into string
        chnanel2position: vectorised numpy function to convert channel into position
    Returns:
        categories: list of categories per event
    """
    
    channel_array=np.vstack(channel_array)
    channel_one = channel_array[:,0].T
    channel_two=channel_array[:,1].T
    
    
    ## convert to the list of strings
    string_one=channel2string(channel_one)
    string_two=channel2string(channel_two)
    string_diff_1= (string_one-string_two)%11
    string_diff_2= (-string_one+string_two)%11
    string_diff = np.array([min(a, b) for a, b in zip(string_diff_1, string_diff_2)])
   
    position_one=channel2position(channel_one)
    position_two=channel2position(channel_two)

    floor_diff = np.abs(position_one-position_two)

    return np.array(string_diff),np.array(floor_diff)

def get_m2_categories(channel_array,channel2string,channel2position):
    """
    Get the categories for the m2 data based on 3 categories (should be in the cfg)
    1) Same string vertical neighbour
    2) Same string not vertical neighbor
    3) Different string
    Parameters:
        channel_array: 2D numpy array of channels
        channel2string: vectorised numpy function to convert channel into string
        chnanel2position: vectorised numpy function to convert channel into position
    Returns:
        categories: list of categories per event
    """
    
    channel_array=np.vstack(channel_array)
    channel_one = channel_array[:,0].T
    channel_two=channel_array[:,1].T
    
    
    ## convert to the list of strings
    string_one=channel2string(channel_one)
    string_two=channel2string(channel_two)
    

 
    
    same_string=string_one==string_two
    position_one=channel2position(channel_one)
    position_two=channel2position(channel_two)
    neighbour = np.abs(position_one-position_two)==1

    is_cat_one = (same_string) & (neighbour)
    is_cat_two = (same_string) &(~neighbour)
    is_cat_three = (~same_string)
    category = 1*is_cat_one+2*is_cat_two+3*is_cat_three
    return np.array(category)

def cross_talk_corrected_energies(energies:np.ndarray,channels:np.ndarray,cross_talk_matrix)->tuple[np.ndarray,bool]:
    """
    Compute the cross talk corrected energies, if we know that the cross talk induced on channel A by channel B is X%
    and the cross talk induced on channel B by channel A is Y%
    Therefore the energy we will measure on channel A is the sum of the deposited energy and the cross talk
    Emeas,A = Edep + EdepB*X/100
    and Edep = Emeas,A - EdepB*X/100
    Since EdepB ~EmeasB to a good approximation
    Edep,A =  Emeas,A - EmeasB*X/100 and
    Edep,B = Emeas,B - EmeasA*Y/100
    
    This generalises for M>2 to become:
    E = E,A - sum_{i=other channels}E_i*C_{i,A}
    where C_AB = cross talk from A on channel B

    In this implementation the cross talk matrix is stored in a JSON file keyed by channel therefore  one can acces
    C_AB = cross talk from A on channel B

    cAB = matrix[channelA][channelB]

    We could probably make a fast implementation using some matrix multipliciation but we should be careful, since the matrixs will be very spare.
    In practice almost all our data is M=1,2 or 3
    
    Parameters:
        - energies: a numpy array of the energies
        - channels: a numpy array of the channel names (strings)
    Returns:
        - a numpy array of the corrected energies
        - a bool to say if its valid
    """

    ## first some checks
    if (len(energies)!=len(channels)):
        raise ValueError("Error the energies and channel vectors must be the same length")

    ## for M = 1 no correction is needed
    if (len(energies)==1):
        return energies,False
    elif (len(energies)==2):

        ## check the keys
        if (channels[1] in cross_talk_matrix and channels[0] in cross_talk_matrix[channels[1]]):
            cross_talk_factors=np.array([cross_talk_matrix[channels[1]][channels[0]],
                                cross_talk_matrix[channels[0]][channels[1]]
                                ])
        else:
            if (channels[0]==channels[1]):
                return np.zeros(2),True
            #logger.info("We have pairs {}, {} in the data and not in the matrix".format(channels[0],channels[1]))
            cross_talk_factors=np.zeros(2)

        energies_flip = np.array([energies[1],energies[0]])

        return energies-energies_flip*cross_talk_factors,True

    else:
        raise ValueError("Error: In the current skm files we should never have >2 energies")

def get_data_awkard(cfg:dict,periods=None,target_key=None,Nmax:int=None,run_list:dict={},bad_keys=[],metadb=LegendMetadata()):
    """
    A function to load the evt tier data into an awkard array also getting the energies
    Parameters
    ----------------------
        - cfg: path to the data for each period
        - periods: list of periods to use
        - Nmax (int): used for debugging, if not None (the default) only Nmax files will be read
        - run_list (dict): dictonary of run_lists
    Returns

    ----------------------
        - an awkard array of the data
    
    Example
    ----------------------
        
    """

    data=None

    ## loop over period and run (so we can save this)
    N=0
    logger.info(f"This is the list of available periods/runs: {json.dumps(run_list,indent=1)}")
    logger.info("Starting to load evt tier for...")

    for period,run_l in tqdm(run_list.items()): 
        
        if (periods is not None and period in periods):
            logger.info(f"...{period}")

            for run in tqdm(run_l):
                logger.info(f"...... {run}")
                tier =cfg[period]["tier"]
                evt_path  =cfg[period]["evt_path"]
                tier = 'evt' if 'tmp-auto' in evt_path else 'pet'
                 
                if tier == 'evt':

                    fl_evt = glob.glob(evt_path+"/"+tier+"/phy/{}/{}/*-tier_evt.lh5".format(period,run))
                    print(evt_path+"/"+tier+"/phy/{}/{}/*-tier_evt.lh5".format(period,run))
                    ### remove the bad keys
                    fl_evt_new=[]
                    for f in fl_evt:
                        if not any(key in f for key in bad_keys):
                            fl_evt_new.append(f) 

                    fl_evt = fl_evt_new

                else:
                    fl_evt = glob.glob(evt_path+"/"+tier+"/phy/{}/{}/*-tier_pet.lh5".format(period,run))
                    for f in fl_evt:
                        if any(key in f for key in bad_keys):
                            raise ValueError(f"Error the key {f} is present in the data but shouldnt be" )

                # filter out files based on target key (if present)
                if target_key is not None:
                    len_before = len(fl_evt)
                    fl_evt = [f for f in fl_evt if datetime.strptime(f.split('-')[-2], "%Y%m%dT%H%M%SZ") <=  datetime.strptime(target_key, "%Y%m%dT%H%M%SZ")]
                    len_after = len(fl_evt)
                    if len_before > len_after:
                        print("you removed", len_after-len_before, "files in", period, run)

                ## loop
                for f_evt in fl_evt:
                    
                    f_tcm = f_evt.replace(tier, "tcm")
                    f_hit = f_evt.replace(tier, "hit" if 'tmp-auto' in evt_path else 'pht')

                    d_evt = lh5.read_as("evt", f_evt, library="ak")
                    if ('is_unphysical_idx_old' in d_evt["geds"].fields): 
       
                        d_evt["geds","is_is_unphysical_idx_new"]=d_evt["geds","is_unphysical_idx"]
                        d_evt["geds","is_unphysical_idx"]=d_evt["geds","is_unphysical_idx_old"]

                    # some black magic to get TCM data corresponding to geds.hit_idx
                    tcm = ak.Array(
                        {
                            k: ak.unflatten(
                                lh5.read_as(f"hardware_tcm_1/array_{k}", f_tcm, library="ak")[
                                    ak.flatten(d_evt.geds.hit_idx_all)
                                ],
                                ak.num(d_evt.geds.hit_idx_all),
                            )
                            for k in ["id", "idx"] 
                        }
                    )
                    tcm_unphysical =ak.Array(
                        {
                            k: ak.unflatten(
                                lh5.read_as(f"hardware_tcm_1/array_{k}", f_tcm, library="ak")[
                                    ak.flatten(d_evt.geds.is_unphysical_idx)
                                ],
                                ak.num(d_evt.geds.is_unphysical_idx),
                            )
                            for k in ["id", "idx"] 
                        }
                    )
                    # get uniques rawids for loading hit data
                    rawids = np.unique(ak.to_numpy(ak.ravel(tcm.id)))
                    energy = None
                    rawid_sort=None
                    physical =None

                    # for each table in the hit file
                    for rawid in rawids:

                        # get the hit table indices we need to load
                        idx_mask = tcm.idx[tcm.id == rawid]
                        idx_loc = ak.count(idx_mask, axis=-1)

                        # read in energy data with mask above
                        energy_data = lh5.read_as(
                            f"ch{rawid}/hit/cuspEmax_ctc_cal",
                            f_hit,
                            library="ak",
                            idx=ak.to_numpy(ak.flatten(idx_mask)),
                        )

                        # now bring back to original shape
                        data_unf = ak.unflatten(energy_data, idx_loc)
                        rawid_unf=ak.full_like(data_unf,rawid,dtype="int")
                        energy = (
                            data_unf if energy is None else ak.concatenate((energy, data_unf), axis=-1)
                        )
                        rawid_sort= (rawid_unf if rawid_sort is None else ak.concatenate((rawid_sort,rawid_unf),axis=-1))

                    d_evt["geds", "unphysical_hit_rawid"] = tcm_unphysical.id
                    d_evt["geds", "hit_rawid"] = rawid_sort
                    d_evt["geds", "energy"] = energy
                 
                    if run in metadb.dataprod.runinfo[period].keys():
                        ch = metadb.channelmap(metadb.dataprod.runinfo[period][run]["phy"]["start_key"])
                    else:
                        start = json.load(open('/data1/users/calgaro/legend-metadata/dataprod/runinfo.json'))[period][run]["phy"]["start_key"]
                        ch = metadb.channelmap(start)
                    
                    ac = [ _dict["daq"]["rawid"] for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["analysis"]["usability"] in ["ac"]]
                    off= [ _dict["daq"]["rawid"] for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["analysis"]["usability"] in ["off"]]

                    d_evt["geds","is_good_channel"]=d_evt["geds","hit_rawid"]>0
                    for a in ac:
                        d_evt["geds","is_good_channel"]=d_evt["geds","is_good_channel"]&(d_evt["geds","hit_rawid"]!=a)
                    for a in off:
                        d_evt["geds","is_good_channel"]=d_evt["geds","is_good_channel"]&(d_evt["geds","hit_rawid"]!=a)
             
                    d_evt["period"]=period
                    d_evt["run"]=run
                  
                    data = d_evt if data is None else ak.concatenate((data, d_evt))

                    tf=time.time()
            
                    N+=1

                    ### for debug
                    if ( Nmax is not None and N>Nmax):
                        break
                if ( Nmax is not None and N>Nmax ):
                    break
            if ( Nmax is not None and N>Nmax):
                break

    # QCs naming convention was switched starting from p10-r001 (included)
    #   - prior: OLD QC -> is_good_hit_old 
    #            NEW QC -> is_good_hit
    #   - after: OLD QC -> is_good_hit 
    #            NEW QC -> is_good_hit_new

    if ('is_good_hit_old' in data["geds"].fields): 
        data["geds","is_good_hit_new"]=data["geds","is_good_hit"]
        data["geds","is_good_hit"]=data["geds","is_good_hit_old"]
        print(data["geds"].fields)

    return data

def filter_off_ac(data,qcs_flag="is_good_hit",ac_dets=[],off_dets=[],verbose=False,recompute_qc_flag=True):
    """
    Function to remove the hits in off detectors and recompute the QC flag.
    Also sets some detectors to AC mode:
    Parameters:
        -data : awkward array of the data
        - qc (str): the quality cut flag
        - ac_dets (list): list of detector rawid's to set to AC
        - off_dets (list): list of detector rawids to set off
        -verbose (bool)
    Returns:
        - a new awkward array

    Example of filtering an off detector:
    
 
    
    }

    """
    data["geds","on_multiplicity"]=ak.sum(data.geds[qcs_flag], axis=-1)

    if (verbose):
        print("Data")
        print_events(data,qcs_flag)
    
    
 
    ### REMOVE hits in off dets in energy, qc_flag and rawid
    ### ----------------------------------------------------
    is_off =data.geds.hit_rawid>0
    for off_det in off_dets:
        is_off = is_off &(data.geds.hit_rawid!=off_det)

    ### we also need to update the QC flag
    filtered_energy = data["geds"]['energy'][ is_off]
    filtered_is_good_hit = data["geds"][qcs_flag][is_off]
    filtered_is_good_channel = data["geds"]["is_good_channel"][is_off]

    filtered_hit_rawid =data["geds"]['hit_rawid'][is_off]

    data["geds","energy"] = filtered_energy
    data["geds",qcs_flag] = filtered_is_good_hit
    data["geds","hit_rawid"] = filtered_hit_rawid
    data["geds","is_good_channel"] = filtered_is_good_channel

    data["geds","multiplicity"]=ak.num(data.geds[qcs_flag], axis=-1)
    data["geds","on_multiplicity"]=ak.sum(data.geds[qcs_flag], axis=-1)

    if (verbose):
        print("After removing OFF")
        print_events(data,qcs_flag)

    ### Recompute is_physical
    ### -----------------------------------------------------
    
    if (recompute_qc_flag==True):

        unphysical_channels =data["geds","unphysical_hit_rawid"]
     
        for c in off_dets:
            unphysical_channels = unphysical_channels[unphysical_channels!=c]

        num_bad_channels = ak.num(unphysical_channels,axis=-1)
        data["geds","unphysical_hit_rawid"]=unphysical_channels
        is_physical_recomputed = (num_bad_channels==0)
        

        data["geds",qcs_flag]=is_physical_recomputed & data["geds","is_good_channel"]
    
    
    data["geds","multiplicity"]=ak.num(data.geds[qcs_flag], axis=-1)
    data["geds","on_multiplicity"]=ak.sum(data.geds[qcs_flag], axis=-1)

    if (verbose):
        print("After fixing QC flag")
        print_events(data,qcs_flag)


    ### Modify the QC flag so AC dets have is_good_hit false
    ### ---------------------------------------------------

    cut=data.geds.hit_rawid>0
    for ac in ac_dets:
        cut=cut & (data.geds.hit_rawid!=ac)

    
    filtered_is_good_hit = data["geds"][qcs_flag] & (cut)
    filtered_is_good_channel = data["geds"]["is_good_channel"] & (cut)

    
    data["geds",qcs_flag] = filtered_is_good_hit
    data["geds","is_good_channel"] = filtered_is_good_channel

    data["geds","multiplicity"]=ak.num(data.geds[qcs_flag], axis=-1)
    data["geds","on_multiplicity"]=ak.sum(data.geds[qcs_flag], axis=-1)
    if (verbose):
        print("After changing ACs")
        print_events(data,qcs_flag)

    ## recompute the multiplicity
    data["geds","multiplicity"]=ak.num(data.geds[qcs_flag], axis=-1)
    data["geds","on_multiplicity"]=ak.sum(data.geds[qcs_flag], axis=-1)

    return data


def print_events(data,qc):
    """ Print of the events """
    vars = ["energy","hit_rawid","is_good_channel","unphysical_hit_rawid",qc,"multiplicity","on_multiplicity"]
    for v in vars:
        p = v+"                 "
        for i in range(min(len(data),5)):
            p+=str(data["geds"][v][i])+" "
        print(p)
  
    print("\n\n")

def main():
    parser = argparse.ArgumentParser(description="Script to load the data for the LEGEND-200 background model")
    parser.add_argument("--output", help="Name of output root file, eg l200a-p10-r000-dataset-tmp-auto")
    parser.add_argument("--p", help="List of periods to inspect")
    parser.add_argument("--proc", help="Boolean flag: True if you want to load from the pet/evt tier; if False and the parquet already exists, then we directly load data from this")
    parser.add_argument("--qc", default="old", help="Set to 'new' if you want to apply new cuts (post Vancouver CM), otherwise default value is set to 'old' cuts")
    parser.add_argument("--recompute", default=False, help="Boolean flag set to True if you want to recompute the QC flag after setting OFF detectors")
    parser.add_argument("--target", default=None, help="Target cycle up to which include data; use format '20240317T141137Z.")

    args = parser.parse_args()
    usability_path = "cfg/usability_changes.json"
    config_path = "/data1/users/tdixon/build_pdf/legend-simflow-config/tier/pdf/l200a/build-pdf-config.json"
    xtc_folder = "/data1/users/tdixon/build_pdf/cross_talk"
    bad_keys_path = "/data1/users/calgaro/legend-dataflow-config/ignore_keys.keylist"
    recompute_qc_flag = args.recompute
    target_key = args.target

    ## read bad-keys
    bad_list=[]
    with open(bad_keys_path, 'r') as file:
        for line in file:   
            bad_list.append(line.split(" ")[0])

    out_name = f"{args.output}.root"
    periods = args.p
    process_evt = False if args.proc=="False" else True 
    print(periods)
    paths_cfg={"p03":
                    {
                        "tier":"pet",
                        "evt_path":"/data2/public/prodenv/prod-blind/ref-v1.0.0/generated/tier/",
                    },
                "p04":
                    {
                        "tier":"pet",
                        "evt_path":"/data2/public/prodenv/prod-blind/ref-v1.0.0/generated/tier/",
                    },
                "p06":
                    {
                        "tier":"pet",
                        "evt_path":"/data2/public/prodenv/prod-blind/ref-v1.0.0/generated/tier/",
                    },
                "p07":
                    {
                        "tier":"pet",
                        "evt_path":"/data2/public/prodenv/prod-blind/ref-v1.0.0/generated/tier/"
                    },
                "p08":
                    {
                        "tier":"pet",
                        "evt_path":"/data2/public/prodenv/prod-blind/ref-v1.0.0/generated/tier/",
                    },
                "p10":
                    {
                        "tier":"evt",
                        "evt_path":"/data2/public/prodenv/prod-blind/tmp-auto/generated/tier/",
                    }
        }

    cross_talk_matrixs={
        'p03': os.path.join(xtc_folder, 'l200-p03-r000-x-talk-matrix.json'),
        'p04': os.path.join(xtc_folder, 'l200-p03-r000-x-talk-matrix.json'),
        'p06': os.path.join(xtc_folder, 'l200-p06-r000-x-talk-matrix.json'),
        'p07': os.path.join(xtc_folder, 'l200-p06-r000-x-talk-matrix.json'),
        'p08': os.path.join(xtc_folder, 'l200-p06-r000-x-talk-matrix.json'),
        'p09': os.path.join(xtc_folder, 'l200-p06-r000-x-talk-matrix.json'),
        'p10': os.path.join(xtc_folder, 'l200-p06-r000-x-talk-matrix.json')
    }

    with Path(config_path).open() as f:
        rconfig = json.load(f)

    with Path(usability_path).open() as f:
        usability = json.load(f)
    #### get the metadata information / mapping
    #### ---------------------------------------------
    logger.info(f"... get the metadata information / mapping")
    metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")
    chmap = metadb.channelmap(rconfig["timestamp"])

    geds_mapping = {
        f"ch{_dict['daq']['rawid']}": _name
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    geds_strings= {
        f"ch{_dict['daq']['rawid']}": _dict["location"]["string"]
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    geds_positions= {
        f"ch{_dict['daq']['rawid']}": _dict["location"]["position"]
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    strings = np.sort([item[1] for item in geds_strings.items()])
    channel2string = get_vectorised_converter(geds_strings)
    channel2position = get_vectorised_converter(geds_positions)

    # analysis runs
    runs=metadb.dataprod.config.analysis_runs
    runs['p10']=['r004']

    os.makedirs("outputs",exist_ok=True)
    output_cache = f"outputs/{out_name.replace('.root', '.parquet')}"
    print(out_name)
    if os.path.exists(output_cache) and process_evt==False:
        logger.info("Get from parquet")
        data =ak.from_parquet(output_cache)
    else:
        data=get_data_awkard(cfg=paths_cfg,periods=periods,target_key=target_key,Nmax=None,run_list=runs,bad_keys=bad_list,metadb=metadb)
        ak.to_parquet(data,output_cache)

    print("Got data")
    print(recompute_qc_flag)
    ### Add cuts on bad channels
    ### ---------------------------------------------------------------------
    ### For ON->AC this is just another cut
    ### FOR ON/AC -> OFF we need to modify the geds.energy geds.hit_rawid and geds.is_good_hit,
    ### to remove this hits, we then need to modify mulitplicity
    qcs_flag = "is_good_hit" if args.qc=="old" else "is_good_hit_new"

    data=filter_off_ac(data,qcs_flag=qcs_flag,off_dets=usability["ac_to_off"],ac_dets=usability["ac"])


    ### and the usual cuts
    data = data[ (~data.trigger.is_forced)    # no forced triggers
                    & (~data.coincident.puls) # no pulser eventsdata
                    & (~data.coincident.muon) # no muons
                    & (data.geds.multiplicity > 0) # no purely lar triggered events
                   &(ak.all(data.geds[qcs_flag],axis=-1))
                    ]
    
    


    if (np.all(data["geds","multiplicity"]==ak.num(data["geds","energy"]))==False):
        raise ValueError("Error the multiplicity must be equal to the length of the energy array")
    
    if (np.all(data["geds","multiplicity"]==ak.num(data["geds","hit_rawid"]))==False):
        raise ValueError("Error the multiplicity must be equal to the length of the is good hit array")
    
    if (np.all(data["geds","multiplicity"]==ak.num(data["geds",qcs_flag]))==False):
        raise ValueError("Error the multiplicity must be equal to the length of the is good hit array")

   
    #### Now create the histograms to fill
    #### --------------------------------------------------------------------
    logger.info(f"... create the histos to fill (full period)")

    hists = {}
    for _cut_name in rconfig["cuts"]:
        if not rconfig["cuts"][_cut_name]["is_sum"]:
            hists[_cut_name] = {}
            for _rawid, _name in sorted(geds_mapping.items()):
                hist_name = f"{_cut_name}_{_rawid}"
                hist_title = f"{_name} energy deposits"
                nbins = rconfig["hist"]["nbins"]
                emin = rconfig["hist"]["emin"]
                emax = rconfig["hist"]["emax"]
                hists[_cut_name][_rawid] = ROOT.TH1F(hist_name, hist_title, nbins, emin, emax)

    ## save also hists per run (total only now)
    logger.info(f"... create the histos to fill (run by run)")
    run_hists={}
    for _cut_name in rconfig["cuts"]:
        if not rconfig["cuts"][_cut_name]["is_sum"]:
            run_hists[_cut_name] = {}
            for _period, _run_list in (runs.items()):
                
                for run in _run_list:
                    hist_name = f"{_cut_name}_{_period}_{run}"
                    hist_title = f"{_period} {run} energy deposits"
                    nbins = rconfig["hist"]["nbins"]
                    emin = rconfig["hist"]["emin"]
                    emax = rconfig["hist"]["emax"]
                    run_hists[_cut_name][f"{_period}_{run}"] = ROOT.TH1F(hist_name, hist_title, nbins, emin, emax)

    sum_hists = {}
    string_diff=np.arange(7)
    names_m2 =[f"sd_{item1}" for item1 in string_diff]
    names_m2.extend(["all","cat_1","cat_2","cat_3"])

    ## now the summed histos
    for _cut_name in rconfig["cuts"]:
        cut_info = rconfig["cuts"][_cut_name]
        
        if cut_info["is_sum"] == False:
            continue
        if "th2" not in cut_info:
            hist_type = ROOT.TH1F
        elif ("th2" in cut_info):
            hist_type = ROOT.TH2F
        
        hist_title = "summed energy deposits"

        if hist_type == ROOT.TH1F:
            sum_hists[_cut_name]={}
           

            
            for name in names_m2:
                hist_name = f"{_cut_name}_{name}_summed"
                sum_hists[_cut_name][name] = hist_type(
                    hist_name,
                    hist_title,
                    rconfig["hist"]["nbins"],
                    rconfig["hist"]["emin"],
                    rconfig["hist"]["emax"],
                )
        elif hist_type==ROOT.TH2F:

            sum_hists[_cut_name]={}


            for name in names_m2:
                hist_name = f"{_cut_name}_{name}_summed"
                sum_hists[_cut_name][name] = hist_type(
                    hist_name,
                    hist_title,
                    rconfig["hist"]["nbins"],
                    rconfig["hist"]["emin"],
                    rconfig["hist"]["emax"],
                    rconfig["hist"]["nbins"],
                    rconfig["hist"]["emin"],
                    rconfig["hist"]["emax"]
                )
    

    #### Now start filling histograms
    #### -----------------------------------------------------
    logger.info(f"... fill histos")
                
    globs = {"ak": ak, "np": np}

    conversions={
            "mul ":"geds.multiplicity ",
            "mul_is_good ":  "geds.on_multiplicity ",
            "and":"&",
            "npe_tot":"spms.energy_sum"
    }

    for _period, _run_list in tqdm(runs.items(), desc='Processing'):

        if (_period not in periods):
            continue
        matrix_path =cross_talk_matrixs[_period]
        with open(matrix_path, 'r') as file:
            cross_talk_matrix = json.load(file)

        for _cut_name, _cut_dict in rconfig["cuts"].items():
            _cut_string = _cut_dict["cut_string"]
            
            for c_mc,c_data in conversions.items():
            
                _cut_string=_cut_string.replace(c_mc,c_data)

        
            ## not a summed spectra
            if _cut_dict["is_sum"] == False:
                ### loop over channels
                for _channel_id,_name in sorted(geds_mapping.items()):
                    _energy_array = (
                            data[eval(_cut_string,globs,data) & 
                                (data["period"]==_period) &
                                (data.geds.hit_rawid[:,0]==int(_channel_id[2:]))
                                ]["geds"]["energy"].to_numpy(
                            )
                        )  
                
                

                    if len(_energy_array) == 0: 
                        continue
                    hists[_cut_name][f"{_channel_id}"].FillN(
                            len(_energy_array), _energy_array, np.ones(len(_energy_array))
                        )
                
                _energy_array = (
                            data[eval(_cut_string,globs,data) & 
                                (data["period"]==_period) 
                                ]["geds"]["energy"].to_numpy(
                            )
                        )  
                _run_array = (
                            data[eval(_cut_string,globs,data) & 
                                (data["period"]==_period) 
                                ]["run"].to_numpy(
                            )
                        )  
                
                ### fill also time dependent hists

                for run in _run_list:
                    _energy_array_tmp = np.array(_energy_array)[_run_array==run]

                    if len(_energy_array_tmp) == 0: 
                        continue
                    run_hists[_cut_name][f"{_period}_{run}"].FillN(
                            len(_energy_array_tmp), _energy_array_tmp, np.ones(len(_energy_array_tmp))
                        )
                
                        
            elif ("cross_talk_correct" not in _cut_dict or _cut_dict["cross_talk_correct"]==False):
                _summed_energy_array = (
                    data[eval(_cut_string,globs,data)]["geds"]["energy_sum"].to_numpy()
                )  

                if len(_summed_energy_array) == 0: 
                    continue
                
                sum_hists[_cut_name]["all"].FillN(
                    len(_summed_energy_array),
                    _summed_energy_array,
                    np.ones(len(_summed_energy_array)),
                )
            
            ### cross talk correct
            else:
                ## this is likely to be pretty slow, we can try to make something faster but id need to think about it
                _mult_energy_array = (
                    data[eval(_cut_string,globs,data)&(data["period"]==_period)]["geds"]["energy"].to_numpy()
                )
                
                _mult_channel_array = (
                            data[eval(_cut_string,globs,data)&(data["period"]==_period)]["geds"]["hit_rawid"].to_numpy()
                )
                ### apply the category selection
                
                _corrected_energy_array=[]
                _corrected_energy_1=[]
                _corrected_energy_2=[]


                # apply cross talk corrected
                for energy,channel in zip(_mult_energy_array,_mult_channel_array):
                
                    channel_names= np.array([geds_mapping["ch"+str(c)] for c in channel])
                    energies_corrected,val = cross_talk_corrected_energies(energy,channel_names,cross_talk_matrix)
                    if (val==True):
                        _corrected_energy_1.append(min(energies_corrected[0],energies_corrected[1]))
                        _corrected_energy_2.append(max(energies_corrected[0],energies_corrected[1]))
                        _corrected_energy_array.append(energies_corrected)
                    else:
                        raise ValueError("Error: cross talk correction didnt work")

                if len(_corrected_energy_array)>0:
                    _summed_energy_array=np.sum(_corrected_energy_array,axis=1)
                else:
                    _summed_energy_array=np.array([])
                    continue
                
                ### get the categories
                for name in names_m2:
                    
                    ### select the right events
                    if (name!="all" ):
                        categories =get_m2_categories(_mult_channel_array,channel2string,channel2position)
                        string_diff,floor_diff =get_string_row_diff(_mult_channel_array,channel2string,channel2position)
                        
                        if ("cat" in name):
                            cat=int(name.split("_")[1])
                            _corrected_energy_1_tmp=np.array(_corrected_energy_1)[np.where(categories==cat)[0]]
                            _corrected_energy_2_tmp=np.array(_corrected_energy_2)[np.where(categories==cat)[0]]
                            _summed_energy_array_tmp=np.array(_summed_energy_array)[np.where(categories==cat)[0]]
                            
                        elif ("sd" in name):
                            sd=int(name.split("_")[1])
                            
                            ids =np.where((string_diff==sd))[0]
                            _corrected_energy_1_tmp=np.array(_corrected_energy_1)[ids]
                            _corrected_energy_2_tmp=np.array(_corrected_energy_2)[ids]
                            _summed_energy_array_tmp=np.array(_summed_energy_array)[ids]
                    
                    ### all case
                    else:
                        _corrected_energy_1_tmp=np.array(_corrected_energy_1)
                        _corrected_energy_2_tmp=np.array(_corrected_energy_2)
                        _summed_energy_array_tmp=np.array(_summed_energy_array)
                        
                        
                    if (len(_corrected_energy_array)==0): 
                        continue

                    _corrected_energy_1_tmp=np.array(_corrected_energy_1_tmp) 
                    _corrected_energy_2_tmp=np.array(_corrected_energy_2_tmp)

                    if len(_summed_energy_array_tmp) == 0: 
                        continue
                    if ("th2" in _cut_dict):
                        sum_hists[_cut_name][name].FillN(
                            len(_corrected_energy_1_tmp),
                            _corrected_energy_1_tmp,_corrected_energy_2_tmp,
                            np.ones(len(_summed_energy_array_tmp)),
                        )
                    else:
                        sum_hists[_cut_name][name].FillN(
                            len(_summed_energy_array_tmp),
                            _summed_energy_array_tmp,
                            np.ones(len(_summed_energy_array_tmp)),
                        )
    for _cut_name in hists:
        hists[_cut_name]["all"] = ROOT.TH1F(
            f"{_cut_name}_all",
            "All energy deposits",
            rconfig["hist"]["nbins"],
            rconfig["hist"]["emin"],
            rconfig["hist"]["emax"],
        )
        for _type in ["bege", "coax", "icpc", "ppc"]:
            hists[_cut_name][_type] = ROOT.TH1F(
                f"{_cut_name}_{_type}",
                f"All {_type} energy deposits",
                rconfig["hist"]["nbins"],
                rconfig["hist"]["emin"],
                rconfig["hist"]["emax"],
            )
        for _rawid, _name in geds_mapping.items():
            hists[_cut_name][chmap[geds_mapping[_rawid]]["type"]].Add(
                hists[_cut_name][_rawid]
            )
            hists[_cut_name]["all"].Add(hists[_cut_name][_rawid])

    # write the hists to file (but only if they have none zero entries)
    # Changes the names to drop type_ etc
    logger.info(f"... save histos")
    out_file = uproot.recreate("outputs/"+out_name)
    for _cut_name, _hist_dict in hists.items():
        dir = out_file.mkdir(_cut_name)
        for key, item in _hist_dict.items():
            dir[key] = item

    for _cut_name, _hist_dict in run_hists.items():
        for key, item in _hist_dict.items():
            out_file[_cut_name+"/"+key] = item

    for _cut_name, diri in sum_hists.items():
        dir = out_file.mkdir(_cut_name)
        for name,_hist in diri.items():
            dir[name] = _hist

    out_file.close()
    logger.info(f"... done!")


if __name__ == "__main__":
    main()
