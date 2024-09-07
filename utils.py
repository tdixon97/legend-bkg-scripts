import legendstyles
import copy
from datetime import datetime

import hist
import matplotlib.pyplot as plt
import numpy as np
from hist import Hist
from legendmeta import LegendMetadata
from scipy.stats import poisson
from datetime import datetime, timezone

import json


plt.style.use(legendstyles.LEGEND)

def number2name(meta:LegendMetadata,number:str)->str:
    """Get the channel name given the channel number.                                                                                                                                                      
                                                                                                                                                                                                           
    Parameters:                                                                                                                                                                                            
        meta (LegendMetadata): An object containing metadata information.                                                                                                                                  
        number (str): The channel number to be converted to a channel name.                                                                                                                                
                                                                                                                                                                                                           
    Returns:                                                                                                                                                                                               
        str: The channel name corresponding to the provided channel number.                                                                                                                                
                                                                                                                                                                                                           
    Raises:                                                                                                                                                                                                
        ValueError: If the provided channel number does not correspond to any detector name.                                                                                                               
                                                                                                                                                                                                           
    """

    chmap = meta.channelmap(datetime.now())

    for detector, dic in meta.dataprod.config.on(datetime.now())["analysis"].items():
        if  f"ch{chmap[detector].daq.rawid:07}"==number:
            return detector
    raise ValueError("Error detector {} does not have a name".format(number))




def string_to_edges(input_string):

    segments = input_string.split(",")
    result = []

    for segment in segments:
        start, value, end = segment.split(":")
        start = float(start)
        value = float(value)
        end = float(end)

        while start <= end:
            result.append(start)
            start += value

    return result


def variable_rebin(histo, edges: list):
    """
    Perform a variable rebinning of a hist object
    Parameters:
        - histo: The histogram object
        - edges: The list of the bin edges
    Returns:
        - variable rebin histo
    """
    histo_var = Hist.new.Variable(edges).Double()
    for i in range(histo.size - 2):
        cent = histo.axes.centers[0][i]

        histo_var[cent * 1j] += histo.values()[i]

    return histo_var


def normalise_histo(hist, factor=1):
    """
    Normalise a histogram into units of counts/keV (by bin width)
     Parameters:
    ----------------------
        - histo: Hist object
        - factor: a scaling factor to multiply the histo by
    Returns
    ----------------------
        - normalised histo
    """

    widths = np.diff(hist.axes.edges[0])

    for i in range(hist.size - 2):
        hist[i] /= widths[i]
        hist[i] *= factor

    return hist


def get_run_times(
    metadb: LegendMetadata, analysis_runs: dict, verbose: bool = True, ac=[], off=[]
) -> dict:
    """Get the livetime from the metadata
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
    output = {}
    first_time = None
    # loop over periods
    for period in runinfo.keys():
        if period in analysis_runs.keys():
            output[period] = {}

        # loop over runs
        for run in runinfo[period].keys():

            # skip 'bad' runs
            if period in analysis_runs.keys() and run in analysis_runs[period]:

                if (
                    "phy" in runinfo[period][run].keys()
                    and "livetime_in_s" in runinfo[period][run]["phy"].keys()
                ):
                    timestamp = datetime.strptime(
                        runinfo[period][run]["phy"]["start_key"], "%Y%m%dT%H%M%SZ"
                    )

                    ch = metadb.channelmap(
                        metadb.dataprod.runinfo[period][run]["phy"]["start_key"]
                    )

                    geds_list = [
                        (_name, _dict["daq"]["rawid"])
                        for _name, _dict in ch.items()
                        if ch[_name]["system"] == "geds"
                        and ch[_name]["analysis"]["usability"] in ["on", "no_psd"]
                    ]

                    start_time = int(timestamp.timestamp())
                    if first_time is None:
                        first_time = start_time

                    start_time -= first_time
                    time = runinfo[period][run]["phy"]["livetime_in_s"]
                    end_time = start_time + time
                    mass = 0
                    for det in geds_list:

                        if (det[1] not in ac) and (det[1] not in off):
                            mass += ch[det[0]].production.mass_in_g / 1000
                    output[period][run] = [start_time, end_time, mass]

    return output


def get_smallest_ci(N, x, y):
    integral = y[np.argmax(y)]
    bin_id_l = np.argmax(y)
    bin_id_u = np.argmax(y)

    integral_tot = np.sum(y)
    while integral < 0.683 * integral_tot:

        # get left bin
        if bin_id_l > 0 and bin_id_l < len(y):
            c_l = y[bin_id_l - 1]
        else:
            c_l = 0

        if bin_id_u > 0 and bin_id_u < len(y):
            c_u = y[bin_id_u + 1]
        else:
            c_u = 0

        if c_l > c_u:
            integral += c_l
            bin_id_l -= 1
        else:
            integral += c_u
            bin_id_u += 1

    low_quant = x[bin_id_l]
    high_quant = x[bin_id_u]
    return N - low_quant, high_quant - N


def get_error_bar(N: float):
    """
    A poisson error-bar for N observed counts.
    """

    x = np.linspace(0, 5 + 2 * N, 5000)
    y = poisson.pmf(N, x)
    histo = Hist.new.Reg(5000, 0, 0.5 + 2 * N).Double()

    for i in range(histo.size - 2):
        histo[i] = y[i]

    return get_smallest_ci(N, x, y)[0], get_smallest_ci(N, x, y)[1], histo


def get_hist(
    obj, range: tuple = (132, 4195), bins: int = 10, variable=None, spectrum="mul_surv"
):
    """
    Extract the histogram (hist package object) from the uproot histogram
    Parameters:
        - obj: the uproot histogram
        - range: (tuple): the range of bins to select (in keV)
        - bins (int): the (constant) rebinning to apply
    Returns:
        - hist
    """

    if spectrum == "mul2_surv_e1":
        h = obj.to_hist().project(1)[range[0] : range[1]]
    elif spectrum == "mul2_surv_e2":
        h = obj.to_hist().project(0)[range[0] : range[1]]
    else:
        h = obj.to_hist()[range[0] : range[1]]

    if variable is not None:
        h = normalise_histo(variable_rebin(h, variable))

    else:
        h = h[hist.rebin(bins)]
    return h


def integrate_hist(hist, energies):

    integral = 0
    for e in energies:
        integral += integrate_hist_one_range(hist, e[0], e[1])

    return integral


def integrate_hist_one_range(hist, low, high):
    """Integrate the histogram"""

    bin_centers = hist.axes.centers[0]

    values = hist.values()
    lower_index = np.searchsorted(bin_centers, low, side="right")
    upper_index = np.searchsorted(bin_centers, high, side="left")
    bin_contents_range = values[lower_index:upper_index]
    bin_centers_range = bin_centers[lower_index:upper_index]

    return np.sum(bin_contents_range)


def sideband_counting(hist, low, center_low, center_high, high, pdf=None, name=""):
    """
    A counting analysis using a bayesian approach,
    the data is modelled as a flat bkg and a central peak,
    the posterior is evaliuated on a grid then the marginalised posterior on
    the signal counts is computed.

    Parameters:
        hist: histogram object
        low, center_low,center_high,high: edges of the bins
    Returns:
        low,med,high
    """

    # get the counts per bin
    n1 = integrate_hist_one_range(hist, low, center_low)
    n2 = integrate_hist_one_range(hist, center_low, center_high)
    n3 = integrate_hist_one_range(hist, center_high, high)

    max_s = 10 + n2 * 2
    max_b = 10 + n2 * 2
    # get bin widths
    w1 = center_low - low
    w2 = center_high - center_low
    w3 = high - center_high

    def likelihood(s, b):
        """
        Likelihood function based on "S" signal counts, "B" background
        """
        return (
            poisson.pmf(n1, b * w1 / w2)
            * poisson.pmf(n2, b + s)
            * poisson.pmf(n3, b * w3 / w2)
        )

    histo = Hist.new.Reg(200, 0, max_s).Reg(200, 0, max_b).Double()

    w, x, y = histo.to_numpy()

    x_2d = np.tile(x, (len(y), 1))
    y_2d = np.tile(y, (len(x), 1)).T

    l = likelihood(x_2d, y_2d)
    maxi = np.max(l)

    best_fit = (
        x_2d.flatten()[np.argmax(l.flatten())],
        y_2d.flatten()[np.argmax(l.flatten())],
    )
    w_x = np.sum(l, axis=0)
    w_y = np.sum(l, axis=1)

    if pdf is not None:
        style = {
            "yerr": False,
            "flow": None,
            "lw": 0.6,
        }
        fig, axes = plt.subplots(
            1, 1, figsize=(3, 3), sharex=True, gridspec_kw={"hspace": 0}
        )

        histo_x = Hist.new.Reg(200, 0, max_s).Double()
        histo_y = Hist.new.Reg(200, 0, max_b).Double()

        for i in range(histo_x.size - 2):
            histo_x[i] = w_x[i]
            histo_y[i] = w_y[i]

        fig, axes = plt.subplots(
            1, 1, figsize=(3, 3), sharex=True, gridspec_kw={"hspace": 0}
        )
        axes.set_xlabel("Counts")
        axes.set_ylabel("Prob [arb units]")

        histo_x.plot(ax=axes, **style, histtype="fill", alpha=0.5, label="Signal")
        histo_y.plot(ax=axes, **style, histtype="fill", alpha=0.5, label="Bakground")
        axes.set_title(name)
        axes.set_xlim(0, max_s)
        plt.legend(loc="best", frameon=True, facecolor="white")

        pdf.savefig()
        plt.close("all")

        hist_fit = copy.deepcopy(hist)
        bw = np.diff(hist.axes.centers[0])[0]
        maxi = 0
        for i in range(hist.size - 2):
            xt = hist.axes.centers[0][i]

            b1 = best_fit[1] * w1 / w2
            b3 = best_fit[1] * w3 / w2

            if xt < low:
                hist_fit[i] = 0
            elif xt < center_low:
                hist_fit[i] = bw * b1 / w1
            elif xt < center_high:
                hist_fit[i] = bw * (best_fit[1] + best_fit[0]) / w2
            elif xt < high:
                hist_fit[i] = bw * b3 / w3
            else:
                hist_fit[i] = 0

            if xt > low and xt < high and hist[i] > maxi:
                maxi = hist[i]

        fig, axes = plt.subplots(
            1, 1, figsize=(3, 3), sharex=True, gridspec_kw={"hspace": 0}
        )
        hist.plot(ax=axes, **style, histtype="fill", alpha=0.5, label="Data")
        hist_fit.plot(ax=axes, **style, color="black", alpha=0.5, label="Fit")

        axes.set_title(name)
        axes.set_xlim(low - 5, high + 5)
        axes.set_ylim(0, maxi + 5)
        axes.set_xlabel("Energy [keV]")
        axes.legend(loc="best")
        pdf.savefig()
        plt.close("all")

    return (
        x[np.argmax(w_x)],
        get_smallest_ci(x[np.argmax(w_x)], x, w_x)[0],
        get_smallest_ci(x[np.argmax(w_x)], x, w_x)[1],
    ), histo_x


def sample_hist(hist, n):
    """
    Generate samples from a histogram
    """
    edges = hist.axes[0].edges
    counts = hist.view()
    probabilities = counts / np.sum(counts)

    sample_indices = np.random.choice(range(len(edges) - 1), size=n, p=probabilities)

    sampled_values = np.random.uniform(edges[sample_indices], edges[sample_indices + 1])

    return sampled_values


def get_psd_usable(period,ch,detector):

    use= ((((ch[detector]["analysis"]["psd"]["is_bb_like"]  == "low_aoe & high_aoe")
                                and ch[detector]["analysis"]["psd"]["status"]["low_aoe"] == "valid"
                                and ch[detector]["analysis"]["psd"]["status"]["high_aoe"] == "valid"
                               ) or 
                            (((ch[detector]["analysis"]["psd"]["is_bb_like"]  == "low_aoe & lq")
                                  and ch[detector]["analysis"]["psd"]["status"]["low_aoe"] == "valid"
                                    and ch[detector]["analysis"]["psd"]["status"]["lq"] == "valid"
                            ) and period in ["p08","p09"]  
                                 )
                               ) and (ch[detector]["analysis"]["usability"] == 'on'))

    return use
      
def get_exposure(group:list,periods:list=["p03","p04","p06","p07","p08","p09"],psd_usable:bool=False)->float:
    """
    Get the livetime for a given group a list of strs of channel names or detector types or "all"
    Parameters:
        group: list of str
        periods: list of periods to consider default is 3,4,6,7 (vancouver dataset)
        psd_use: get exposure usable for psd (a subset)
    Returns:
        - a float of the exposure (summed over the group)
    """

    meta = LegendMetadata("../LEGEND/legend-metadata")

    analysis_runs=meta.dataprod.config.analysis_runs
    ### also need the channel map
  
    groups_exposure=np.zeros(len(group))

    for period in periods:

        if period not in analysis_runs.keys():
            continue
        ## loop over runs
        for run,run_dict in meta.dataprod.runinfo[period].items():
            
            ## skip 'bad' runs
            if ( run in analysis_runs[period] and "phy" in run_dict):

                ch = meta.channelmap(run_dict["phy"]["start_key"])

                for i in range(len(group)):
                    item =group[i]

                    if (psd_usable is False):
                        if item=="all":
                            geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["analysis"]["usability"] in ["on","no_psd"]]
                        elif ((item=="bege")or(item=="icpc")or (item=="coax")or (item=="ppc")):
                            geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["type"]==item and ch[_name]["analysis"]["usability"] in ["on","no_psd"]]
                        else:
                            geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and f"ch{ch[_name]['daq']['rawid']}"==item and ch[_name]["analysis"]["usability"] in ["on","no_psd"]]

                    else:
                        if item=="all":
                            geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and get_psd_usable(period,ch,_name) is True]
                        elif ((item=="bege")or(item=="icpc")or (item=="coax")or (item=="ppc")):
                            geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and ch[_name]["type"]==item and get_psd_usable(period,ch,_name) is True]
                        else:
                            geds_list= [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and f"ch{ch[_name]['daq']['rawid']}"==item and get_psd_usable(period,ch,_name) is True]

                    ## loop over detectors
                    for det in geds_list:
                        mass = ch[det].production.mass_in_g/1000
                        if "phy" in run_dict:
                            groups_exposure[i] += mass * (run_dict["phy"]["livetime_in_s"]/(60*60*24*365.25
                                                                                                ))
   
    return np.sum(groups_exposure)

def get_channels_map():
    """ Get the channels map
    Parameters:
        None
    Returns:
        - a dictonary of string channels of the form

        {
        1:
        {      
            ["channel1","channel2", etc (the names V002 etc)
        },
        }

        - a dictonary of types like

        {   
        1:
        {
            ["icpc","bege",...]
        }
        
    """

    meta = LegendMetadata("../LEGEND/legend-metadata/")
 
    ### 1st July should be part of TAUP dataset
    time = datetime(2023, 7, 1, 00, 00, 00, tzinfo=timezone.utc)
    chmap = meta.channelmap(datetime.now())

   
    string_channels={}
    string_types={}
    for string in range(1,12):
        if (string==6):
            continue
        channels_names = [
            f"ch{chmap[detector].daq.rawid:07}"
            for detector, dic in meta.dataprod.config.on(time)["analysis"].items()
            if dic["processable"] == True
            and detector in chmap 
            and chmap[detector]["system"] == 'geds'
            and chmap[detector]["location"]["string"] == string
        ]
        channels_types = [
            chmap[detector]['type']
            for detector, dic in meta.dataprod.config.on(time)["analysis"].items()
            if dic["processable"] == True
            and detector in chmap
            and chmap[detector]["system"] == 'geds'
            and chmap[detector]["location"]["string"] == string
        ]
       
        string_channels[string]=channels_names
        string_types[string]=channels_types
    
    return string_channels,string_types


def number2name(meta:LegendMetadata,number:str)->str:
    """Get the channel name given the channel number.

    Parameters:
        meta (LegendMetadata): An object containing metadata information.
        number (str): The channel number to be converted to a channel name.

    Returns:
        str: The channel name corresponding to the provided channel number.

    Raises:
        ValueError: If the provided channel number does not correspond to any detector name.

    """
 
    chmap = meta.channelmap(datetime.now())

    for detector, dic in meta.dataprod.config.on(datetime.now())["analysis"].items():
        if  f"ch{chmap[detector].daq.rawid:07}"==number:
            return detector
    raise ValueError("Error detector {} does not have a name".format(number))
            




def get_det_types(group_type:str,string_sel:int=None,det_type_sel:str=None,level_groups:str="cfg/level_groups_Sofia.json",psd_usable:bool=False):
    """
    Extract a dictonary of detectors in each group

    Parameters:
        grpup_type (str): 
            The type of grouping (sum,types,string,chan,floor or all)
        string_sel (int, optional): 
            An integer representing the selection of a particular string. Default is None ie select all strings
        det_type_sel (str, optional): 
            A string representing the selection of a particular detector type. Default is None (select all det types)
        level_groups (str, optional) a str of where to find a json file to get the groupings for floor default level_groups.json
    Returns:
        tuple: A tuple containing three elements:
            - A dictionary containing the channels in each group, with the structure
                "group":
                {
                    "names": ["det1","det2", ...],
                    "types": ["icpc","bege",....],
                    "exposure": XXX, 
                },
            - A string representing the name of this grouping
            - A list of the number of channels per grouping - only used for the chans option

    Example:
        det_info, selected_det_type, selected_det = get_det_types("sum")
    Raises:
    """


    ### extract the meta data
    meta = LegendMetadata("../LEGEND/legend-metadata")
    Ns=[]

    ### just get the total summed over all channels
    if (group_type=="sum" or group_type=="all"):
        if (det_type_sel is None):
            det_types={"all":{"names":["icpc","bege","ppc","coax"],"types":["icpc","bege","ppc","coax"]}}
            namet="all"
        else:
            det_types={"all":{"names":[det_type_sel],"types":[det_type_sel]}}
            namet="all"
    elif (group_type=="types"):
        if (det_type_sel is None):

            det_types={"icpc": {"names":["icpc"],"types":["icpc"]},
                "bege": {"names":["bege"],"types":["bege"]},
                "ppc": {"names":["ppc"],"types":["ppc"]},
                "coax": {"names":["coax"],"types":["coax"]}
                }
            namet="by_type"
        else:
            det_types={det_type_sel: {"names":[det_type_sel],"types":[det_type_sel]}
                }
            namet="by_type"
        
    elif (group_type=="string"):
        string_channels,string_types = get_channels_map()
        det_types={}

        ### loop over strings
        for string in string_channels.keys():
            det_types[string]={"names":[],"types":[]}

            for dn,dt in zip(string_channels[string],string_types[string]):
                
                if (det_type_sel==None or det_type_sel==dt):
                    det_types[string]["names"].append(dn)
                    det_types[string]["types"].append(dt)

        namet="string"

    elif (group_type=="chan"):
        string_channels,string_types = get_channels_map()
        det_types={}
        namet="channels"
        Ns=[]
        N=0
        for string in string_channels.keys():

            if (string_sel==None or string_sel==string):
                chans = string_channels[string]
                types = string_types[string]

                for chan,type in zip(chans,types):
                    if (det_type_sel==None or type==det_type_sel):
                        det_types[chan]={"names":[chan],"types":[type]}
                        N+=1
            Ns.append(N)

    elif (group_type=="floor"):
   
        string_channels,string_types = get_channels_map()
        groups=["top","mid_top","mid","mid_bottom","bottom"]
        det_types={}
        namet="floor"
        for group in groups:
            det_types[group]={"names":[],"types":[]}
        
        for string in string_channels.keys():
            channels = string_channels[string]
            
            ## loop over channels per string
            for i in range(len(channels)):
                chan = channels[i]
                name = number2name(meta,chan)
                group = get_channel_floor(name,groups_path=level_groups)
                if (string_sel==None or string_sel==string):
                    
                    if (det_type_sel==None or string_types[string][i]==det_type_sel):
                        det_types[group]["names"].append(chan)
                        det_types[group]["types"].append(string_types[string][i])
    else:
        raise ValueError("group type must be either floor, chan, string sum all or types")
    
    ### get the exposure
    ### --------------
    for group in det_types:
        list_of_names= det_types[group]["names"]
        exposure = get_exposure(list_of_names,psd_usable=psd_usable)

        det_types[group]["exposure"]=exposure

    return det_types,namet,Ns

def get_data_counts_total(spectrum:str,det_types:dict,regions:dict,file:str,det_sel:str=None,key_list=[]):
    """
    Function to get the data counts in different regions
    Parameters:
        - spectrum:  the data spectrum to use (str)
        - det_types: the dictonary of detector types
        - regions:   the dictonary of regions
        - file   :   str path to the file
        - det_sel : a particular detector type (def None)
    Returns:
        - a dictonary of the counts in each region
    
    """
    data_counts_total ={}
    for det_name,det_info in det_types.items():

        det_list=det_info["names"]
        dt=det_info["types"]
        data_counts={}
        for key in key_list:
            data_counts[key]=0

        for det,type in zip(det_list,dt):
          
            if (type==det_sel or det_sel==None):
                data_counts = sum_effs(data_counts,get_data_counts(spectrum,det,regions,file))

        data_counts_total[det_name]=data_counts

    return data_counts_total

def get_data_counts(spectrum:str,det_type:str,regions:dict,file):
    """
    
    Function to get the counts in the data in a range:
    Parameters:
        - spectrum (str): the spectrum to use
        - det_type (str): the detector type to select
        - regions (dict): dictonary of regions to get counts in
        - file: uproot file of the data
    Returns:
        - dict of the data counts
    
    """

    if "{}/{}".format(spectrum,det_type) in file:
        hist =file["{}/{}".format(spectrum,det_type)]
    else:
        data_counts={}
        for region in regions.keys():
            data_counts[region]=0
        return data_counts
    
    data_counts={}
    hist=hist.to_hist()
    for region,rangel in regions.items():
        data=0
        for i in range(len(rangel)):
            data+= float(integrate_hist_one_range(hist,rangel[i][0],rangel[i][1]))
        
        data_counts[region]=data
    return data_counts


def sum_effs(eff1:dict,eff2:dict)->dict:
    """ 
    Function to sum the two efficiency dictonaries up to two layers
    
    Parameters:
        - eff1: (dict) the first dictonary
        - eff2: (dict) the second
    Returns:
        - dictonary where every item with the same key is summed
    """


    dict_sum={}
    for key in set(eff1) | set(eff2):  # Union of keys from both dictionaries

        ## sum two layers
        dict_sum[key]={}
     
        if (isinstance(eff1[key],dict) or isinstance(eff2[key],dict)):
      
            for key2 in set(eff1[key]) | set(eff2[key]):
                dict_sum[key][key2] = eff1[key].get(key2, 0) + eff2[key].get(key2, 0)

        ## sum one layer
        else:
            dict_sum[key]= eff1.get(key, 0) + eff2.get(key, 0)
    return dict_sum

def get_channel_floor(name:str,groups_path:str="cfg/level_groups_Sofia.json")->str:
    """Get the z group for the detector
    Parameters:
        name: (str) 
            - the channel name
        groups_path: (str, optional)
            - the path to a JSON file containing the groupings, default level_groups.json
    Returns:
        - str: The group a given channel corresponds to
    Raises:
        - Value error if the channel isnt in the json file
    """

    with open(groups_path, 'r') as json_file:
        level_groups =json.load(json_file)

    for key,chan_list in level_groups.items():
        if name in chan_list:
            return key
        
    raise ValueError("Channel {} has no position ".format(name))