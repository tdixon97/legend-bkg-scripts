"""
Script to load the data for the LEGEND-200 background model
Main Authors: Sofia Calgaro, Toby Dixon, Luigi Pertoldi based on a script from William Quinn
"""

import argparse
import glob
import json
import copy
import sys
import logging
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import awkward as ak
import numpy as np
import ROOT
import uproot
from legendmeta import LegendMetadata
from lgdo import lh5
from tqdm import tqdm

# -----------------------------------------------------------
# LOGGER SETTINGS
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# -----------------------------------------------------------


def get_vectorised_converter(mapping):
    def channel2other(channel):
        """Extract which string a given channel is in"""

        return mapping[f"ch{channel}"]

    return np.vectorize(channel2other)


def get_string_row_diff(channel_array, channel2string, channel2position):
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

    channel_array = np.vstack(channel_array)
    channel_one = channel_array[:, 0].T
    channel_two = channel_array[:, 1].T

    # convert to the list of strings
    string_one = channel2string(channel_one)
    string_two = channel2string(channel_two)
    string_diff_1 = (string_one - string_two) % 11
    string_diff_2 = (-string_one + string_two) % 11
    string_diff = np.array([min(a, b) for a, b in zip(string_diff_1, string_diff_2)])

    position_one = channel2position(channel_one)
    position_two = channel2position(channel_two)

    floor_diff = np.abs(position_one - position_two)

    return np.array(string_diff), np.array(floor_diff)


def get_m2_categories(channel_array, channel2string, channel2position):
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

    channel_array = np.vstack(channel_array)
    channel_one = channel_array[:, 0].T
    channel_two = channel_array[:, 1].T

    # convert to the list of strings
    string_one = channel2string(channel_one)
    string_two = channel2string(channel_two)

    same_string = string_one == string_two
    position_one = channel2position(channel_one)
    position_two = channel2position(channel_two)
    neighbour = np.abs(position_one - position_two) == 1

    is_cat_one = (same_string) & (neighbour)
    is_cat_two = (same_string) & (~neighbour)
    is_cat_three = ~same_string
    category = 1 * is_cat_one + 2 * is_cat_two + 3 * is_cat_three
    return np.array(category)


def make_unphysical_rate_plot(run,period,unphysical,geds_mapping,chmax=30,is_forced=False,norm=None,time=None):

    # get total number of events
    n=0
    chans=[]
    events=[]
    counter=0
    unphysical=dict(unphysical)
    n_chans= len(list(geds_mapping.keys()))
    for key, item in sorted(unphysical.items(), key=lambda item: item[1], reverse=True):        
        n+=item
        if (geds_mapping[f"ch{key}"]=="V07646A"):
            continue
        if (counter<chmax):
            chans.append(geds_mapping[f"ch{key}"])
            if (norm is not None):
                events.append(100*item/norm)
            elif (time is not None):
                events.append(100*item/time)
            else:
                events.append(item)
        counter+=1

    logger.info(f"In total we have {n} unphysical events")
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(chans,events)
    if (norm is not None):
        ax.set_ylabel("Fraction of unphysical hits [%]")
    elif (time is not None):
        ax.set_ylabel("Rate of unphysical hits [mHz]")
    else:
        ax.set_ylabel("Number of unphysical events")

    plt.xticks(rotation=90, fontsize=8) 
    if (norm is None and time is None):
        plt.axhline(y=n/n_chans,label="Average",linestyle="--",color="red")
    plt.legend()
    if (norm is not None):
        plt.title(f"Fraction of unphysical hits for {period} - {run}")
    elif (time is not None):
        plt.title(f"Rate of unphysical hits for {period} - {run}")

    else:
        plt.title(f"Number of unphysical hits for {period} - {run}")

    plt.tight_layout()
    if (is_forced and norm is None):

        plt.savefig(f"plots/unphysical_forced_trigger_{period}_{run}.pdf")
    elif (norm is not None):
        plt.savefig(f"plots/unphysical_forced_trigger_norm_{period}_{run}.pdf")
    elif (time is not None):
        plt.savefig(f"plots/unphysical_rate_{period}_{run}.pdf")

    else:
        plt.savefig(f"plots/unphysical_{period}_{run}.pdf")


def get_data_awkard(
    cfg: dict,
    period=None,
    run = None,
    metadb=LegendMetadata(),
):
    """
    A function to load the evt tier data into an awkard array also getting the energies
    Parameters
    ----------
        - cfg: path to the data for each period
        - period: period to use
        - run_list: run to process
        - metadab
    Returns
    -------
        - an awkard array of the data
    """

    data = None

    # loop over period and run (so we can save this)
    logger.info("Starting to load evt tier for...")
    logger.info(f"...... {period} {run}")

    tier = cfg[period]["tier"]
    evt_path = cfg[period]["evt_path"]

    logger.debug(evt_path + "/" + tier + f"/phy/{period}/{run}/")

    if (os.path.exists(evt_path + "/" + tier + f"/phy/l200-{period}-{run}-phy-tier_pet.lh5")):
        fl_evt = glob.glob(evt_path + "/" + tier + f"/phy/l200-{period}-{run}-phy-tier_pet.lh5")
    else:
        fl_evt = glob.glob(evt_path + "/" + tier + f"/phy/{period}/{run}/*")

    for idx,f_evt in enumerate(fl_evt):
        if (idx%50==0):
            logger.debug(f"Reading >>> {f_evt} ({idx} out of {len(fl_evt)})")

        d_evt = lh5.read_as("evt", f_evt, library="ak")
        d_evt["period"] = period
        d_evt["run"] = run

        data = d_evt if data is None else ak.concatenate((data, d_evt))

    return data

def main():
    parser = argparse.ArgumentParser(
        description="Script to load the data for the LEGEND-200 background model"
    )
    parser.add_argument(
        "--output", help="Name of output root file, eg l200a-p10-r000-dataset-tmp-auto"
    )
    parser.add_argument("--p", help="List of periods to inspect")
    parser.add_argument("--r", help="List of runs to inspect",default=None)

    parser.add_argument("--c", help="Cycle to use")
    parser.add_argument(
        "--use_qc",
        default=1,
        help="Set to 0 if you want to not use QCs",
    )
    args = parser.parse_args()
    cycle= args.c
    run_list=args.r.split(",")
    print(run_list)
    use_qc =bool(int(args.use_qc))
    logger.info(f"use qc {use_qc}")
    config_path = "cfg/build-pdf-config.json"
    prod_cycle = f"/data2/public/prodenv/prod-blind/{cycle}/"
    meta_path  = f"/data2/public/prodenv/prod-blind/{cycle}/inputs/"
    tier="pet"
    paths_cfg = {
        "p03": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p04": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p06": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p07": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p08": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p09": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p10": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        },
        "p11": {
            "tier": tier,
            "evt_path": f"{prod_cycle}/generated/tier/",

        }
    }
    out_name = f"{args.output}"
    periods = args.p
    if (isinstance(periods,str)):
        periods=[periods]

    with Path(config_path).open() as f:
        rconfig = json.load(f)
    # get the metadata information / mapping
    # --------------------------------------
    logger.info("... get the metadata information / mapping")
    metadb = LegendMetadata(f"/data2/public/prodenv/prod-blind/{cycle}/inputs/")
    chmap = metadb.channelmap(rconfig["timestamp"])

    geds_mapping = {
        f"ch{_dict['daq']['rawid']}": _name
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    geds_strings = {
        f"ch{_dict['daq']['rawid']}": _dict["location"]["string"]
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    geds_positions = {
        f"ch{_dict['daq']['rawid']}": _dict["location"]["position"]
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    geds_types = {
        f"ch{_dict['daq']['rawid']}": _dict["type"]
        for _name, _dict in chmap.items()
        if chmap[_name]["system"] == "geds"
    }
    channel2string = get_vectorised_converter(geds_strings)
    channel2position = get_vectorised_converter(geds_positions)
    channel2type = get_vectorised_converter(geds_types)

    # analysis runs
    runs_old = metadb.dataprod.runinfo
    runs=copy.deepcopy(runs_old)
    if (run_list is not None):
        for period in periods:
            for run in runs_old[period]:
                if (run not in run_list):
                    runs[period].pop(run)

    logger.info("... create the histos to fill (full period)")
    run_hist_list=["","_icpc","_bege","_ppc","_coax"]
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
                hists[_cut_name][_rawid] = ROOT.TH1F(
                    hist_name, hist_title, nbins, emin, emax
                )

    # save also hists per run (total only now)
    logger.info("... create the histos to fill (run by run)")
    run_hists = {}
    for _cut_name in rconfig["cuts"]:
        if not rconfig["cuts"][_cut_name]["is_sum"]:
            run_hists[_cut_name] = {}
            for _period, _run_list in runs.items():
                if (_period not in periods):
                    continue
                for run in _run_list:
                    for name in run_hist_list:
                        hist_name = f"{_cut_name}_{_period}_{run}{name}"
                        hist_title = f"{_period} {run} energy deposits"
                        nbins = rconfig["hist"]["nbins"]
                        emin = rconfig["hist"]["emin"]
                        emax = rconfig["hist"]["emax"]
                        run_hists[_cut_name][f"{_period}_{run}{name}"] = ROOT.TH1F(
                            hist_name, hist_title, nbins, emin, emax
                        )
                    

    sum_hists = {}
    string_diff = np.arange(7)
    names_m2 = [f"sd_{item1}" for item1 in string_diff]
    names_m2.extend(["all", "cat_1", "cat_2", "cat_3"])
    names_m2.extend(["e1_icpc", "e1_bege", "e1_ppc", "e1_coax"])

    # now the summed histos
    for _cut_name in rconfig["cuts"]:
        cut_info = rconfig["cuts"][_cut_name]

        if cut_info["is_sum"] is False:
            continue
        if "is_2d" not in cut_info or cut_info["is_2d"] is False:
            hist_type = ROOT.TH1F
        elif "is_2d" in cut_info:
            hist_type = ROOT.TH2F

        hist_title = "summed energy deposits"

        if hist_type == ROOT.TH1F:
            sum_hists[_cut_name] = {}

            for name in names_m2:
                hist_name = f"{_cut_name}_{name}_summed"
                sum_hists[_cut_name][name] = hist_type(
                    hist_name,
                    hist_title,
                    rconfig["hist"]["nbins"],
                    rconfig["hist"]["emin"],
                    rconfig["hist"]["emax"],
                )
        elif hist_type == ROOT.TH2F:

            sum_hists[_cut_name] = {}

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
                    rconfig["hist"]["emax"],
                )
   
    logger.info(f"... fill histos")
    globs = {"ak": ak, "np": np}

    conversions = {
        "mul ": "geds.multiplicity ",
        "mul_is_good ": "geds.on_multiplicity ",
        "and": "&",
        "npe_tot": "spms.energy_sum",
    }

    energy_name = "energy"
    qcs_flag = "is_good_channel"
    rawid_name ="rawid"
    evt_quality_flag="is_bb_like"
    for period, _run_list in runs.items():

        if period not in periods:
            continue

        for run in _run_list:
            if "phy" not in runs[period][run]:
                continue
            if (period == "p10" and run =="r002"):
                continue
            data = get_data_awkard(
                cfg=paths_cfg,
                period=period,
                run =run,
                metadb = metadb
            )
            if (data is None):
                continue
                

            data["geds","on_multiplicity"]=ak.sum(data["geds","quality","is_good_channel"],axis=-1)

            # and the usual cuts
            data = data[
                 (~data.coincident.puls)  # no pulser eventsdata
                & (~data.coincident.muon_offline)  # no muons
                & (~data.geds.quality.is_not_bb_like.is_delayed_discharge)
                & (ak.all(data.geds["quality"][qcs_flag], axis=-1))
            ]
            
            import collections
            # first for real triggers
            unphysical = collections.Counter(ak.flatten(data[~data.trigger.is_forced]["geds"]["quality"]["is_not_bb_like"]["rawid"]))
            make_unphysical_rate_plot(args.r,args.p,unphysical,geds_mapping,is_forced=False,time=runs[period][run]["phy"]["livetime_in_s"])
            



            unphysical = collections.Counter(ak.flatten(data[data.trigger.is_forced]["geds"]["quality"]["is_not_bb_like"]["rawid"]))
            n_forced = ak.num(data[data.trigger.is_forced],axis=0)
            n_forced_pass =  ak.num(data[(data.geds.quality[evt_quality_flag]) & (data.trigger.is_forced)],axis=0)
            logger.info(f"n forced {n_forced} and n forced pass {n_forced_pass}")
            p = n_forced_pass/n_forced
            err = np.sqrt(p*(1-p)/n_forced)
            logger.info(f"Forced trigger survival = {100*n_forced_pass/n_forced} +/- {100*err}")

            
            make_unphysical_rate_plot(args.r,args.p,unphysical,geds_mapping,is_forced=True,norm=n_forced)

            # remove forced triggers
            data=data[(~data.trigger.is_forced) & (data.geds.multiplicity > 0)]


            if (use_qc):
                data = data[(data.geds["quality"][evt_quality_flag])]
            data["pass_psd"]=ak.all(data.geds.psd.is_bb_like, axis=-1)

            for _cut_name, _cut_dict in rconfig["cuts"].items():
                _cut_string = _cut_dict["cut_string"]

                for c_mc, c_data in conversions.items():
                    _cut_string = _cut_string.replace(c_mc, c_data)

                # not a summed spectra
                if _cut_dict["is_sum"] is False:

                    # loop over channels
                    for _channel_id, _name in sorted(geds_mapping.items()):
                        
                        _energy_array = ak.flatten(data[
                            eval(_cut_string, globs, data)
                            & (data["period"] == period)
                            & (data.geds[rawid_name][:, 0] == int(_channel_id[2:]))
                        ]["geds"][energy_name],axis=-1).to_numpy().astype(np.float64)

                        if len(_energy_array) == 0:
                            continue
                        hists[_cut_name][f"{_channel_id}"].FillN(
                            len(_energy_array),_energy_array, np.ones(len(_energy_array)),
                        )


                        # fill also time dependent hists
                        for name_run_hist in run_hist_list:

                            if (name_run_hist==""):
                                run_hists[_cut_name][f"{period}_{run}"].FillN(
                                    len(_energy_array),
                                    _energy_array,
                                    np.ones(len(_energy_array)),
                                )
                            else:
                                if (chmap[geds_mapping[_channel_id]]["type"] in name_run_hist):
                                    run_hists[_cut_name][f"{period}_{run}{name_run_hist}"].FillN(
                                        len(_energy_array),
                                        _energy_array,
                                        np.ones(len(_energy_array)),
                                    )

                elif (
                    "is_2d" not in _cut_dict
                    or _cut_dict["is_2d"] is False
                ):  
                    
                    _summed_energy_array = ak.sum(data[eval(_cut_string, globs, data)]["geds"][
                        energy_name
                    ],axis=-1).to_numpy().astype(np.float64)

                    if len(_summed_energy_array) == 0:
                        continue

                    sum_hists[_cut_name]["all"].FillN(
                        len(_summed_energy_array),
                        _summed_energy_array,
                        np.ones(len(_summed_energy_array)),
                    )

                else:
                    _mult_energy_array = data[
                        eval(_cut_string, globs, data) & (data["period"] == period)
                    ]["geds"][energy_name]
                    _mult_channel_array = data[
                        eval(_cut_string, globs, data) & (data["period"] == period)
                    ]["geds"][rawid_name][ak.argsort(_mult_energy_array, axis=-1)].to_numpy()

                    
                    # apply the category selection

                    _corrected_energy_1 = _mult_energy_array[_mult_energy_array==ak.min(_mult_energy_array,axis=-1)].to_numpy().astype(np.float64)
                    _corrected_energy_2 = _mult_energy_array[_mult_energy_array==ak.max(_mult_energy_array,axis=-1)].to_numpy().astype(np.float64)
                    _summed_energy_array=_corrected_energy_1+_corrected_energy_2

                    # get the categories
                    for name in names_m2:

                        # select the right events
                        if name != "all" and "e1" not in name:
                            categories = get_m2_categories(
                                _mult_channel_array, channel2string, channel2position
                            )
                            string_diff, floor_diff = get_string_row_diff(
                                _mult_channel_array, channel2string, channel2position
                            )

                            if "cat" in name:
                                cat = int(name.split("_")[1])
                                _corrected_energy_1_tmp = np.array(_corrected_energy_1)[
                                    np.where(categories == cat)[0]
                                ]
                                _corrected_energy_2_tmp = np.array(_corrected_energy_2)[
                                    np.where(categories == cat)[0]
                                ]
                                _summed_energy_array_tmp = np.array(_summed_energy_array)[
                                    np.where(categories == cat)[0]
                                ]

                            elif "sd" in name:
                                sd = int(name.split("_")[1])

                                ids = np.where(string_diff == sd)[0]
                                _corrected_energy_1_tmp = np.array(_corrected_energy_1)[ids]
                                _corrected_energy_2_tmp = np.array(_corrected_energy_2)[ids]
                                _summed_energy_array_tmp = np.array(_summed_energy_array)[
                                    ids
                                ]

                        # all case
                        elif "e1" in name:
                            e1_rawid = _mult_channel_array[:, 1]
                            types = channel2type(e1_rawid)
                       
                            ids = np.where(types == name.split("_")[1])
                            _corrected_energy_1_tmp = np.array(_corrected_energy_1)[ids]
                            _corrected_energy_2_tmp = np.array(_corrected_energy_2)[ids]
                            _summed_energy_array_tmp = np.array(_summed_energy_array)[
                                    ids
                                ]
                        else:
                            _corrected_energy_1_tmp = np.array(_corrected_energy_1)
                            _corrected_energy_2_tmp = np.array(_corrected_energy_2)
                            _summed_energy_array_tmp = np.array(_summed_energy_array)

    
                        if len(_summed_energy_array_tmp) == 0:
                            continue

                        _corrected_energy_1_tmp = np.array(_corrected_energy_1_tmp)
                        _corrected_energy_2_tmp = np.array(_corrected_energy_2_tmp)

                    
                        sum_hists[_cut_name][name].FillN(
                            len(_corrected_energy_1_tmp),
                            _corrected_energy_1_tmp,
                            _corrected_energy_2_tmp,
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
    out_file = uproot.recreate("outputs/" + out_name)
    for _cut_name, _hist_dict in hists.items():
        dir = out_file.mkdir(_cut_name)
        for key, item in _hist_dict.items():
            dir[key] = item

    for _cut_name, _hist_dict in run_hists.items():
        for key, item in _hist_dict.items():
            out_file[_cut_name + "/" + key] = item

    for _cut_name, diri in sum_hists.items():
        dir = out_file.mkdir(_cut_name)
        for name, _hist in diri.items():
            dir[name] = _hist

    out_file.close()
    logger.info(f"... done!")


if __name__ == "__main__":
    main()
