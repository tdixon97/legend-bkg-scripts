"""
Script to load the calibration data for comparison with MC
Main Authors: Sofia Calgaro, Toby Dixon
"""

import json
import logging
import os
from datetime import datetime

import awkward as ak
import numpy as np
import pandas as pd
from legendmeta import LegendMetadata
from lgdo import lh5

# -----------------------------------------------------------
# LOGGER SETTINGS
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
# -----------------------------------------------------------

sto = lh5.LH5Store()


def get_data_awkard(
    cfg: dict,
    file_list_evt: list,
    metadb=LegendMetadata(),
    threshold=500,
    period="p03",
    run="r000",
):
    """
    A function to load the evt tier data into an awkard array also getting the energies
    Parameters
    ----------------------
        - cfg: path to the data for each period
        - periods: list of periods to use
        - n_max (int): used for debugging, if not None (the default) only n_max files will be read
        - run_list (dict): dictonary of run_lists
    Returns

    ----------------------
        - an awkard array of the data

    Example
    ----------------------

    """

    data = None
    for idx, f_evt in enumerate(file_list_evt):

        logger.info(f"{idx} out of {len(file_list_evt)}")
        tier = "pet"
        f_tcm = f_evt.replace(tier, "tcm").replace("v1.0.1", "v1.0.0")
        f_hit = f_evt.replace(tier, "pht")
        d_evt = lh5.read_as("evt", f_evt, library="ak")
        logger.debug("read evt tier file")
        d_evt["geds", "is_is_unphysical_idx"] = d_evt["geds", "is_unphysical_idx"]

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
        tcm_unphysical = ak.Array(
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
        logger.debug("read tcm file")

        # get uniques rawids for loading hit data
        rawids = np.unique(ak.to_numpy(ak.ravel(tcm.id)))
        energy = None
        rawid_sort = None

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
            rawid_unf = ak.full_like(data_unf, rawid, dtype="int")
            energy = (
                data_unf
                if energy is None
                else ak.concatenate((energy, data_unf), axis=-1)
            )
            rawid_sort = (
                rawid_unf
                if rawid_sort is None
                else ak.concatenate((rawid_sort, rawid_unf), axis=-1)
            )
        logger.debug("read energies file")

        d_evt["geds", "unphysical_hit_rawid"] = tcm_unphysical.id
        d_evt["geds", "hit_rawid"] = rawid_sort
        d_evt["geds", "energy"] = energy

        ch = metadb.channelmap(metadb.dataprod.runinfo[period][run]["phy"]["start_key"])

        ac = [
            _dict["daq"]["rawid"]
            for _name, _dict in ch.items()
            if ch[_name]["system"] == "geds"
            and ch[_name]["analysis"]["usability"] in ["ac"]
        ]
        off = [
            _dict["daq"]["rawid"]
            for _name, _dict in ch.items()
            if ch[_name]["system"] == "geds"
            and ch[_name]["analysis"]["usability"] in ["off"]
        ]

        d_evt["geds", "is_good_channel"] = d_evt["geds", "hit_rawid"] > 0
        for a in ac:
            d_evt["geds", "is_good_channel"] = d_evt["geds", "is_good_channel"] & (
                d_evt["geds", "hit_rawid"] != a
            )
        for a in off:
            d_evt["geds", "is_good_channel"] = d_evt["geds", "is_good_channel"] & (
                d_evt["geds", "hit_rawid"] != a
            )

        logger.debug("start filtering")

        d_evt = d_evt[ak.any(d_evt.geds.energy > threshold, axis=-1)]
        d_evt["period"] = period
        d_evt["run"] = run

        data = d_evt if data is None else ak.concatenate((data, d_evt))
        logger.debug("Done reading file")

    # QCs naming convention was switched starting from p10-r001 (included)
    #   - prior: OLD QC -> is_good_hit_old
    #            NEW QC -> is_good_hit
    #   - after: OLD QC -> is_good_hit
    #            NEW QC -> is_good_hit_new

    if "is_good_hit_old" in data["geds"].fields:
        data["geds", "is_good_hit_new"] = data["geds", "is_good_hit"]
        data["geds", "is_good_hit"] = data["geds", "is_good_hit_old"]

    return data


def get_vectorised_converter(mapping):
    def channel2other(channel):
        """Extract which string a given channel is in"""

        return mapping[f"ch{channel}"]

    return np.vectorize(channel2other)


def get_time_unix(time_str: str):

    dt_object = datetime.strptime(time_str, "%Y%m%dT%H%M%SZ")

    return dt_object.timestamp()


def get_file_list(
    tcm_path: str, key_path: str, period: str, run: str, pos: str
) -> list:
    """
    Gets a list of files
    Parameters
    ----------
    tcm_path
        path to the tcm data
    key_path
        path to the folder containing the metadata on cal data
    period
        period in the form pXY
    run
        run in the form rXYZ
    pos
        either pos1 or pos2
    """
    json_file = f"{key_path}/{period}/l200-{period}-{run}-cal-T%-keys.json"

    with open(json_file) as json_file:
        run_info = json.load(json_file)

    edge_keys = run_info["info"]["source_positions"][pos]["keys"]
    tstart = get_time_unix(edge_keys[0].split("-")[-1])
    tstop = get_time_unix(edge_keys[1].split("-")[-1])
    all_keys = sorted(os.listdir(f"{tcm_path}/{period}/{run}/"))
    output = []
    for key in all_keys:

        time = get_time_unix(key.split("-")[-2])
        if time >= tstart and time <= tstop:
            output.append(key)
    return output


def get_tcm_pulser_ids(tcm_file, channel, multiplicity_threshold):
    if isinstance(channel, str):
        if channel[:2] == "ch":
            chan = int(channel[2:])
        else:
            chan = int(channel)
    else:
        chan = channel

    data = pd.DataFrame(
        {
            "array_id": sto.read("hardware_tcm_1/array_id", tcm_file)[0].view_as("np"),
            "array_idx": sto.read("hardware_tcm_1/array_idx", tcm_file)[0].view_as(
                "np"
            ),
        }
    )
    cumulength = sto.read("hardware_tcm_1/cumulative_length", tcm_file)[0].view_as("np")
    cumulength = np.append(np.array([0]), cumulength)
    n_channels = np.diff(cumulength)
    evt_numbers = np.repeat(np.arange(0, len(cumulength) - 1), np.diff(cumulength))
    evt_mult = np.repeat(np.diff(cumulength), np.diff(cumulength))
    data["evt_number"] = evt_numbers
    data["evt_mult"] = evt_mult
    high_mult_events = np.where(n_channels > multiplicity_threshold)[0]  # noqa: F841
    ids = data.query(f"array_id=={chan} and evt_number in @high_mult_events")[
        "array_idx"
    ].to_numpy()
    mask = np.zeros(len(data.query(f"array_id=={chan}")), dtype="bool")
    mask[ids] = True
    return ids, mask


def get_livetime(path: str, file_list: list, delta_t: float):
    """Get the livetime for a certain list of files"""

    sto = lh5.LH5Store()
    time_tot = 0
    for file in file_list:
        ids, mask = get_tcm_pulser_ids(path + file, "ch1027201", 0)
        time = delta_t * len(ids)
        time_tot += time
    return time_tot


def add_livetime_to_json(periods, path, cfg):

    for period in periods:
        runs = os.listdir(path + period)
        for run in runs:

            json_file_path = f"{cfg}/{period}/l200-{period}-{run}-cal-T%-keys.json"

            with open(json_file_path) as json_file:
                run_info = json.load(json_file)
            positions = run_info["info"]["source_positions"].keys()

            for pos in positions:
                list_pos = get_file_list(f"{path}", cfg, period, run, pos)
                livetime = get_livetime(f"{path}/{period}/{run}/", list_pos, 2)
                run_info["info"]["source_positions"][pos]["livetime_in_s"] = livetime
                logger.info(f"livetime for {period} {run} {pos} : {livetime}")
            with open(json_file_path, "w") as json_file:
                json.dump(run_info, json_file, indent=3)


# get the metadata information / mapping
# --------------------------------------
periods = ["p03", "p04", "p06", "p07", "p08"]
prod_cycle = "/data2/public/prodenv/prod-blind/ref-v1.0.1/"
out_name = "l200a-cal-dataset.root"
logger.info("... get the metadata information / mapping")
metadb = LegendMetadata(prod_cycle + "inputs/")

chmap = metadb.channelmap(datetime.now())

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
channel2string = get_vectorised_converter(geds_strings)
channel2position = get_vectorised_converter(geds_positions)

runs = metadb.dataprod.config.analysis_runs

compute_livetime = False
cfg = "cfg/DataKeys/cal/"
if compute_livetime is True:
    add_livetime_to_json(periods, f"{prod_cycle}/generated/tier/tcm/cal/", cfg)


prod_cycle_tcm = prod_cycle.replace("v1.0.1", "v1.0.0")

paths_cfg = {
    "p03": {
        "tier": "pet",
        "evt_path": f"{prod_cycle}/generated/tier/",
        "tcm_path": f"{prod_cycle_tcm}/generated/tier/",
    },
    "p04": {
        "tier": "pet",
        "evt_path": f"{prod_cycle}/generated/tier/",
        "tcm_path": f"{prod_cycle_tcm}/generated/tier/",
    },
    "p06": {
        "tier": "pet",
        "evt_path": f"{prod_cycle}/generated/tier/",
        "tcm_path": f"{prod_cycle_tcm}/generated/tier/",
    },
    "p07": {
        "tier": "pet",
        "evt_path": f"{prod_cycle}/generated/tier/",
        "tcm_path": f"{prod_cycle_tcm}/generated/tier/",
    },
    "p08": {
        "tier": "pet",
        "evt_path": f"{prod_cycle}/generated/tier/",
        "tcm_path": f"{prod_cycle_tcm}/generated/tier/",
    },
}
runs = metadb.dataprod.config.analysis_runs


os.makedirs("outputs/cal/", exist_ok=True)
output_cache = f"outputs/cal/{out_name.replace('.root', '.parquet')}"
data_tot = None
process_evt = False

# loop over periods, runs and positions
for p in periods:
    for r in runs[p]:

        # extract the json file
        json_file_path = f"{cfg}/{p}/l200-{p}-{r}-cal-T%-keys.json"
        with open(json_file_path) as json_file:
            run_info = json.load(json_file)

        positions = run_info["info"]["source_positions"].keys()

        for pos in positions:

            # get file lists
            path = f"{prod_cycle_tcm}/generated/tier/tcm/cal/"
            list_pos = get_file_list(f"{path}", cfg, p, r, pos)

            list_evt = [
                f"{prod_cycle}/generated/tier/pet/cal/{p}/{r}/"
                + tf.replace("tcm", "pet").replace("v1.0.0", "v1.0.1")
                for tf in list_pos
            ]

            n = out_name.split(".")[-2]

            # output file for parqeut
            output_cache = f"outputs/{n}-{p}-{r}-{pos}.parquet"
            logger.info(f"{p},{r}, {pos}")

            if os.path.exists(output_cache) and process_evt is False:
                logger.info("Get from parquet")
                data = ak.from_parquet(output_cache)
            else:
                data = get_data_awkard(
                    cfg=paths_cfg,
                    file_list_evt=list_evt,
                    metadb=metadb,
                    period=p,
                    run=r,
                )
                ak.to_parquet(data, output_cache)
                data = None
