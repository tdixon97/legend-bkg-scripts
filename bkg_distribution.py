"""
Script to plot the detectors with their correct shape (check function 'plot_detector_map').
Main Authors: Sofia Calgaro based on a script from Dashboard Team
"""

import math
import os

import numpy as np
from bokeh.palettes import *
from legendmeta import LegendMetadata
from legendmeta.catalog import Props


def is_coax(d):
    return d["type"] == "coax"


def is_taper(f):
    return f != {"angle_in_deg": 0, "height_in_mm": 0} and f != {
        "radius_in_mm": 0,
        "height_in_mm": 0,
    }


def is_bulletized(f):
    return "bulletization" in f and f["bulletization"] != {
        "top_radius_in_mm": 0,
        "bottom_radius_in_mm": 0,
        "borehole_radius_in_mm": 0,
        "contact_radius_in_mm": 0,
    }


def has_groove(f):
    return "groove" in f and f["groove"] != {
        "outer_radius_in_mm": 0,
        "depth_in_mm": 0,
        "width_in_mm": 0,
    }


def has_borehole(f):
    return "borehole" in f and f["borehole"] != {"gap_in_mm": 0, "radius_in_mm": 0}


def plot_geometry(d, R, H):

    coax = is_coax(d)

    g = d["geometry"]

    DH = g["height_in_mm"]
    DR = g["radius_in_mm"]

    xbot = []
    ybot = []

    # botout = g['taper']['bottom']['outer']
    botout = g["taper"]["bottom"]
    if is_taper(botout):
        TH = botout["height_in_mm"]
        TR = (
            botout["radius_in_mm"]
            if "radius_in_mm" in botout
            else TH * math.sin(botout["angle_in_deg"] * math.pi / 180)
        )
        xbot.extend([DR, DR - TR])
        ybot.extend([H - DH + TH, H - DH])
    else:
        xbot.append(DR)
        ybot.append(H - DH)

    if has_groove(g):
        # GR = g['groove']['outer_radius_in_mm']
        GR = g["groove"]["radius_in_mm"]["outer"]
        GH = g["groove"]["depth_in_mm"]
        # GW = g['groove']['width_in_mm']
        GW = g["groove"]["radius_in_mm"]["outer"] - g["groove"]["radius_in_mm"]["inner"]
        xbot.extend([GR, GR, GR - GW, GR - GW])
        ybot.extend([H - DH, H - DH + GH, H - DH + GH, H - DH])

    if coax:
        BG = g["borehole"]["depth_in_mm"]
        BR = g["borehole"]["radius_in_mm"]
        xbot.extend([BR, BR])
        ybot.extend([H - DH, H - DH + BG])

    xtop = []
    ytop = []

    # topout = g['taper']['top']['outer']
    topout = g["taper"]["top"]
    if is_taper(topout):
        TH = topout["height_in_mm"]
        TR = TH * math.sin(topout["angle_in_deg"] * math.pi / 180)
        xtop.extend([DR, DR - TR])
        ytop.extend([H - TH, H])
    else:
        xtop.append(DR)
        ytop.append(H)

    if has_borehole(g) and not coax:
        BG = g["borehole"]["depth_in_mm"]
        BR = g["borehole"]["radius_in_mm"]

        # topin  = g['taper']['top']['inner']
        topin = g["taper"]["top"]
        if is_taper(topin):
            TH = topin["height_in_mm"]
            TR = TH * math.sin(topin["angle_in_deg"] * math.pi / 180)
            xtop.extend([BR + TR, BR, BR])
            ytop.extend([H, H - TH, H - DH + BG])
        else:
            xtop.extend([BR, BR])
            ytop.extend([H, H - DH + BG])

    x = np.hstack(
        (
            [-x + R for x in xbot],
            [x + R for x in xbot[::-1]],
            [x + R for x in xtop],
            [-x + R for x in xtop[::-1]],
        )
    )
    y = np.hstack((ybot, ybot[::-1], ytop, ytop[::-1]))
    return x, y


sort_dict = {
    "String": {
        "out_key": "{key}:{k:02}",
        "primary_key": "location.string",
        "secondary_key": "location.position",
    },
    "CC4": {
        "out_key": "{key}:{k}",
        "primary_key": "electronics.cc4.id",
        "secondary_key": "electronics.cc4.channel",
    },
    "HV": {
        "out_key": "{key}:{k:02}",
        "primary_key": "voltage.card.id",
        "secondary_key": "voltage.channel",
    },
    "Det_Type": {"out_key": "{k}", "primary_key": "type", "secondary_key": "name"},
    "DAQ": {"out_key": None, "primary_key": None, "secondary_key": None},
}


def sorter(path, timestamp, key="String", datatype="cal", spms=False):
    prod_config = os.path.join(path, "config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]

    cfg_file = prod_config["paths"]["metadata"]
    configs = LegendMetadata(path=cfg_file)
    chmap = configs.channelmap(timestamp).map("daq.rawid")

    software_config_path = prod_config["paths"]["config"]
    software_config_db = LegendMetadata(path=software_config_path)
    software_config = software_config_db.on(timestamp, system=datatype).analysis

    out_dict = {}
    # SiPMs sorting
    if spms:
        chmap = chmap.map("system", unique=False)["spms"]
        if key == "Barrel":
            mapping = chmap.map("daq.rawid", unique=False)
            for pos in ["top", "bottom"]:
                for barrel in ["IB", "OB"]:
                    out_dict[f"{barrel}-{pos}"] = [
                        k
                        for k, entry in sorted(mapping.items())
                        if barrel in entry["location"]["fiber"]
                        and pos in entry["location"]["position"]
                    ]
        return out_dict, chmap

    # Daq needs special item as sort on tertiary key
    if key == "DAQ":
        mapping = chmap.map("daq.crate", unique=False)
        for k, entry in sorted(mapping.items()):
            for m, item in sorted(entry.map("daq.card.id", unique=False).items()):
                out_dict[f"DAQ:Cr{k:02},Ch{m:02}"] = [
                    item.map("daq.channel")[pos].daq.rawid
                    for pos in sorted(item.map("daq.channel"))
                    if item.map("daq.channel")[pos].system == "geds"
                ]
    else:
        out_key = sort_dict[key]["out_key"]
        primary_key = sort_dict[key]["primary_key"]
        secondary_key = sort_dict[key]["secondary_key"]
        mapping = chmap.map(primary_key, unique=False)
        for k, entry in sorted(mapping.items()):
            out_dict[out_key.format(key=key, k=k)] = [
                entry.map(secondary_key)[pos].daq.rawid
                for pos in sorted(entry.map(secondary_key))
                if entry.map(secondary_key)[pos].system == "geds"
            ]

    out_dict = {
        entry: out_dict[entry] for entry in list(out_dict) if len(out_dict[entry]) > 0
    }
    return out_dict, software_config, chmap


def plot_detector_map(
    ge_keys,
    data_array,
    strings_dict,
    chdict,
    chmap,
    title="",
    label="data",
    color_map="viridis",
    save_name=None,
):
    """
    Function to plot "data" in each channel (eg rate or number of counts). Ac detectors are showed as well.

    Parameters:
        - ge_keys : list of all (both on, ac and off) germanium channel names
        - data_array: list of data to plot (one entry per channel)
        - strings_dict, chdict, chmap: array-related info;
            can be retrieved with:
                'strings_dict, chdict, chmap = sorter(inputs_path, start_key, key="String")
            where
            1. inputs_path is the global path to the inputs/ folder (eg "/data2/public/prodenv/prod-blind/ref-v1.0.0/inputs")
            2. start_key is the starting key of the time interval of interest (usability mask changes over time)
        - title: string to display as a plot title
        - label: string to display next to the color bar
        - color_map: default=viridis
        - save_name: name of output file (include the extension! eg "k_rates.pdf"); if none, the plot is not saved
    Returns:
        - a figure if save_name is not specified
    """
    xs, ys = [], []
    ax, ay = [], []
    R, H, maxH = 0, 0, 0
    xlabels = dict()
    for name, string in strings_dict.items():
        xlabels[R] = name
        for channel_no in string:
            channel = chmap[channel_no]
            x, y = plot_geometry(channel, R, H)
            H -= channel["geometry"]["height_in_mm"] + 40
            xs.append(x)
            ys.append(y)
            ax.append(R)
            ay.append(H + 10)
        R += 160
        maxH = min(H, maxH)
        H = 0

    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = plt.get_cmap(color_map)
    norm = plt.Normalize(min(data_array), max(data_array))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label=label, ax=ax)
    for i, ge_key in enumerate(ge_keys):
        x, y = list(xs[i]), list(ys[i])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, c="k")
        # fancy way to write over ac detectors 'ac'
        if chdict[ge_key]["usability"] != "off":
            if chdict[ge_key]["usability"] == "ac":
                plt.fill(x, y, color=cmap(norm(data_array[i])))
                center_x = sum(x) / len(x)
                center_y = sum(y) / len(y)
                plt.text(
                    center_x,
                    center_y,
                    "AC",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="r",
                    weight="bold",
                )
            else:
                plt.fill(x, y, color=cmap(norm(data_array[i])))
        """
        # less fancy where we don't mark 'ac' detectors
        if chdict[ge_key]["usability"] != "off":
            plt.fill(x, y, color=cmap(norm(data_array[i])))
        else:
            plt.fill(x, y, color="r")
        """
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.title(title)
    if save_name is not None:
        fig.savefig(save_name, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
