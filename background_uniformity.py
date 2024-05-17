# Background uniformity in the ROI
# LEGEND data: p03/p04 (taup dataset)
# 01 August 2023, Elisabetta Bossio

import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from legendmeta import LegendMetadata


def plot_uniformity(
    total_bi,
    i_cts,
    i_exp,
    i_ids,
    ylabel="Counts [cts/kg/yr]",
    figsize=(9, 7),
    title="a title",
):

    expected_cts = np.multiply(i_exp, total_bi)
    one_sigma = stats.poisson.interval(0.68, expected_cts)
    two_sigma = stats.poisson.interval(0.95, expected_cts)
    three_sigma = stats.poisson.interval(0.999, expected_cts)

    fig = plt.figure(figsize=figsize)
    i_x = np.arange(0, len(i_cts), 1)
    plt.title(title)
    plt.fill_between(
        i_x,
        three_sigma[0] / i_exp,
        three_sigma[1] / i_exp,
        color="coral",
        label="99.9%",
    )
    plt.fill_between(
        i_x, two_sigma[0] / i_exp, two_sigma[1] / i_exp, color="yellow", label="95%"
    )
    plt.fill_between(
        i_x, one_sigma[0] / i_exp, one_sigma[1] / i_exp, color="lime", label="68%"
    )
    plt.errorbar(
        i_x,
        i_cts / i_exp,
        np.sqrt(i_cts) / i_exp,
        linestyle="None",
        marker="o",
        color="black",
        label="Data",
    )
    plt.plot(i_x, expected_cts / i_exp, color="black", label="Expectation")
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.set_xticks(i_x)
    ax.set_xticklabels(i_ids, rotation=90)
    plt.legend()
    plt.tight_layout()
    return fig


# define test
def test(n_vector, mu_vector):
    test_num = np.sum(stats.poisson.logpmf(k=n_vector, mu=mu_vector))
    test_den = np.sum(stats.poisson.logpmf(k=n_vector, mu=n_vector))
    return -2 * test_num + 2 * test_den


def run_test(expected_cts, observed_cts, plotting=True):
    # define structures for toy MC
    mu_vector = expected_cts
    test_statistic_values = []

    # run toy MC
    for i in range(0, 100000):
        n_vector = np.random.poisson(mu_vector)
        test_statistic_values.append(test(n_vector=n_vector, mu_vector=mu_vector))

    # compute test value for observed dataset
    dataset_test = test(observed_cts, mu_vector)

    # test statistic distribution
    hh, bins = np.histogram(
        test_statistic_values, range=(0, 300), bins=1000, density=True
    )

    # observed pvalue
    cum = 1 - np.cumsum(hh * np.diff(bins))  # multiply for bin size

    if plotting == True:
        # plot hist with
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all")
        ax1.set_ylabel("pdf")
        ax1.grid(True)
        ax1.stairs(hh, bins)
        ax1.axvline(x=dataset_test)
        ax1.set_xlim(bins[0], bins[-1])

        ax2.set_xlabel("test")
        ax2.set_ylabel("cdf")
        ax2.grid(True)
        ax2.stairs(cum, bins)
        ax2.axvline(x=dataset_test)
        ax2.set_xlim(bins[0], bins[-1])
        plt.tight_layout()

    try:
        pvalue = cum[np.where(bins >= dataset_test)[0][0]]
        return pvalue
    except:
        return 0


def prepare_grups(fdata, roi_df, channels, dataset="silver"):
    # grouping by detector
    det_exp = np.zeros(0)
    det_cts = np.zeros(0)
    det_chn = np.zeros(0)
    # grouping by detector type
    typ_ids = np.array(["bege", "icpc", "coax", "ppc"])
    if dataset == "golden":
        typ_ids = np.array(["bege", "icpc"])
    typ_exp = np.zeros(len(typ_ids))
    typ_cts = np.zeros(len(typ_ids))
    # grouping by string
    str_ids = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
    str_exp = np.zeros(len(str_ids))
    str_cts = np.zeros(len(str_ids))
    # grouping by position in the array
    pos_det = [
        [
            "V02160A",
            "B00035C",
            "C000RG1",
            "B00032B",
            "B00091C",
            "B00000B",
            "B00000C",
            "V08682B",
            "V02162B",
            "B00089C",
            "B00089D",
            "B00002A",
            "B00089A",
            "B00000D",
            "B00002C",
            "B00000A",
            "B00002B",
        ],
        [
            "V02160B",
            "V05261B",
            "C000RG2",
            "P00574B",
            "P00665A",
            "B00061A",
            "B00061C",
            "B00076C",
            "V08682A",
            "V02166B",
            "C00ANG4",
            "B00091A",
            "B00091D",
            "P00537A",
            "B00032C",
            "B00032D",
            "B00035A",
            "B00032A",
            "P00538B",
        ],
        [
            "V05266A",
            "V05266B",
            "C00ANG3",
            "C00ANG5",
            "P00698A",
            "P00712A",
            "P00909C",
            "B00079B",
            "B00079C",
            "V01386A",
            "V09372A",
            "V09374A",
            "V09724A",
            "V04199A",
            "V04545A",
            "V00048B",
            "V00050A",
            "V00050B",
            "P00538A",
            "P00573A",
            "P00661C",
            "B00035B",
            "B00061B",
            "B00089B",
            "V00048A",
            "P00573B",
            "P00575A",
            "P00574C",
            "P00661A",
        ],
        [
            "V05268B",
            "V05612A",
            "C00ANG2",
            "V00074A",
            "P00661B",
            "P00574A",
            "V01403A",
            "V01404A",
            "V01406A",
            "V05267B",
            "V07646A",
            "V01387A",
            "V05261A",
            "P00662C",
            "P00664A",
            "P00665C",
            "V01240A",
            "V01389A",
            "P00662A",
            "P00662B",
            "P00665B",
        ],
        [
            "V07647A",
            "V07647B",
            "V04549A",
            "V07298B",
            "V01415A",
            "V07302B",
            "V05268A",
            "V07302A",
            "P00748B",
            "P00909B",
            "B00091B",
            "V05267A",
            "V05612B",
            "P00748A",
            "P00698B",
        ],
    ]
    pos_ids = np.array(["top", "midtop", "middle", "midbottom", "bottom"])
    pos_exp = np.zeros(len(pos_ids))
    pos_cts = np.zeros(len(pos_ids))
    if dataset == "silver":
        exposure = "exposure-silver"
    elif dataset == "golden":
        exposure = "exposure-golden"
    else:
        print("Specify a valide dataset: silver or golden")

    # filling arrays
    for ch_name in fdata.keys():
        if fdata[ch_name][exposure] == 0:
            continue
        det_chn = np.append(det_chn, ch_name)
        det_exp = np.append(det_exp, fdata[ch_name][exposure])
        det_cts = np.append(
            det_cts,
            len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            ),
        )
        if ch_name[0] == "B":
            typ_exp[0] += fdata[ch_name][exposure]
            typ_cts[0] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        if ch_name[0] == "V":
            typ_exp[1] += fdata[ch_name][exposure]
            typ_cts[1] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        if dataset == "silver":
            if ch_name[0] == "C":
                typ_exp[2] += fdata[ch_name][exposure]
                typ_cts[2] += len(
                    roi_df.loc[
                        roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                    ]
                )
            if ch_name[0] == "P":
                typ_exp[3] += fdata[ch_name][exposure]
                typ_cts[3] += len(
                    roi_df.loc[
                        roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                    ]
                )
        for i in str_ids:
            idx = np.where(str_ids == i)
            if channels.map("name")[ch_name].location.string == i:
                str_exp[idx] += fdata[ch_name][exposure]
                str_cts[idx] += len(
                    roi_df.loc[
                        roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                    ]
                )
        if ch_name in pos_det[0]:
            pos_exp[0] += fdata[ch_name][exposure]
            pos_cts[0] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        elif ch_name in pos_det[1]:
            pos_exp[1] += fdata[ch_name][exposure]
            pos_cts[1] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        elif ch_name in pos_det[2]:
            pos_exp[2] += fdata[ch_name][exposure]
            pos_cts[2] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        elif ch_name in pos_det[3]:
            pos_exp[3] += fdata[ch_name][exposure]
            pos_cts[3] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        elif ch_name in pos_det[4]:
            pos_exp[4] += fdata[ch_name][exposure]
            pos_cts[4] += len(
                roi_df.loc[
                    roi_df["channel_id"] == channels.map("name")[ch_name].daq.rawid
                ]
            )
        else:
            print("WARNING! ", ch_name, "not included!")

    if (
        (int(np.sum(det_exp) * 1000) != int(np.sum(typ_exp) * 1000))
        | (int(np.sum(det_exp) * 1000) != int(np.sum(str_exp) * 1000))
        | (int(np.sum(det_exp) * 1000) != int(np.sum(pos_exp) * 1000))
    ):
        print("WARNING! Something wrong with the exposures!")
    if (
        (np.sum(det_cts) != np.sum(typ_cts))
        | (np.sum(det_cts) != np.sum(str_cts))
        | (np.sum(det_cts) != np.sum(pos_cts))
    ):
        print("WARNING! Something wrong with the total number of counts!")
    return (
        det_chn,
        det_exp,
        det_cts,
        typ_ids,
        typ_exp,
        typ_cts,
        str_ids,
        str_exp,
        str_cts,
        pos_ids,
        pos_exp,
        pos_cts,
    )


def run_all_tests(fdata, df, channels, dataset="silver", cuts="none", what_test=""):

    if what_test not in ["roi", "k42", "k40", "tl208"]:
        print("WARNING: select one test among roi, k42, k40, or tl208!")
        exit()

    # Select only ROI
    roi_df = df.copy()
    if what_test == "roi":
        roi_df = df.loc[
            ((df["energy"] > 1930) & (df["energy"] < 2099))
            | ((df["energy"] > 2109) & (df["energy"] < 2114))
            | ((df["energy"] > 2124) & (df["energy"] < 2190))
        ]
        ylabel = "Counts in the analysis window [cts/kg/yr]"
        filename = "uniformity_roi_"
        print("Testing the 0vbb analysis window")
    elif what_test == "k42":
        roi_df = df.loc[((df["energy"] > 1520) & (df["energy"] < 1530))]
        ylabel = "Counts in the 1525 keV line [cts/kg/yr]"
        filename = "uniformity_k42line_"
        print("Testing the K42 gamma line at 1525 keV")
    elif what_test == "k40":
        roi_df = df.loc[((df["energy"] > 1455) & (df["energy"] < 1465))]
        ylabel = "Counts in the 1460 keV line [cts/kg/yr]"
        filename = "uniformity_k40line_"
        print("Testing the K40 gamma line at 1460 keV")
    elif what_test == "tl208":
        roi_df = df.loc[((df["energy"] > 2610) & (df["energy"] < 2620))]
        ylabel = "Counts in the 2615 keV line [cts/kg/yr]"
        filename = "uniformity_tl208line_"
        print("Testing the Tl208 gamma line at 2615 keV")

    if cuts == "lar":
        roi_df = roi_df.loc[roi_df["is_lar_rejected"] == False]
        filename += "afterLAr_"
    elif cuts == "psd":
        roi_df = roi_df.loc[
            (roi_df["is_aoe_tagged"] == False) & (roi_df["is_usable_aoe"] == True)
        ]
        filename += "afterPSD_"
    elif cuts == "none":
        filename += "beforecuts_"

    # prepare datasets
    (
        det_chn,
        det_exp,
        det_cts,
        typ_ids,
        typ_exp,
        typ_cts,
        str_ids,
        str_exp,
        str_cts,
        pos_ids,
        pos_exp,
        pos_cts,
    ) = prepare_grups(fdata, roi_df, channels, dataset)
    total_exp = np.sum(det_exp)
    total_cts = np.sum(det_cts)
    total_bi = total_cts / total_exp
    if dataset == "silver":
        if cuts == "none":
            title = "Silver dataset - before analysis cuts"
        elif cuts == "lar":
            title = "Silver dataset - after LAr veto cut"
    elif dataset == "golden":
        if cuts == "none":
            title = "Golden dataset - before analysis cuts"
        elif cuts == "lar":
            title = "Golden dataset - after LAr veto cut"
        elif cuts == "psd":
            title = "Golden dataset - after PSD cut"

    print("*****************************************************")
    print(title)
    print("total counts = ", total_cts)
    print("total exposure = ", total_exp, " kg yr")
    print("mean counts rate = ", total_bi, " cts/kg/yr")
    print("*****************************************************")
    # Plot counts observed and expected
    fig1 = plot_uniformity(
        total_bi, det_cts, det_exp, det_chn, ylabel, figsize=(14, 7), title=title
    )
    fig2 = plot_uniformity(
        total_bi, typ_cts, typ_exp, typ_ids, ylabel, figsize=(9, 7), title=title
    )
    fig3 = plot_uniformity(
        total_bi, str_cts, str_exp, str_ids, ylabel, figsize=(9, 7), title=title
    )
    fig4 = plot_uniformity(
        total_bi, pos_cts, pos_exp, pos_ids, ylabel, figsize=(9, 7), title=title
    )
    fig1.savefig("plots/" + filename + "bydet.png", bbox_inches="tight")
    fig2.savefig("plots/" + filename + "bytype.png", bbox_inches="tight")
    fig3.savefig("plots/" + filename + "bystring.png", bbox_inches="tight")
    fig4.savefig("plots/" + filename + "bypos.png", bbox_inches="tight")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    # Test uniformity
    det_expexted = np.multiply(det_exp, total_bi)
    det_pval = run_test(det_expexted, det_cts, plotting=False)
    print("Testing uniformity of the background among detectors - p-value = ", det_pval)
    typ_expected = np.multiply(typ_exp, total_bi)
    typ_pval = run_test(typ_expected, typ_cts, plotting=False)
    print(
        "Testing uniformity of the background among detector types - p-value = ",
        typ_pval,
    )
    str_expected = np.multiply(str_exp, total_bi)
    str_pval = run_test(str_expected, str_cts, plotting=False)
    print("Testing uniformity of the background among strings - p-value = ", str_pval)
    pos_expexted = np.multiply(pos_exp, total_bi)
    pos_pval = run_test(pos_expexted, pos_cts, plotting=False)
    print(
        "Testing uniformity of the background among position in the array - p-value = ",
        pos_pval,
    )
    print("*****************************************************")
    return 1


def main():

    # Load all LEGEND data (from Patrick's skimmed files) into a dataframe
    df_r0 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p03/p03_r000_high_level.hdf"
    )
    df_r1 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p03/p03_r001_high_level.hdf"
    )
    df_r2 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p03/p03_r002_high_level.hdf"
    )
    df_r3 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p03/p03_r003_high_level.hdf"
    )
    df_r4 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p03/p03_r004_high_level.hdf"
    )
    df_r5 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p03/p03_r005_high_level.hdf"
    )
    df_4r0 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p04/p04_r000_high_level.hdf"
    )
    df_4r1 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p04/p04_r001_high_level.hdf"
    )
    df_4r2 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p04/p04_r002_high_level.hdf"
    )
    df_4r3 = pd.read_hdf(
        "/mnt/atlas01/users/krause/share/data/l200/high_lvl/v01_06/p04/p04_r003_high_level.hdf"
    )
    df = pd.concat(
        [df_r0, df_r1, df_r2, df_r3, df_r4, df_r5, df_4r0, df_4r1, df_4r2, df_4r3],
        verify_integrity=True,
    )
    # Apply basic cuts (multiplicity, quality cuts, muon veto)
    df = df.loc[
        (df["multiplicity"] == 1)
        & (df["is_physical"] == True)
        & (df["is_valid_channel"] == True)
    ]
    df = df.loc[
        (df["is_muon_tagged"] == False)
        & (df["is_baseline"] == False)
        & (df["is_pulser"] == False)
        & (df["is_saturated"] == False)
    ]
    # retrieve LEGEND metadata
    first_key = datetime.datetime.fromtimestamp(df.iloc[0].name)
    lmeta = LegendMetadata("/mnt/atlas01/users/krause/share/meta/l200/metadata090823")
    chmap = lmeta.channelmap(on=first_key)  # get the channel map
    channels = chmap.map("system", unique=False).geds  # select HPGe channels

    fdata = json.load(
        open("/mnt/atlas01/users/bossioel/legend/taup-ana/det-data_taup-dataset.json")
    )
    # fig settings
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["font.size"] = 14

    run_all_tests(fdata, df, channels, dataset="silver", cuts="none", what_test="k42")
    run_all_tests(fdata, df, channels, dataset="silver", cuts="lar", what_test="k42")
    run_all_tests(fdata, df, channels, dataset="silver", cuts="none", what_test="k40")
    run_all_tests(fdata, df, channels, dataset="silver", cuts="lar", what_test="k40")
    run_all_tests(fdata, df, channels, dataset="silver", cuts="none", what_test="tl208")
    run_all_tests(fdata, df, channels, dataset="silver", cuts="none", what_test="roi")
    run_all_tests(fdata, df, channels, dataset="silver", cuts="lar", what_test="roi")
    run_all_tests(fdata, df, channels, dataset="golden", cuts="none", what_test="roi")
    run_all_tests(fdata, df, channels, dataset="golden", cuts="lar", what_test="roi")
    run_all_tests(fdata, df, channels, dataset="golden", cuts="psd", what_test="roi")


if __name__ == "__main__":
    main()
