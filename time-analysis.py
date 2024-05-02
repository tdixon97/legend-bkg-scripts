"""
time-analysis.py
Authors: Toby Dixon and Sofia Calgaro
"""

import argparse
import copy
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tol_colors as tc
import uproot
from hist import Hist
from legend_plot_style import LEGENDPlotStyle as lps
from legendmeta import LegendMetadata
from matplotlib.backends.backend_pdf import PdfPages

import utils

lps.use("legend")
vset = tc.tol_cset("vibrant")
mset = tc.tol_cset("muted")
plt.rc("axes", prop_cycle=plt.cycler("color", list(vset)))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
stream_handler.setFormatter(formatter)


def expo(t, A, B, T):

    return B + A * np.exp(-t * np.log(2) / T)


style = {
    "yerr": False,
    "flow": None,
    "lw": 0.6,
}


# looad the arguments
parser = argparse.ArgumentParser(
    description="Script to plot the time dependence of counting rates in L200"
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="Name of output root file, eg l200a-vancouver_full_alpha will be appended with energy range",
    default="l200a-vancouver_full_alpha",
)
parser.add_argument(
    "--input",
    "-i",
    type=str,
    help="Name of input root file",
    default="outputs/l200a-vancouver_full-dataset-v0.0.0.root",
)
parser.add_argument(
    "--input_p10", "-I", type=str, help="Name of input root file for p10", default=None
)
parser.add_argument(
    "--energy",
    "-e",
    type=str,
    help="comma seperate energy range should be of the form (a,b),(c,d) etc",
    default="4000,6000",
)
parser.add_argument(
    "--plot_hist", "-p", type=bool, help="Boolean flag to plot data as histogram "
)
parser.add_argument(
    "--spectrum", "-s", type=str, help="Spectrum to fit", default="mul_surv"
)
parser.add_argument(
    "--BAT_overlay",
    "-B",
    type=str,
    help="Overlay the BAT fit? argument is the directory with BAT fit results",
    default=None,
)
parser.add_argument(
    "--average",
    "-a",
    type=bool,
    help="Boolean flag to show the average rate over the plot",
    default=False,
)
parser.add_argument(
    "--subtract",
    "-m",
    type=bool,
    help="Boolean flag to subtract sidebands (actually a Bayesian counting analysis)",
    default=False,
)

args = parser.parse_args()
output = args.output
input = args.input
input_p10 = args.input_p10
subtract = args.subtract
plot_hist = bool(args.plot_hist)
spectrum = args.spectrum
overlay = args.BAT_overlay
average = args.average

include_p10 = True

if input_p10 is None:
    include_p10 = False

energy = args.energy

energy_list = []
# just two numbers
if "(" not in energy:
    energy_low = float(energy.split(",")[0])
    energy_high = float(energy.split(",")[1])
    energy_list.append((energy_low, energy_high))
# a list
else:
    energy_split = energy.split("(")
    for e in energy_split:
        if "," not in e:
            continue
        else:
            energy = e.split(")")[0]
            energy_low = float(energy.split(",")[0])
            energy_high = float(energy.split(",")[1])
            energy_list.append((energy_low, energy_high))

if len(energy_list) > 1 and subtract:
    raise ValueError("cant specifiy a split range for the counting analysis ")

# load the meta-data
# ----------------------


metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")

chmap = metadb.channelmap(datetime.now())
runs = metadb.dataprod.config.analysis_runs

# hardcoded for now
if include_p10:
    runs["p10"] = ["r000", "r001", "r002", "r003"]

run_times = utils.get_run_times(metadb, runs, verbose=1)

# load the data
e_str = ""
for energy_low, energy_high in energy_list:
    e_str += f"{int(energy_low)}_{int(energy_high)}_"
out_name = f"{output}_{e_str}.root"

hists = {}
hists_p10 = {}
hist_tot = None
hist_tot_p10 = None

# extract the data and start / stop times
# ----------------------------------------

periods = ["p03", "p04", "p06", "p07", "p08"]
if include_p10:
    periods.append("p10")

bins = []

with uproot.open(input) as f:

    for period in periods:
        run_list = runs[period]
        hists[period] = {}

        for run in run_list:
            if run not in run_times[period]:
                continue
            tstart, tstop, mass = run_times[period][run]
            bins.append(tstart / 60 / 60 / 24)
            bins.append(tstop / 60 / 60 / 24)

            if (
                period != "p10" and f"{spectrum}/{period}_{run};1" in f.keys()
            ) and mass > 0:
                hists[period][run] = utils.get_hist(
                    f[f"{spectrum}/{period}_{run}"], (0, 6000), 1
                )

                if hist_tot is None:
                    hist_tot = copy.deepcopy(hists[period][run])
                else:
                    hist_tot += hists[period][run]

# now also p10
if include_p10:
    with uproot.open(input_p10) as f:

        for period in periods:
            hists_p10[period] = {}

            for run in run_list:
                if run not in run_times[period]:
                    continue

                if (
                    period == "p10" and f"{spectrum}/{period}_{run};1" in f.keys()
                ) and mass > 0:
                    hists_p10[period][run] = utils.get_hist(
                        f[f"{spectrum}/{period}_{run}"], (0, 6000), 1
                    )

                    if hist_tot_p10 is None:
                        hist_tot_p10 = copy.deepcopy(hists_p10[period][run])
                    else:
                        hist_tot_p10 += hists_p10[period][run]


# get counts in each run
# -------------------------------------------------

counts = {}
counts_p10 = {}


if subtract:
    pdf = PdfPages("outputs/" + out_name[0:-5] + "_counting_analysis.pdf")


# get the counts in each period
# -----------------------------

for period in periods:
    run_list = runs[period]

    if period != "p10":
        counts[period] = {}
    elif include_p10:
        counts_p10[period] = {}

    for run in run_list:
        if run in hists[period].keys():
            if subtract is False:
                c_tmp = utils.integrate_hist(hists[period][run], energy_list)
                counts[period][run] = (
                    c_tmp,
                    utils.get_error_bar(c_tmp)[0],
                    utils.get_error_bar(c_tmp)[1],
                )
            else:
                counts[period][run], _ = utils.sideband_counting(
                    hists[period][run],
                    energy_low - 15,
                    energy_low,
                    energy_high,
                    energy_high + 15,
                    pdf,
                    f" {energy_low} to {energy_high} keV {period}-{run}",
                )

        if include_p10:
            if run in hists_p10[period].keys():
                if subtract is False:
                    c_tmp = utils.integrate_hist(hists_p10[period][run], energy_list)
                    counts_p10[period][run] = (
                        c_tmp,
                        utils.get_error_bar(c_tmp)[0],
                        utils.get_error_bar(c_tmp)[1],
                    )
                else:
                    counts_p10[period][run], _ = utils.sideband_counting(
                        hists_p10[period][run],
                        energy_low - 15,
                        energy_low,
                        energy_high,
                        energy_high + 15,
                        pdf,
                        f" {energy_low} to {energy_high} keV {period}-{run}",
                    )


# and for the total
if subtract is False:
    c_tmp = utils.integrate_hist(hist_tot, energy_list)
    error = utils.get_error_bar(c_tmp)
    counts_total = (c_tmp, error[0], error[1])
    posterior = error[2]
    c_tmp = utils.integrate_hist(hist_tot_p10, energy_list)
    error_p10 = utils.get_error_bar(c_tmp)
    counts_total_p10 = (c_tmp, error_p10[0], error_p10[1])
    posterior_p10 = error_p10[2]
else:
    counts_total_p10, posterior_p10 = utils.sideband_counting(
        hist_tot_p10,
        energy_low - 15,
        energy_low,
        energy_high,
        energy_high + 15,
        pdf,
        f" {energy_low} to {energy_high} keV p10-TOTAL",
    )
    counts_total, posterior = utils.sideband_counting(
        hist_tot,
        energy_low - 15,
        energy_low,
        energy_high,
        energy_high + 15,
        pdf,
        f" {energy_low} to {energy_high} keV p3-p8-TOTAL",
    )


if subtract:
    pdf.close()

# fill the histograms and arrays
# -------------------------------------------------

x = []
y = []
ey_low = []
ey_high = []
if include_p10:
    x_p10 = []
    y_p10 = []
    ey_low_p10 = []
    ey_high_p10 = []


histo_time = Hist.new.Variable(bins).Double()
histo_mass = Hist.new.Variable(bins).Double()

if include_p10:
    histo_time_p10 = Hist.new.Variable(bins).Double()
    histo_mass_p10 = Hist.new.Variable(bins).Double()

total_exposure = 0
total_exposure_p10 = 0


for period in periods:
    run_list = runs[period]
    for run in run_list:
        if run not in run_times[period]:
            continue
        tstart, tstop, mass = run_times[period][run]

        # fill histo and arrays
        if period != "p10":
            histo_mass[(tstart / 60 / 60 / 24 + tstop / 60 / 60 / 24) * 0.5j] = mass
            histo_time[(tstart / 60 / 60 / 24 + tstop / 60 / 60 / 24) * 0.5j] = counts[
                period
            ][run][0]

            norm = (tstop - tstart) / (60 * 60 * 24)

            if norm > 1:
                x.append((tstart / 60 / 60 / 24 + tstop / 60 / 60 / 24) * 0.5)
                y.append(counts[period][run][0] / (norm * mass))
                ey_low.append(counts[period][run][1] / (norm * mass))
                ey_high.append(counts[period][run][2] / (norm * mass))

            total_exposure += (tstop / 60 / 60 / 24 - tstart / 60 / 60 / 24) * mass

        # same for p10
        elif include_p10:
            histo_mass_p10[(tstart / 60 / 60 / 24 + tstop / 60 / 60 / 24) * 0.5j] = mass
            histo_time_p10[(tstart / 60 / 60 / 24 + tstop / 60 / 60 / 24) * 0.5j] = (
                counts_p10[period][run][0]
            )
            total_exposure_p10 += (tstop / 60 / 60 / 24 - tstart / 60 / 60 / 24) * mass
            norm = (tstop - tstart) / (60 * 60 * 24)

            if norm > 1:
                x_p10.append((tstart / 60 / 60 / 24 + tstop / 60 / 60 / 24) * 0.5)
                y_p10.append(counts_p10[period][run][0] / (norm * mass))
                ey_low_p10.append(counts_p10[period][run][1] / (norm * mass))
                ey_high_p10.append(counts_p10[period][run][2] / (norm * mass))

# save the time-histo (for fitting)
with uproot.recreate("outputs/" + out_name) as output_file:
    output_file["counts"] = histo_time
    output_file["mass"] = histo_mass
    if include_p10:
        output_file["counts_p10"] = histo_time_p10
        output_file["mass_p10"] = histo_mass_p10

if plot_hist is True:
    histo_time_plot = utils.normalise_histo(histo_time)
else:
    histo_time_plot = copy.deepcopy(histo_time)

if include_p10:
    if plot_hist is True:
        histo_time_plot_p10 = utils.normalise_histo(histo_time_p10)
    else:
        histo_time_plot_p10 = copy.deepcopy(histo_time_p10)


# Normalise the histos and save graph (errorbar) for plotting
# ----------------------------------------------------------


widths = np.diff(histo_mass.axes.edges[0])
centers = histo_mass.axes.edges[0]


# normalise (for plots)
for i in range(histo_mass.size - 2):
    if histo_mass[i] > 0 and widths[i] > 1:

        if plot_hist:
            histo_time_plot[i] /= histo_mass[i]

    else:
        histo_time_plot[i] = 0


if include_p10:
    for i in range(histo_mass_p10.size - 2):
        if histo_mass_p10[i] > 0 and widths[i] > 1:

            if plot_hist:
                histo_time_plot_p10[i] /= histo_mass_p10[i]

        else:
            histo_time_plot_p10[i] = 0


# Make plots
# --------------------------------------------------------

fig, axes_full = lps.subplots(1, 1, figsize=(4, 3), sharex=True)

# set y axis limits

if plot_hist:
    histo_time_plot.plot(
        ax=axes_full,
        **style,
        color=vset.orange,
        histtype="fill",
        alpha=0.5,
        label="With OB",
    )
    if include_p10:
        histo_time_plot_p10.plot(
            ax=axes_full,
            **style,
            color=vset.blue,
            histtype="fill",
            alpha=0.5,
            label="NO OB",
        )

else:
    axes_full.errorbar(
        x=x,
        y=y,
        yerr=[np.abs(ey_low), np.abs(ey_high)],
        color=vset.red,
        fmt="o",
        ecolor=vset.orange,
        label="With OB",
    )
    if include_p10:
        axes_full.errorbar(
            x=x_p10,
            y=y_p10,
            yerr=[np.abs(ey_low_p10), np.abs(ey_high_p10)],
            color=vset.blue,
            fmt="o",
            ecolor="grey",
            label="No OB",
        )

axes_full.set_xlabel("Time [days]")
axes_full.set_ylabel("Counts / kg -day")
if include_p10:
    axes_full.legend(loc="best")
axes_full.set_title(f"{e_str} keV")


# set x ranges for bands
end_p8 = x[-1]
start_p10 = x_p10[0]
end_p10 = x_p10[-1]
axes_full.set_xlim(-2, end_p10 + 20)
middle = (start_p10 + end_p8) / 2
max_x = axes_full.get_xlim()[1]

# overlay average band
if average is True:

    low_rate = (counts_total[0] - counts_total[1]) / total_exposure
    high_rate = (counts_total[0] + counts_total[2]) / total_exposure

    axes_full.axhspan(
        low_rate, high_rate, xmin=0, xmax=(middle) / max_x, color=vset.orange, alpha=0.5
    )

    if include_p10:
        low_rate_p10 = (counts_total_p10[0] - counts_total_p10[1]) / total_exposure_p10
        high_rate_p10 = (counts_total_p10[0] + counts_total_p10[2]) / total_exposure_p10

        axes_full.axhspan(
            low_rate_p10,
            high_rate_p10,
            xmin=(middle) / max_x,
            xmax=(end_p10 + 10) / max_x,
            color=vset.blue,
            alpha=0.5,
        )

# overlay BAT fit results
range_x = axes_full.get_xlim()[1]
if overlay is not None:
    with uproot.open(overlay + "/analysis.root") as file:
        tree = file["fit_par_results"]
        branches = tree.keys()
        data = {}
        df_res = pd.DataFrame()
        for branch_name in branches:
            data[branch_name] = tree[branch_name].array()

        df_res = pd.DataFrame(data)


if overlay is not None:
    t = np.linspace(0, range_x, 10000)
    N = expo(t, df_res["glob_mode"][0], df_res["glob_mode"][1], df_res["glob_mode"][2])

    plt.plot(t, N, label="Global mode")
    plt.legend(loc="best")

plt.savefig("outputs/" + out_name[0:-5] + ".pdf")


# make some posterior plots / extract some numbers
# ------------------------------------------------

counts_samples = utils.sample_hist(posterior, N=int(1e6))
counts_p10_samples = utils.sample_hist(posterior_p10, N=int(1e6))

rates = counts_samples / total_exposure
rates_p10 = counts_p10_samples / total_exposure_p10

div = rates_p10 / rates

histo_div = Hist.new.Reg(2000, 0, 1.5).Double()
histo = Hist.new.Reg(2000, 0, max(max(rates), max(rates_p10))).Double()
histo_p10 = Hist.new.Reg(2000, 0, max(max(rates), max(rates_p10))).Double()

histo.fill(rates)
histo_p10.fill(rates_p10)
histo_div.fill(div)
fig, axes_full = lps.subplots(1, 1, figsize=(4, 3), sharex=True)

histo.plot(
    ax=axes_full, **style, histtype="fill", alpha=0.4, color=vset.blue, label="With OB"
)
histo_p10.plot(
    ax=axes_full, **style, histtype="fill", alpha=0.4, color=vset.orange, label="No OB"
)

axes_full.set_xlabel("counts/kg/day")
axes_full.set_ylabel("Prob [arb]")
axes_full.legend(loc="best")
axes_full.set_title(f"{e_str} keV")

plt.savefig("outputs/" + out_name[0:-5] + "_posterior.pdf")

fig, axes_full = lps.subplots(1, 1, figsize=(4, 3), sharex=True)

histo_div.plot(ax=axes_full, **style, color=vset.blue, histtype="fill", alpha=0.4)

axes_full.set_xlabel("rate (p10)/rate (p3-p8)")

plt.savefig("outputs/" + out_name[0:-5] + "_ratio_posterior.pdf")

# now the summary stats
w, x = histo.to_numpy()
w_p10, x_p10 = histo_p10.to_numpy()
w_div, x_div = histo_div.to_numpy()

best_fit = x[np.argmax(w)]
best_fit_p10 = x_p10[np.argmax(w_p10)]
best_div = x_div[np.argmax(w_div)]

errors = utils.get_smallest_ci(best_fit, x, w)
errors_p10 = utils.get_smallest_ci(best_fit_p10, x_p10, w_p10)
errors_div = utils.get_smallest_ci(best_div, x_div, w_div)

logger.info(
    f"For p3-8 rate  = {best_fit:.3g} + {errors[1]:.2g} - {errors[0]:.2g} cts/kg/yr"
)
logger.info(
    f"For p10 rate   = {best_fit_p10:.3g} + {errors_p10[1]:.2g} - {errors_p10[0]:.2g} cts/kg/day"
)
logger.info(
    f"p10/p3-8       = {best_div:.3g} + {errors_div[1]:.2g} - {errors_div[0]:.2g}"
)
