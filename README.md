# legend-bkg-scripts
Python scripts for looking at the LEGEND background data.

Currently there is:

1. `build_data.py`:     Read the event tier files into histograms.
2. `time-analysis.py`:  Look at (normalised) counting rates per run
3. `plot-spectra.py` :  Plot the overlayed spectra
4. `remove-OB.py` :     Script used to estimate the sensitivity to removing the OB

The `build_data.py` script can be run in LNGS in the current production enviroment as loaded with:
    
    source "/data2/public/prodenv/setup.sh"

As:

    dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v1.0.0/config.json -- python build_data.py --output <OUTNAME> --p <LIST OF PERIODS> --proc <PROC EVENT> --recompute <RECOMPUTE> --target <TARGET KEY>

Where:
* `<OUTNAME>`: is the output file name (eg l200a-p10-r000-dataset-tmp-auto)
* `<LIST OF PERIODS>`: Is the list of periods to proces eg ['p10']
* `<PROC EVENT>`: is a boolean to get the data from evt files (True) or from parquet (if it exists - False)
* `<RECOMPUTE>` : is a boolean of whether to recompute QC flags based on a list of bad detectors
* `<TARGET KEY>` : is a string of type YMDTDHMZ, eg '20240317T141137Z', up to which data are included; later cycles are not loaded

The rest of the scripts do not depend on `pygama` and can be run in the LEGEND container on LNGS. 
Both have argsparsers and can be run with `python <SCRIPT> -h ` to see the options:
More details:

`plot-spectra.py` plots two spectra normalising the first one by exposure:
It can be run fom inside the LEGEND container at LNGS (can also be run locally with the correct python packages) run:

    python plot-spectra.py -o <OUTPUT PDF> -i <INPUT> -I <INPUT_P10> -e <ENERGY> -b <BINNING> -s <SPECTRUM>

Where:
* `<OUTPUT PDF>` is the name of the pdf file to save the plot (in the `plots` directory)
* `<INPUT>`: is the root file for the vancouver dataset for comparison (default `/data1/users/tdixon/build_pdf/outputs/l200a-p34678-dataset-v1.0.root`)
* `<INPUT P10>`: Is the root file for period 10
* `<ENERGY>` : is the energy range to plot (comma seperated eg 0,4000)
* `<BINNING>` is the binning
* `<SPECTRUM>` is the spectrum to plot, default `mul_surv` could also be `mul_lar_surv` etc (look in the root files)

`time-analysis.py` plots the rate of events in a certain energy window, it can also be run in the LNGS container:

    python time-analysis.py -o <OUTPUT> -i <INPUT> -I <INPUT_P10> -e <ENERGY> -p <PLOT_HIST> -s <SPECTRUM> -B <BAT_OVERLAY>

Most of the options are the same as `plot-spectra.py`, the difference is:
* `<OUTPUT>` is the path to save both a ROOT file with the time-histogram and a PDF, it will be appended with the enegry range.
* `<PLOT_HIST>`: is a boolean to plot the data as a histogram not a graph
* `<BAT_OVERLAY>` is a flag to overlay a BAT fit to the data, the argument is the path to the directory contaning the BAT output.