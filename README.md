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

    dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v1.0.0/config.json -- python build_data.py --output <OUTNAME> --p <LIST OF PERIODS> --proc <PROC EVENT>

Where:
* < OUTNAME >: is the output file name (eg l200a-p10-r000-dataset-tmp-auto)
* < LIST OF PERIODS >: Is the list of periods to proces eg ['p10']
* < PROC EVENT >: is a boolean to get the data from evt files (True) or from parquet (if it exists - False)
