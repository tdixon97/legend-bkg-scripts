# legend-bkg-scripts
Python scripts for looking at the LEGEND background data.

Currently there is:

1. `build_data.py`:     Read the event tier files into histograms.
2. `time-analysis.py`:  Look at (normalised) counting rates per run
3. `plot-spectra.py` :  Plot the overlayed spectra
4. `remove-OB.py` :     Script used to estimate the sensitivity to removing the OB

The scripts can be run in LNGS in the current production enviroment as loaded with:
    
    source "/data2/public/prodenv/setup.sh"

As:

    dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v1.0.0/config.json -- python <SCRIPT>