#!/bin/bash

source /data1/users/tdixon/Load.sh

dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r000-with-qc-dataset-tmp-auto --p "p11" --use_qc 1 --r "r000"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r001-with-qc-dataset-tmp-auto --p "p11" --use_qc 1 --r "r001"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r002-with-qc-dataset-tmp-auto --p "p11" --use_qc 1 --r "r002"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r003-with-qc-dataset-tmp-auto --p "p11" --use_qc 1 --r "r003"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r004-with-qc-dataset-tmp-auto --p "p11" --use_qc 1 --r "r004"

dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r000-no-qc-dataset-tmp-auto --p "p11" --use_qc 0 --r "r000"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r001-no-qc-dataset-tmp-auto --p "p11" --use_qc 0 --r "r001"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r002-no-qc-dataset-tmp-auto --p "p11" --use_qc 0 --r "r002"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r003-no-qc-dataset-tmp-auto --p "p11" --use_qc 0 --r "r003"
dataprod-load-sw /data2/public/prodenv/prod-blind/ref-v2.1.0/config.json -- python build_data.py  --cluster legend-login --version tmp-auto --output l200a-p11-r004-no-qc-dataset-tmp-auto --p "p11" --use_qc 0 --r "r004"
