#!/bin/bash

source /data1/users/tdixon/Load.sh
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-all-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r000','r001','r002','r003','r004' --use_qc 1 


dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r000-with-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r000' --use_qc 1 
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r001-with-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r001' --use_qc 1
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r002-with-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r002' --use_qc 1
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r003-with-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r003' --use_qc 1
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r004-with-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r004' --use_qc 1

dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r000-no-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r000' --use_qc 0 
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r001-no-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r001' --use_qc 0
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r002-no-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r002' --use_qc 0
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r003-no-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r003' --use_qc 0
dataprod-load-sw /data2/public/prodenv/prod-blind/tmp-p11/config.json -- python build_new_data.py --output l200a-p11-r004-no-qc-dataset-tmp-p11-fix.root --p 'p11' --c tmp-p11 --r 'r004' --use_qc 0