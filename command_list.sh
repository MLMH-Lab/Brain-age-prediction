## Initiate virtual enviroment
#source venv/bin/activate
#
## Make all files executable
#chmod -R +x ./
#
## Run python scripts
## ----------------------------- Getting data -------------------------------------
## Download data from network-attached storage (MLMH lab use only)
./download_fs_data.py -P "/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/"
./download_ants_data.py

# ----------------------------- Preprocessing ------------------------------------
# Clean UK Biobank data.
./preprocessing_clean_data.py -E "biobank_scanner1" -S "Scanner1"
./preprocessing_clean_data.py -E "biobank_scanner2" -S "Scanner2"


## ...
## ----------------------------- Regressor Comparison ----------------------------
## ...

./regressors_comparison.py -E "biobank_scanner1" -S "fs" -M "SVM" "RVM" "GPR"

## ...
## ----------------------------- Generalization comparison -----------------------
## ...

./regressors_comparison.py -E "biobank_scanner2" -S "generalization" -M "SVM" "RVM" "GPR"
