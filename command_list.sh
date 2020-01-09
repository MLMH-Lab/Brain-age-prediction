## Initiate virtual enviroment
#source venv/bin/activate
#
## Make all files executable
#chmod -R +x ./
#
## Run python scripts
## ...
## ----------------------------- Regressor Comparison ----------------------------
## ...

./regressors_comparison.py -E "biobank_scanner1" -S "fs" -M "SVM" "RVM" "GPR"

## ...
## ----------------------------- Generalization comparison -----------------------
## ...

./regressors_comparison.py -E "biobank_scanner2" -S "generalization" -M "SVM" "RVM" "GPR"
