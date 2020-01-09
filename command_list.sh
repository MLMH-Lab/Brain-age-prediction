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

# Perform quality control.
./preprocessing_quality_control.py -E "biobank_scanner1" -S "Scanner1"
./preprocessing_quality_control.py -E "biobank_scanner2" -S "Scanner2"

# Make gender homogeneous along age range (performed only in the scanner1
# because we were concerned in not create a biased regressor).
./preprocessing_homogenize_gender.py -E "biobank_scanner1"

# Create kernel matrix for voxel-based analysis
./preprocessing_compute_kernel_matrix.py

# ----------------------------- Regressors comparison ------------------------------------
./comparison_train_svm_fs_data.py
./comparison_train_rvm_fs_data.py
./comparison_train_gpr_fs_data.py

./comparison_train_voxel_data.py

./comparison_statistical_analsysis.py -E "biobank_scanner1" -S "fs" -M "SVM" "RVM" "GPR"

## ----------------------------- Generalization comparison -----------------------
./comparison_statistical_analsysis.py -E "biobank_scanner2" -S "generalization" -M "SVM" "RVM" "GPR"

# ----------------------------- Sample size analysis ------------------------------------
./sample_size_create_ids.py

./sample_size_gp_fs_analysis.py
./sample_size_rvm_fs_analysis.py
./sample_size_svm_fs_analysis.py

./sample_size_create_figures.py

# ----------------------------- Permutation ------------------------------------
#./permutation_lauch_subprocesses.py
./permutation_train_models.py
./permutation_significance_test.py
