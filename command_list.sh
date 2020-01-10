## Initiate virtual enviroment
#source venv/bin/activate
#
## Make all files executable
#chmod -R +x ./
#
## Run python scripts
## ----------------------------- Getting data -------------------------------------
## Download data from network-attached storage (MLMH lab use only)
./download_data.py -P "/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/"
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

./comparison_feature_importance_visualisation.py
./comparison_feature_importance_voxel_data.py

./comparison_statistical_analsysis.py -E "biobank_scanner1" -S "_all" -M "SVM" "RVM" "GPR" "voxel_SVM" "voxel_RVM"

## ----------------------------- Generalization comparison -----------------------
./generalisation_test_svm_fs_data.py -M "SVM"
./generalisation_test_svm_fs_data.py -M "RVM"
./generalisation_test_svm_fs_data.py -M "GPR"

./comparison_statistical_analsysis.py -E "biobank_scanner2" -S "_generalization" -M "SVM" "RVM" "GPR"

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

# ----------------------------- Covariates analysis ------------------------------------
./covariates_create_variables_biobank.py
./covariates_create_variables_indices_of_deprivation.py
./covariates_ensemble_output.py
./covariates_lsoa_corr.py
./covariates_statistical_analysis.py

# ----------------------------- Miscelanious ------------------------------------
# Univariate analysis on freesurfer data
./misc_univariate_analysis.py

./misc_classifier_train_svm.py
./misc_classifier_regressor_comparison.py

# Performance of different values of the SVM hyperparameter (C)
./misc_svm_hyperparameters_analysis.py

# ----------------------------- Exploratory Data Analysis ------------------------------------
./eda_demographic_data.py
./eda_education_age.py
./eda_gender_age.py
