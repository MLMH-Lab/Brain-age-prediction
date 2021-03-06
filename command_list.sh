## Initiate virtual environment
#source venv/bin/activate
#
## Make all files executable
#chmod -R +x ./
#
export PYTHONPATH=$PYTHONPATH:./src
## Run python scripts
## ----------------------------- Getting data -------------------------------------
## Download data from network-attached storage (MLMH lab use only)
./src/download/download_data.py -N "/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/"
./src/download/download_ants_data.py -N "/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/" -S "BIOBANK-SCANNER01" -O "/media/kcl_1/SSD2"
./src/download/download_ants_data.py -N "/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/" -S "BIOBANK-SCANNER02" -O "/media/kcl_1/HDD/DATASETS/BIOBANK"

# ----------------------------- Preprocessing ------------------------------------
# Clean UK Biobank data
./src/preprocessing/clean_data.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/preprocessing/clean_data.py -E "biobank_scanner2" -S "BIOBANK-SCANNER02"

# Perform quality control
./src/preprocessing/quality_control.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/preprocessing/quality_control.py -E "biobank_scanner2" -S "BIOBANK-SCANNER02"

# Make gender homogeneous along age range
# This was only performed in scanner1 because we were concerned not to create a biased regressor
./src/preprocessing/homogenize_gender.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"

# Create kernel matrix for voxel-based analysis
./src/preprocessing/compute_kernel_matrix.py -P "/media/kcl_1/SSD2/BIOBANK" -E "biobank_scanner1"
./src/preprocessing/compute_kernel_matrix_general.py -P "/media/kcl_1/SSD2/BIOBANK" -E "biobank_scanner1" -P2 "/media/kcl_1/HDD/DATASETS/BIOBANK/BIOBANK" -E2 "biobank_scanner2"

# Create pca models
./src/preprocessing/create_pca_models.py -P "/media/kcl_1/SSD2/BIOBANK" -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/preprocessing/compute_principal_components.py -P "/media/kcl_1/SSD2/BIOBANK" -E "biobank_scanner1"
./src/preprocessing/compute_principal_components.py -P "/media/kcl_1/HDD/DATASETS/BIOBANK/BIOBANK" -E "biobank_scanner2" -I "cleaned_ids.csv" -S "_general"

# ----------------------------- Regressor comparison ------------------------------------
./src/comparison/comparison_fs_data_train_svm.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/comparison/comparison_fs_data_train_rvm.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/comparison/comparison_fs_data_train_gp.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"

./src/comparison/comparison_voxel_data_train_svm.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/comparison/comparison_voxel_data_train_rvm.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"

./src/comparison/comparison_pca_data_train_rvm.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/comparison/comparison_pca_data_train_svm.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"
./src/comparison/comparison_pca_data_train_gp.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"

./src/comparison/comparison_statistical_analysis.py -E "biobank_scanner1" -S "_all" -M "SVM" "RVM" "GPR" "voxel_SVM" "voxel_RVM" "pca_RVM" "pca_SVM" "pca_GPR"

./src/comparison/comparison_voxel_data_svm_primal_weights.py -E "biobank_scanner1" -P "/media/kcl_1/SSD2/BIOBANK"

./src/comparison/comparison_voxel_data_rvm_relevance_vectors_weights.py -E "biobank_scanner1" -P "/media/kcl_1/SSD2/BIOBANK"

./comparison_feature_importance_visualisation.py

## ----------------------------- Generalisation comparison -----------------------
./src/generalisation/generalisation_test_fs_data.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "SVM" -I "cleaned_ids.csv"
./src/generalisation/generalisation_test_fs_data.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "RVM" -I "cleaned_ids.csv"
./src/generalisation/generalisation_test_fs_data.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "GPR" -I "cleaned_ids.csv"

./src/generalisation/generalisation_test_voxel_data_svm.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "voxel_SVM" -P "/media/kcl_1/HDD/DATASETS/BIOBANK/BIOBANK"
./src/generalisation/generalisation_test_voxel_data_rvm.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "voxel_RVM" -P "/media/kcl_1/HDD/DATASETS/BIOBANK/BIOBANK"

./src/generalisation/generalisation_test_pca_data.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "pca_RVM" -I "cleaned_ids.csv"
./src/generalisation/generalisation_test_pca_data.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "pca_SVM" -I "cleaned_ids.csv"
./src/generalisation/generalisation_test_pca_data.py -T "biobank_scanner1" -G "biobank_scanner2" -S "BIOBANK-SCANNER02" -M "pca_GPR" -I "cleaned_ids.csv"

./src/comparison/comparison_statistical_analysis.py -E "biobank_scanner2" -S "_generalization" -M "SVM" "RVM" "GPR" "voxel_SVM" "voxel_RVM" "pca_RVM" "pca_SVM" "pca_GPR"

# ----------------------------- Training set size analysis ------------------------------------
./src/sample_size/sample_size_create_ids.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"

./src/sample_size/sample_size_fs_data_svm_analysis.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01" -G "biobank_scanner2" -C "BIOBANK-SCANNER02"
./src/sample_size/sample_size_fs_data_gp_analysis.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01" -G "biobank_scanner2" -C "BIOBANK-SCANNER02"
./src/sample_size/sample_size_fs_data_rvm_analysis.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01" -G "biobank_scanner2" -C "BIOBANK-SCANNER02"

./src/sample_size/sample_size_voxel_data_svm_analysis.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01" -G "biobank_scanner2" -C "BIOBANK-SCANNER02"
./src/sample_size/sample_size_voxel_data_rvm_analysis.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01" -G "biobank_scanner2" -C "BIOBANK-SCANNER02"

./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "SVM"
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "RVM"
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "GPR"
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "pca_RVM" -F 3 -R 10
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "pca_SVM" -F 3 -R 10
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "pca_GPR" -F 3 -R 10
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "voxel_SVM"
./src/sample_size/sample_size_create_figures.py -E "biobank_scanner1" -M "voxel_RVM"

# ----------------------------- Miscellaneous ------------------------------------
# Univariate analysis on FreeSurfer data
./src/misc/misc_univariate_analysis.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01"

./misc_classifier_train_svm.py
./misc_classifier_regressor_comparison.py

# Performance of different values of the SVM hyperparameter (C)
./src/misc/misc_svm_hyperparameters_analysis.py -E "biobank_scanner1"

# ----------------------------- Exploratory Data Analysis ------------------------------------
./src/eda/eda_demographic_data.py -E "biobank_scanner1" -S "BIOBANK-SCANNER01" -U "_homogenized" -I 'homogenized_ids.csv'
./src/eda/eda_demographic_data.py -E "biobank_scanner2" -S "BIOBANK-SCANNER02" -U "_cleaned" -I 'cleaned_ids.csv'
./src/eda/eda_education_age.py