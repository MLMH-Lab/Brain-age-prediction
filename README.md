# Paper title
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/Warvito/discovering-hidden-factors-of-variation-in-deep-networks/blob/master/LICENSE)

Repository with Lea&#39;s project (predicted_brain_age) using only UK BIOBANK.


## Abstract
Put paper abstract here


## Requirements
- Python 3
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Statsmodels](https://www.statsmodels.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [PyTables](https://www.pytables.org/)


## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 ./venv

Install dependencies

    pip3 install -r requirements.txt


## How to run

Order of scripts:
1. eda_demographic_data: Assess and visualise dataset in terms of age, gender, ethnicity
2.  clean_biobank1_data: Remove subjects with a brain-based disorder, ethnic minorities, and age groups with <100 subjects
3.  qualitycheck_biobank1_data: Remove subjects that did not pass quality checks of MRI and Freesurfer segmentation
4.  create_homogeneous_dataset: Create dataset that is homogeneous in terms of age and gender
5.  univariate_analysis: Univariate analysis of age and regional volume
6.  create_h5_dataset: Create hdf5 file of homogeneous dataset for machine learning analysis
7.  train_svm_on_freesurfer_data: Create SVR models
8.  train_rvm_on_freesurfer_data: Create RVM models
9.  train_gpr_on_freesurfer_data: Create GPR models
10.  permutation: Permutation of SVR models
11. run_permutation_significance_test: Assess significance of SVR models
12. params_analysis: Compare SVM models with different hyperparameters
13. create_bootstrap_ids: Create gender-homogeneous bootstrap datasets
14. create_h5_bootstrap: Create bootstrap files in hdf5 format
15. svm_bootstrap: Run SVR on bootstrap datasets
16. svm_classifier_bootstrap: Run SVC on bootstrap datasets
17. regressors_comparison: Compares performance of SVM, RVM and GPR models
18. regressor_classifier_comparison: Compare performance of SVR and SVC models
19. create_ensemble_output: Create variables for average age predictions and prediction errors
20. create_variables_biobank: Prepare Biobank variables for correlation analysis
21. correlation_analysis: Perform correlation analysis of age predictions and Biobank variables
22. eda_education_age: Assess and visualise distribution of education levels across age groups
23. create_variables_indices_of_deprivation (optional): Prepare variables from English Indices of Deprivation for correlation analysis
24. lsoa_corr: erform correlation analysis of age predictions and English Indices of Deprivation


## Citation
If you find this code useful for your research, please cite:

    @article{}