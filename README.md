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
1.  eda_demographic_data: Assess and visualise dataset in terms of age, gender, ethnicity
2.  clean_biobank1_data: Remove subjects with a brain-based disorder, ethnic minorities, and age groups with <100 subjects
3.  create_homogeneous_dataset: Create dataset that is homogeneous in terms of age and gender
4.  univariate_analysis: Univariate analysis of age and regional volume
5.  create_h5_dataset: Create hdf5 file of homogeneous dataset for machine learning analysis
6.  train_svm_on_freesurfer_data: Create SVR models
7.  train_rvm_on_freesurfer_data: Create RVM models
8.  train_gpr_on_freesurfer_data: Create GPR models
9.  run_permutation_subprocess: Permutation of SVR models
10. run_permutation_significance_test: Assess significance of SVR models
11. params_analysis: Compare SVM models with different hyperparameters
12. create_bootstrap_ids: Create gender-homogeneous bootstrap datasets
13. create_h5_bootstrap: Create bootstrap files in hdf5 format
14. svm_bootstrap: Run SVR on bootstrap datasets
15. svm_classifier_bootstrap: Run SVC on bootstrap datasets
16. regressors_comparison: Compares performance of SVM, RVM and GPR models
17. regressor_classifier_comparison: Compare performance of SVR and SVC models
18. create_ensemble_output: Create variables for average age predictions and prediction errors
19. create_variables_biobank: Prepare Biobank variables for correlation analysis
20. correlation_analysis: Perform correlation analysis of age predictions and Biobank variables
21. eda_education_age: Assess and visualise distribution of education levels across age groups
22. create_variables_indices_of_deprivation (optional): Prepare variables from English Indices of Deprivation for correlation analysis
23. lsoa_corr: erform correlation analysis of age predictions and English Indices of Deprivation


## Citation
If you find this code useful for your research, please cite:

    @article{}