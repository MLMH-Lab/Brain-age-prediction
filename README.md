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


## Installing the dependencies
Install virtualenv and creating a new virtual environment:

    pip install virtualenv
    virtualenv -p /usr/bin/python3 ./venv

Install dependencies

    pip3 install -r requirements.txt


## How to run

Order of scripts:
1. dem_bias: Assess and correct for demographic bias
2. gender_age: Assess and visualise gender distribution
3. univariate_analysis: Univariate analysis of age and regional volume
4. create_h5_dataset: Create h5 file for machine learning analysis
5. svm: Create SVR model
6. permutation: Permutation of SVR models
7. permutation_sig: Assess significance of SVR models
8. var_corr: Analyse demographic covariates in BIOBANK dataset
9. education_age: Assess and visualise education distribution (!This this script does not run in this position (age_predictions_demographics.csv missing))
10. lsoa_corr: Analyse demographic covariates from English Index of Multiple Deprivation


## Citation
If you find this code useful for your research, please cite:

    @article{}