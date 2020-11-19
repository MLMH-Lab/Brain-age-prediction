"""Script to get BrainAGER (Brain Age Gap Residualised [1] for all subjects
across all models.
BrainAGER removes the confounding effect of chronological age on brain
age prediction and is useful for further covariate analysis.

References:
[1] Le, T. T., Kuplicki, R. T., McKinney, B. A., Yeh, H. W., Thompson, W. K.,
Paulus, M. P., & Tulsa 1000 Investigators (2018). A Nonlinear Simulation
Framework Supports Adjusting for Age When Analyzing BrainAGE. Frontiers in
Aging Neuroscience, 10, 317. https://doi.org/10.3389/fnagi.2018.00317"""

import argparse
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pathlib import Path

PROJECT_ROOT = Path.cwd()
#TODO: this script has overlap with covariates_ensemble_output, so decide which one is better

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

args = parser.parse_args()


def get_brainager(age, predicted_age):
    """Calculates BrainAGER, the residual error of the regression of
    chronological and predicted brain age [1]

    Parameters
    ----------
    age: float
        Chronological age of the subject
    predicted age: float
        Mean predicted age of the subject over all repetitions

    Returns
    -------
    residuals: float
        The residuals of the regression
    """

    age = sm.add_constant(age)
    model = sm.OLS(predicted_age, age)
    results = model.fit()
    residuals = results.resid

    return residuals


def main(experiment_name):
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # Load file that contains the brain age predictions from all models
    if experiment_name == 'biobank_scanner1':
        age_predictions_allmodels = pd.read_csv(experiment_dir /
                                                'age_predictions_allmodels.csv',
                                                index_col = 0)
    elif experiment_name == 'biobank_scanner2':
        age_predictions_allmodels = pd.read_csv(experiment_dir /
                                                'age_predictions_test_allmodels.csv',
                                                index_col = 0)

    # Get BrainAGER for all models and add to age_predictions_allmodels df as new columns
    for model_name in model_ls:
        brainage_col_name = model_name + '_brainAGE'
        brainager_residuals = get_brainager(age_predictions_allmodels['Age'],
                                            age_predictions_allmodels[brainage_col_name])
        brainager_col_name = model_name + '_brainAGER'
        age_predictions_allmodels[brainager_col_name] = brainager_residuals

    # Test if BrainAGER removed age bias from BrainAGE
    for model_name in model_ls:
        brainage_col_name = model_name + '_brainAGE'
        brainager_col_name = model_name + '_brainAGER'
        brainage_age_error_corr, _ = stats.spearmanr(
            age_predictions_allmodels[brainage_col_name],
            age_predictions_allmodels['Age'])
        brainager_age_error_corr, _ = stats.spearmanr(
            age_predictions_allmodels[brainager_col_name],
            age_predictions_allmodels['Age'])
        print(model_name, brainage_age_error_corr, brainager_age_error_corr)

    # Save age_predictions_allmodels with new BrainAGER columns
    age_predictions_allmodels.to_csv(
        experiment_dir / 'age_predictions_allmodels_brainager.csv')


if __name__ == '__main__':
    main(args.experiment_name)
