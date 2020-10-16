"""Script to get BrainAGER (Brain Age Gap Residualized [1] for all subjects
across all models.
BrainAGER removes the confounding effect of chronological age on brain
age prediction and is useful for further covariate analysis.

References:
[1] Le, T. T., Kuplicki, R. T., McKinney, B. A., Yeh, H. W., Thompson, W. K.,
Paulus, M. P., & Tulsa 1000 Investigators (2018). A Nonlinear Simulation
Framework Supports Adjusting for Age When Analyzing BrainAGE. Frontiers in
Aging Neuroscience, 10, 317. https://doi.org/10.3389/fnagi.2018.00317"""

import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # Load file that contains the brain age predictions from all models
    age_predictions_allmodels = pd.read_csv(experiment_dir /
                                            'age_predictions_allmodels.csv',
                                            index_col = 0)

    # Calculate brainAGE-Residualised (brainAGER) for all models and add to
    # age_predictions_allmodels df as new columns
    for model in model_ls:
        model_name = model + '_brainAGE'
        x = age_predictions_allmodels['Age']
        y = age_predictions_allmodels[model_name]
        x = sm.add_constant(x)
        brainager_model = sm.OLS(y, x)
        brainager_results = brainager_model.fit()
        brainager_residuals = brainager_results.resid
        brainager_col_name = model + '_brainAGER'
        age_predictions_allmodels[brainager_col_name] = brainager_residuals
        print(model_name, brainager_residuals.mean())

    # Test for age bias to check if brainAGER successfully removed it
    # from brainAGE
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

    # Save age_predictions_allmodels with new brainAGE and brainAGER columns
    age_predictions_allmodels.to_csv(
        experiment_dir / 'age_predictions_allmodels_brainager.csv')


if __name__ == '__main__':
    main()
