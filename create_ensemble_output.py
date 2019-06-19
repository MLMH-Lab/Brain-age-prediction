"""Create variables for average age predictions and prediction errors, incl. BrainAGE, BrainAGER [1]

References:
[1] Le TT, Kuplicki RT, McKinney BA, Yeh H-W, Thompson WK, Paulus MP and Tulsa 1000 Investigators (2018)
A Nonlinear Simulation Framework Supports Adjusting for Age When Analyzing BrainAGE.
Front. Aging Neurosci. 10:317. doi: 10.3389/fnagi.2018.00317
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    n_repetitions = 10

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / 'SVM'
    correlation_dir = experiment_dir / 'correlation_analysis'
    correlation_dir.mkdir(exist_ok=True)

    # Load SVR age predictions
    age_predictions_df = pd.read_csv(svm_dir / 'age_predictions.csv')

    ensemble_df = pd.DataFrame()
    ensemble_df['Age'] = age_predictions_df[['Age', ]]

    repetition_column_name = []
    for i_repetition in range(n_repetitions):
        repetition_column_name.append('Prediction repetition {:02d}'.format(i_repetition))



    ensemble_df['Mean_predicted_age'] = age_predictions_df[repetition_column_name].mean(axis=1)
    ensemble_df['Median_predicted_age'] = age_predictions_df.iloc[:, 2:last_col].median(axis=1)
    ensemble_df['Std_predicted_age'] = age_predictions_df.iloc[:, 2:last_col].std(axis=1)

    # Add new columns to age_pred for age prediction error BrainAGE (Brain Age Gap Estimate)
    # BrainAGE is the difference between mean/median predicted and chronological age
    age_predictions_df['BrainAGE_predmean'] = age_predictions_df['Mean_predicted_age'] - age_predictions_df['Age']
    age_predictions_df['BrainAGE_predmedian'] = age_predictions_df['Median_predicted_age'] - age_predictions_df['Age']

    # Add new columns to age_pred for absolute BrainAGE
    age_predictions_df['Abs_BrainAGE_predmean'] = abs(age_predictions_df['BrainAGE_predmean'])
    age_predictions_df['Abs_BrainAGE_predmedian'] = abs(age_predictions_df['BrainAGE_predmedian'])

    # Add new columns to age_pred for BrainAGER (Brain Age Gap Estimate Residualized)
    # BrainAGER is a more robust measure of age prediction error (see Le et al. 2018)
    x = age_predictions_df['Age']
    y = age_predictions_df['Mean_predicted_age']
    x = sm.add_constant(x)
    brainager_model_predmean = sm.OLS(y, x)
    brainager_results_predmean = brainager_model_predmean.fit()
    brainager_residuals_predmean = brainager_results_predmean.resid
    age_predictions_df['BrainAGER_predmean'] = brainager_residuals_predmean

    x = age_predictions_df['Age']
    y = age_predictions_df['Median_predicted_age']
    x = sm.add_constant(x)
    brainager_model_predmedian = sm.OLS(y, x)
    brainager_results_predmedian = brainager_model_predmedian.fit()
    brainager_residuals_predmedian = brainager_results_predmedian.resid
    age_predictions_df['BrainAGER_predmedian'] = brainager_residuals_predmedian

    # Add new columsn to age_pred for absolute BrainAGER
    age_predictions_df['Abs_BrainAGER_predmean'] = abs(age_predictions_df['BrainAGER_predmean'])
    age_predictions_df['Abs_BrainAGER_predmedian'] = abs(age_predictions_df['BrainAGER_predmedian'])


if __name__ == "__main__":
    main()
