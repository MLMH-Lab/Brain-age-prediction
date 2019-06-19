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


def get_brainager(age, predicted_age):
    """"""
    age = sm.add_constant(age)
    model = sm.OLS(predicted_age, age)
    results = model.fit()
    residuals = results.resid

    return residuals


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

    ensemble_df = pd.DataFrame(age_predictions_df[['Participant_ID', 'Age']])

    repetition_column_name = []
    for i_repetition in range(n_repetitions):
        repetition_column_name.append('Prediction repetition {:02d}'.format(i_repetition))

    ensemble_df['Mean_predicted_age'] = age_predictions_df[repetition_column_name].mean(axis=1)
    ensemble_df['Median_predicted_age'] = age_predictions_df[repetition_column_name].median(axis=1)
    ensemble_df['Std_predicted_age'] = age_predictions_df[repetition_column_name].std(axis=1)

    # Add new columns to age_pred for age prediction error BrainAGE (Brain Age Gap Estimate)
    # BrainAGE is the difference between mean/median predicted and chronological age
    ensemble_df['BrainAGE_predmean'] = ensemble_df['Mean_predicted_age'] - ensemble_df['Age']
    ensemble_df['BrainAGE_predmedian'] = ensemble_df['Median_predicted_age'] - ensemble_df['Age']

    # Add new columns to age_pred for absolute BrainAGE
    ensemble_df['Abs_BrainAGE_predmean'] = abs(ensemble_df['BrainAGE_predmean'])
    ensemble_df['Abs_BrainAGE_predmedian'] = abs(ensemble_df['BrainAGE_predmedian'])

    # Add new columns to age_pred for BrainAGER (Brain Age Gap Estimate Residualized)
    # BrainAGER is a more robust measure of age prediction error (see Le et al. 2018)
    ensemble_df['BrainAGER_predmean'] = get_brainager(ensemble_df['Age'],
                                                      ensemble_df['Mean_predicted_age'])

    ensemble_df['BrainAGER_predmedian'] = get_brainager(ensemble_df['Age'],
                                                        ensemble_df['Median_predicted_age'])

    ensemble_df.to_csv(correlation_dir / 'ensemble_output.csv', index=False)


if __name__ == "__main__":
    main()
