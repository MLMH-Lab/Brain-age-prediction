#!/usr/bin/env python3
"""Create variables for average age predictions and prediction errors for
one model at a time, incl. Brain Age Gap Estimate (BrainAGE),
Brain Age Gap Estimate Residualised (BrainAGER) [1]

References:
[1] Le TT, Kuplicki RT, McKinney BA, Yeh H-W, Thompson WK, Paulus MP and Tulsa
1000 Investigators (2018)
A Nonlinear Simulation Framework Supports Adjusting for Age When Analyzing BrainAGE.
Front. Aging Neurosci. 10:317. doi: 10.3389/fnagi.2018.00317
"""
import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-M', '--model_name',
                    dest='model_name',
                    help='Name of the model.')

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


def main(experiment_name, model_name):
    """"""
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    model_dir = experiment_dir / model_name
    correlation_dir = experiment_dir / 'correlation_analysis'
    correlation_dir.mkdir(exist_ok=True)

    # ------------------------------
    # Load age predictions and get names of age prediction columns
    # Note: this assumes that the comparison analysis for the model has been run
    n_repetitions = 10
    n_folds = 10
    repetition_column_name = []

    if experiment_name == 'biobank_scanner1':
        age_predictions_df = pd.read_csv(model_dir / 'age_predictions.csv')
        for i_repetition in range(n_repetitions):
            repetition_column_name.append(
                f'Prediction repetition {i_repetition:02d}')

    elif experiment_name == 'biobank_scanner2':
        age_predictions_df = pd.read_csv(model_dir / 'age_predictions_test.csv')
        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                repetition_column_name.append(
                    f'Prediction {i_repetition:02d}_{i_fold:02d}')

    ensemble_df = age_predictions_df[['image_id', 'Age']]

    # Add new column to ensemble_df for mean age predictions for each subject across all model repetitions
    ensemble_df['Mean_predicted_age'] = \
        age_predictions_df[repetition_column_name].mean(axis=1)

    # Add new column to ensemble_df for age prediction error BrainAGE
    ensemble_df['BrainAGE'] = \
        ensemble_df['Mean_predicted_age'] - ensemble_df['Age']

    # Add new column to ensemble_df for BrainAGER
    ensemble_df['BrainAGER'] = \
        get_brainager(ensemble_df['Age'], ensemble_df['Mean_predicted_age'])

    # Save ensemble output
    ensemble_df.to_csv(correlation_dir / f'ensemble_{model_name}_output.csv',
                       index=False)


if __name__ == '__main__':
    main(args.experiment_name,
         args.model_name)
