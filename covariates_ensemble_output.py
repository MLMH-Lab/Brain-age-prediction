#!/usr/bin/env python3
"""Create variables for average age predictions and prediction errors,
incl. Brain Age Gap Estimate (BrainAGE), Brain Age Gap Estimate Residualised (BrainAGER) [1]

References:
[1] Le TT, Kuplicki RT, McKinney BA, Yeh H-W, Thompson WK, Paulus MP and Tulsa 1000 Investigators (2018)
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
    """Calculates BrainAGER, the residual error of the regression of chronological and predicted brain age"""
    age = sm.add_constant(age)
    model = sm.OLS(predicted_age, age)
    results = model.fit()
    residuals = results.resid

    return residuals


def main(experiment_name, model_name):
    """"""
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / model_name
    correlation_dir = experiment_dir / 'correlation_analysis'
    correlation_dir.mkdir(exist_ok=True)

    # Load SVR age predictions
    age_predictions_df = pd.read_csv(svm_dir / 'age_predictions.csv')

    ensemble_df = age_predictions_df[['Image_ID', 'Age']]

    n_repetitions = 10
    repetition_column_name = []
    for i_repetition in range(n_repetitions):
        repetition_column_name.append(f'Prediction repetition {i_repetition:02d}')

    ensemble_df['Mean_predicted_age'] = age_predictions_df[repetition_column_name].mean(axis=1)

    # Add new columns to ensemble_df for age prediction error BrainAGE (Brain Age Gap Estimate)
    # BrainAGE is the difference between mean predicted and chronological age
    ensemble_df['BrainAGE_predmean'] = ensemble_df['Mean_predicted_age'] - ensemble_df['Age']

    # Add new columns to ensemble_df for absolute BrainAGE
    ensemble_df['Abs_BrainAGE_predmean'] = abs(ensemble_df['BrainAGE_predmean'])

    # Add new columns to ensemble_df for BrainAGER (Brain Age Gap Estimate Residualized)
    # BrainAGER is a more robust measure of age prediction error (see Le et al. 2018)
    ensemble_df['BrainAGER_predmean'] = get_brainager(ensemble_df['Age'],
                                                      ensemble_df['Mean_predicted_age'])

    ensemble_df.to_csv(correlation_dir / f'ensemble_{model_name}_output.csv', index=False)


if __name__ == '__main__':
    main(args.experiment_name,
         args.model_name)
