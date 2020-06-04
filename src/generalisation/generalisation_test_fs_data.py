#!/usr/bin/env python3
"""Script to test SVM models developed using FreeSurfer data from Biobank Scanner1
on previously unseen data from Biobank Scanner2 to predict brain age.

The script loops over the 100 SVM models created in train_svm_on_freesurfer_data.py, loads their regressors,
applies them to the Scanner2 data and saves all predictions per subjects in a csv file"""
import argparse
import random
from math import sqrt
from pathlib import Path

import numpy as np
from joblib import load
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import COLUMNS_NAME, load_freesurfer_dataset

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-T', '--training_experiment_name',
                    dest='training_experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-G', '--test_experiment_name',
                    dest='test_experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

parser.add_argument('-M', '--model_name',
                    dest='model_name',
                    help='Name of the model.')

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='cleaned_ids.csv',
                    help='Filename indicating the ids to be used.')

args = parser.parse_args()


def main(training_experiment_name, test_experiment_name, scanner_name, model_name, input_ids_file):
    # ----------------------------------------------------------------------------------------
    training_experiment_dir = PROJECT_ROOT / 'outputs' / training_experiment_name
    test_experiment_dir = PROJECT_ROOT / 'outputs' / test_experiment_name

    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / test_experiment_name / input_ids_file

    test_model_dir = test_experiment_dir / model_name
    test_model_dir.mkdir(exist_ok=True)

    training_cv_dir = training_experiment_dir / model_name / 'cv'
    test_cv_dir = test_model_dir / 'cv'
    test_cv_dir.mkdir(exist_ok=True)

    dataset = load_freesurfer_dataset(participants_path, ids_path, freesurfer_path)

    # ----------------------------------------------------------------------------------------
    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes in testing dataset by total intracranial volume (tiv)
    regions = dataset[COLUMNS_NAME].values

    tiv = dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    regions_norm = np.true_divide(regions, tiv)
    age = dataset['Age'].values

    # Create dataframe to hold actual and predicted ages
    age_predictions = dataset[['image_id', 'Age']]
    age_predictions = age_predictions.set_index('image_id')

    n_repetitions = 10
    n_folds = 10

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            # Load model and scaler
            prefix = f'{i_repetition:02d}_{i_fold:02d}'

            model = load(training_cv_dir / f'{prefix}_regressor.joblib')
            scaler = load(training_cv_dir / f'{prefix}_scaler.joblib')

            # Use RobustScaler to transform testing data
            x_test = scaler.transform(regions_norm)

            # Apply model to scaled data
            predictions = model.predict(x_test)

            absolute_error = mean_absolute_error(age, predictions)
            root_squared_error = sqrt(mean_squared_error(age, predictions))
            r2 = r2_score(age, predictions)
            age_error_corr, _ = stats.spearmanr(np.abs(age - predictions), age)

            # Save prediction per model in df
            age_predictions[f'Prediction {i_repetition:02d}_{i_fold:02d}'] = predictions

            # Save model scores
            scores_array = np.array([r2, absolute_error, root_squared_error, age_error_corr])
            np.save(test_cv_dir / f'{prefix}_scores.npy', scores_array)

    # Save predictions
    age_predictions.to_csv(test_model_dir / 'age_predictions_test.csv')


if __name__ == '__main__':
    main(args.training_experiment_name, args.test_experiment_name,
         args.scanner_name, args.model_name,
         args.input_ids_file)
