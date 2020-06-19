#!/usr/bin/env python3
"""Script to perform the sample size analysis using Relevant Vector Machine. """
import argparse
import random
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn_rvm import EMRVR

from utils import COLUMNS_NAME, load_freesurfer_dataset

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

parser.add_argument('-N', '--n_bootstrap',
                    dest='n_bootstrap',
                    type=int, default=1000,
                    help='Number of bootstrap iterations.')

parser.add_argument('-R', '--n_max_pair',
                    dest='n_max_pair',
                    type=int, default=20,
                    help='Number maximum of pairs.')

parser.add_argument('-G', '--general_experiment_name',
                    dest='general_experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-C', '--general_scanner_name',
                    dest='general_scanner_name',
                    help='Name of the scanner for generalization.')

parser.add_argument('-I', '--general_input_ids_file',
                    dest='general_input_ids_file',
                    default='cleaned_ids.csv',
                    help='Filename indicating the ids to be used.')

args = parser.parse_args()


def main(experiment_name, scanner_name, n_bootstrap, n_max_pair,
         general_experiment_name, general_scanner_name, general_input_ids_file):
    model_name = 'RVM'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'freesurferData.csv'

    general_participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / general_scanner_name / 'participants.tsv'
    general_freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / general_scanner_name / 'freesurferData.csv'

    general_ids_path = PROJECT_ROOT / 'outputs' / general_experiment_name / general_input_ids_file
    general_dataset = load_freesurfer_dataset(general_participants_path, general_ids_path, general_freesurfer_path)

    # Normalise regional volumes by total intracranial volume (tiv)
    general_regions = general_dataset[COLUMNS_NAME].values

    general_tiv = general_dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

    x_general = np.true_divide(general_regions, general_tiv)
    y_general = general_dataset['Age'].values

    # ----------------------------------------------------------------------------------------

    # Loop over the 20 bootstrap samples with up to 20 gender-balanced subject pairs per age group/year
    for i_n_subject_pairs in range(1, n_max_pair + 1):
        print(f'Bootstrap number of subject pairs: {i_n_subject_pairs}')
        ids_with_n_subject_pairs_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}' / 'ids'

        scores_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}' / 'scores'
        scores_dir.mkdir(exist_ok=True)

        # Loop over the 1000 random subject samples per bootstrap
        for i_bootstrap in range(n_bootstrap):
            print(f'Sample number within bootstrap: {i_bootstrap}')

            prefix = f'{i_bootstrap:04d}_{i_n_subject_pairs:02d}'
            train_dataset = load_freesurfer_dataset(participants_path,
                                                    ids_with_n_subject_pairs_dir / f'{prefix}_train.csv',
                                                    freesurfer_path)
            test_dataset = load_freesurfer_dataset(participants_path,
                                                   ids_with_n_subject_pairs_dir / f'{prefix}_test.csv',
                                                   freesurfer_path)

            # Initialise random seed
            np.random.seed(42)
            random.seed(42)

            # Normalise regional volumes by total intracranial volume (tiv)
            regions = train_dataset[COLUMNS_NAME].values

            tiv = train_dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]

            x_train = np.true_divide(regions, tiv)
            y_train = train_dataset['Age'].values

            test_tiv = test_dataset.EstimatedTotalIntraCranialVol.values[:, np.newaxis]
            test_regions = test_dataset[COLUMNS_NAME].values

            x_test = np.true_divide(test_regions, test_tiv)
            y_test = test_dataset['Age'].values

            # Scaling in range [-1, 1]
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Systematic search for best hyperparameters
            rvm = EMRVR(kernel='linear', threshold_alpha=1e9)
            rvm.fit(x_train, y_train)

            # Test data
            predictions = rvm.predict(x_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)

            scores = np.array([r2, mae, rmse, age_error_corr])
            np.save(str(scores_dir / f'scores_{i_bootstrap:04d}_{model_name}.npy'), scores)

            print(f'R2: {r2:0.3f} MAE: {mae:0.3f} RMSE: {rmse:0.3f} CORR: {age_error_corr:0.3f}')

            # Train data
            train_predictions = rvm.predict(x_train)
            train_mae = mean_absolute_error(y_train, train_predictions)
            train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
            train_r2 = r2_score(y_train, train_predictions)
            train_age_error_corr, _ = stats.spearmanr(np.abs(y_train - train_predictions), y_train)

            train_scores = np.array([train_r2, train_mae, train_rmse, train_age_error_corr])
            np.save(str(scores_dir / f'scores_{i_bootstrap:04d}_{model_name}_train.npy'), train_scores)

            # Generalisation data
            x_general_norm = scaler.transform(x_general)
            general_predictions = rvm.predict(x_general_norm)
            general_mae = mean_absolute_error(y_general, general_predictions)
            general_rmse = sqrt(mean_squared_error(y_general, general_predictions))
            general_r2 = r2_score(y_general, general_predictions)
            general_age_error_corr, _ = stats.spearmanr(np.abs(y_general - general_predictions), y_general)

            general_scores = np.array([general_r2, general_mae, general_rmse, train_age_error_corr])
            np.save(str(scores_dir / f'scores_{i_bootstrap:04d}_{model_name}_general.npy'), general_scores)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.n_bootstrap, args.n_max_pair,
         args.general_experiment_name, args.general_scanner_name, args.general_input_ids_file)
