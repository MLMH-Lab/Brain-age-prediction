#!/usr/bin/env python3
"""Script to perform the sample size analysis using Relevant Vector Machine

NOTE: This script is adapted from comparison_train_gp_fs_data.py but
it uses KFold instead of StratifiedKFold to account for the bootstrap
samples with few participants
"""
import argparse
import random
import warnings
from math import sqrt
from pathlib import Path
import gc

import nibabel as nib
import numpy as np
from nilearn.masking import apply_mask
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA

from utils import COLUMNS_NAME, load_demographic_data

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


def load_all_subjects(subjects_path, mask_img):
    imgs = []
    subj_pbar = tqdm(subjects_path)
    for subject_path in subj_pbar:
        subj_pbar.set_description(f'Loading image {subject_path}')
        # Read in the images in this block
        try:
            img = nib.load(str(subject_path))
        except FileNotFoundError:
            print(f'No image file {subject_path}.')
            raise

        # Extract only the brain voxels. This will create a 1D array.
        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float32')
        img = np.nan_to_num(img)
        imgs.append(img)
    return imgs


def main(experiment_name, scanner_name, input_path, n_bootstrap, n_max_pair,
         general_experiment_name, general_scanner_name, input_general_path,
         general_input_ids_file, input_data_type, mask_filename):
    # ----------------------------------------------------------------------------------------
    model_name = 'pca_SVM'

    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'

    general_participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / general_scanner_name / 'participants.tsv'

    general_ids_path = PROJECT_ROOT / 'outputs' / general_experiment_name / general_input_ids_file
    general_dataset = load_demographic_data(general_participants_path, general_ids_path)

    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'homogenized_ids.csv'
    ids_df = pd.read_csv(ids_path)

    dataset_path = Path(input_path)
    subjects_path = [str(dataset_path / f'{subject_id}_Warped{input_data_type}') for subject_id in ids_df['image_id']]
    print(f'Total number of images: {len(ids_df)}')

    # Dataset_2
    dataset_path_2 = Path(input_general_path)
    ids_df_2 = pd.read_csv(PROJECT_ROOT / 'outputs' / general_experiment_name / general_input_ids_file)
    subjects_path_2 = [str(dataset_path_2 / f'{subject_id}_Warped{input_data_type}') for subject_id in ids_df_2['image_id']]


    print(f'Total number of images: {len(ids_df_2)}')

    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
    mask_img = nib.load(str(brain_mask))

    dataset_site1 = load_all_subjects(subjects_path, mask_img)
    dataset_site2 = load_all_subjects(subjects_path_2, mask_img)
    x_general = np.array(dataset_site2)
    y_general = general_dataset['Age'].values
    # ----------------------------------------------------------------------------------------

    # Loop over the 20 bootstrap samples with up to 20 gender-balanced subject pairs per age group/year
    for i_n_subject_pairs in range(3, n_max_pair + 1):
        print(f'Bootstrap number of subject pairs: {i_n_subject_pairs}')
        ids_with_n_subject_pairs_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}' / 'ids'

        scores_dir = experiment_dir / 'sample_size' / f'{i_n_subject_pairs:02d}' / 'scores'
        scores_dir.mkdir(exist_ok=True)

        # Loop over the 1000 random subject samples per bootstrap
        for i_bootstrap in range(n_bootstrap):
            print(f'Sample number within bootstrap: {i_bootstrap}')

            prefix = f'{i_bootstrap:04d}_{i_n_subject_pairs:02d}'
            train_ids = load_demographic_data(participants_path,
                                                    ids_with_n_subject_pairs_dir / f'{prefix}_train.csv')

            test_ids = load_demographic_data(participants_path,
                                                   ids_with_n_subject_pairs_dir / f'{prefix}_test.csv')

            # Initialise random seed
            np.random.seed(42)
            random.seed(42)

            indices = []
            for index, row in train_ids.iterrows():
                indices.append(list(ids_df['image_id']).index(row['image_id']))

            train_data = []
            for idx in indices:
                train_data.append(dataset_site1[idx])

            train_data = np.array(train_data)

            # test_data
            test_indices = []
            for index, row in test_ids.iterrows():
                test_indices.append(list(ids_df['image_id']).index(row['image_id']))

            test_data = []
            for idx in test_indices:
                test_data.append(dataset_site1[idx])

            test_data = np.array(test_data)

            pca = PCA(n_components=150, copy=False)
            x_train = pca.fit_transform(train_data)
            y_train = train_ids['Age'].values

            x_test = pca.transform(test_data)
            y_test = test_ids['Age'].values

            # Scaling in range [-1, 1]
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            svm = LinearSVR(loss='epsilon_insensitive')

            # Systematic search for best hyperparameters
            search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}
            n_nested_folds = 5
            nested_kf = KFold(n_splits=n_nested_folds, shuffle=True, random_state=i_bootstrap)
            gridsearch = GridSearchCV(svm,
                                      param_grid=search_space,
                                      scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_kf,
                                      verbose=0, n_jobs=-1)

            gridsearch.fit(x_train, y_train)

            best_model = gridsearch.best_estimator_

            # Test data
            predictions = best_model.predict(x_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions),
                                                y_test)

            scores = np.array([r2, mae, rmse, age_error_corr])
            np.save(
                str(scores_dir / f'scores_{i_bootstrap:04d}_{model_name}.npy'),
                scores)

            print(
                f'R2: {r2:0.3f} MAE: {mae:0.3f} RMSE: {rmse:0.3f} CORR: {age_error_corr:0.3f}')

            # Train data
            train_predictions = best_model.predict(x_train)
            train_mae = mean_absolute_error(y_train, train_predictions)
            train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
            train_r2 = r2_score(y_train, train_predictions)
            train_age_error_corr, _ = stats.spearmanr(
                np.abs(y_train - train_predictions), y_train)

            train_scores = np.array(
                [train_r2, train_mae, train_rmse, train_age_error_corr])
            np.save(str(
                scores_dir / f'scores_{i_bootstrap:04d}_{model_name}_train.npy'),
                    train_scores)

            # Generalisation data
            x_general_components = pca.transform(x_general)
            x_general_norm = scaler.transform(x_general_components)
            general_predictions = best_model.predict(x_general_norm)
            general_mae = mean_absolute_error(y_general, general_predictions)
            general_rmse = sqrt(
                mean_squared_error(y_general, general_predictions))
            general_r2 = r2_score(y_general, general_predictions)
            general_age_error_corr, _ = stats.spearmanr(
                np.abs(y_general - general_predictions), y_general)

            general_scores = np.array(
                [general_r2, general_mae, general_rmse, train_age_error_corr])
            np.save(str(
                scores_dir / f'scores_{i_bootstrap:04d}_{model_name}_general.npy'),
                    general_scores)

            del pca, test_data, train_data
            gc.collect()


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name,
         args.n_bootstrap, args.n_max_pair,
         args.general_experiment_name, args.general_scanner_name,
         args.general_input_ids_file)
