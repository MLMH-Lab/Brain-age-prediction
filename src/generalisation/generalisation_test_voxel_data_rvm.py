#!/usr/bin/env python3
"""Script to test SVM models developed using FreeSurfer data from Biobank Scanner1
on previously unseen data from Biobank Scanner2 to predict brain age.

The script loops over the 100 SVM models created in train_svm_on_freesurfer_data.py, loads their regressors,
applies them to the Scanner2 data and saves all predictions per subjects in a csv file"""
import argparse
import random
from math import sqrt
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import load
from nilearn.masking import apply_mask
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import pairwise_kernels
from tqdm import tqdm

from utils import load_demographic_data

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

parser.add_argument('-P', '--input_path',
                    dest='input_path_str',
                    help='Path to the local folder with preprocessed images.')

parser.add_argument('-I', '--input_ids_file',
                    dest='input_ids_file',
                    default='cleaned_ids.csv',
                    help='Filename indicating the ids to be used.')

parser.add_argument('-N', '--mask_filename',
                    dest='mask_filename',
                    default='mni_icbm152_t1_tal_nlin_sym_09c_mask.nii',
                    help='Input data type')

parser.add_argument('-D', '--input_data_type',
                    dest='input_data_type',
                    default='.nii.gz',
                    help='Input data type')

args = parser.parse_args()


def main(training_experiment_name,
         test_experiment_name,
         scanner_name,
         input_path_str,
         model_name,
         input_ids_file,
         input_data_type,
         mask_filename):
    # ----------------------------------------------------------------------------------------
    input_path = Path(input_path_str)
    training_experiment_dir = PROJECT_ROOT / 'outputs' / training_experiment_name
    test_experiment_dir = PROJECT_ROOT / 'outputs' / test_experiment_name

    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / scanner_name / 'participants.tsv'
    ids_path = PROJECT_ROOT / 'outputs' / test_experiment_name / input_ids_file

    test_model_dir = test_experiment_dir / model_name
    test_model_dir.mkdir(exist_ok=True)

    training_cv_dir = training_experiment_dir / model_name / 'cv'
    test_cv_dir = test_model_dir / 'cv'
    test_cv_dir.mkdir(exist_ok=True)

    demographic = load_demographic_data(participants_path, ids_path)

    # ----------------------------------------------------------------------------------------
    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / mask_filename
    mask_img = nib.load(str(brain_mask))

    age = demographic['Age'].values

    # Create dataframe to hold actual and predicted ages
    age_predictions = demographic[['image_id', 'Age']]
    age_predictions = age_predictions.set_index('image_id')

    n_repetitions = 10
    n_folds = 10

    relevance_vector = {}
    models = {}
    i = 0
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            prefix = f'{i_repetition:02d}_{i_fold:02d}'
            relevance_vector[i] = np.load(training_cv_dir / f'{prefix}_relevance_vectors.npz')['relevance_vectors_']
            models[i] = load(training_cv_dir / f'{prefix}_regressor.joblib')
            i = i + 1

    for i, subject_id in enumerate(tqdm(demographic['image_id'])):
        subject_path = input_path / f"{subject_id}_Warped{input_data_type}"
        print(subject_path)

        try:
            img = nib.load(str(subject_path))
        except FileNotFoundError:
            print(f'No image file {subject_path}.')
            raise

        img = apply_mask(img, mask_img)
        img = np.asarray(img, dtype='float64')
        img = np.nan_to_num(img)

        j = 0
        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                # Load model and scaler
                prefix = f'{i_repetition:02d}_{i_fold:02d}'

                try:
                    K = pairwise_kernels(img[None,:], Y=relevance_vector[j] , metric='linear')
                except:
                    K = [[]]
                K = K / models[j]._scale
                K = np.hstack((np.ones((1, 1)), K))

                prediction = K @ models[j].mu_

                # Save prediction per model in df
                age_predictions.loc[subject_id, f'Prediction {prefix}'] = prediction

                j =j+1

    # Save predictions
    age_predictions.to_csv(test_model_dir / 'age_predictions_test.csv')

    # Get and save scores
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            prefix = f'{i_repetition:02d}_{i_fold:02d}'
            predictions = age_predictions[f'Prediction {prefix}'].values

            mae = mean_absolute_error(age, predictions)
            rmse = sqrt(mean_squared_error(age, predictions))
            r2 = r2_score(age, predictions)
            age_error_corr, _ = stats.spearmanr(np.abs(age - predictions), age)

            # Save model scores
            scores_array = np.array([r2, mae, rmse, age_error_corr])
            np.save(test_cv_dir / f'{prefix}_scores.npy', scores_array)


if __name__ == '__main__':
    main(args.training_experiment_name, args.test_experiment_name, args.scanner_name,
         args.input_path_str, args.model_name, args.input_ids_file,
         args.input_data_type, args.mask_filename)
