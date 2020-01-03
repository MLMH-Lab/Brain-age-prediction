"""
Script to implement SVM in BIOBANK Scanner1 using voxel data to predict brain age.

This script assumes that a kernel has been already pre-computed. To compute the
kernel use the script `precompute_3Ddata.py`
"""
from math import sqrt
from pathlib import Path
import random
import warnings
import argparse

from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.model_selection import GridSearchCV
import nibabel as nib
from nilearn.masking import apply_mask

from helper_functions import load_demographic_data

PROJECT_ROOT = Path.cwd()

warnings.filterwarnings('ignore')


def main(i_repetition):
    # --------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'participants.tsv'
    id_path = PROJECT_ROOT / 'outputs' / experiment_name / 'dataset_homogeneous.csv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'freesurferData.csv'
    dataset_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1'

    # TODO: select only subjects in the kernel (for 100 subjects analysis)
    kernel_path = PROJECT_ROOT / 'outputs' / 'kernels' / 'kernel_100.csv'

    # Load demographics
    demographics = load_demographic_data(demographic_path, id_path)
    freesurfer = pd.read_csv(freesurfer_path)
    freesurfer['Participant_ID'] = freesurfer['Image_ID'].str.split('_', expand=True)[0]
    demographics = pd.merge(freesurfer, demographics, on='Participant_ID')
    demographics.set_index('Participant_ID', inplace=True)

    # Load the Gram matrix
    kernel = pd.read_csv(kernel_path, header=0, index_col=0)

    # Remove the additional subjects from kernel
    # TODO: Just a hack to make it run but we should make sure that the ids from
    # dataset_homogeneous is the same form the cleaned_ids.csv. Or just use the
    # dataset_homogeneous to create the kernel.
    kernel.columns = kernel.columns.str.replace('(', '').str.replace(')','').str.replace('\'','').str.replace(',', '')
    idx_missing = kernel[~kernel.index.isin(demographics['Age'].index)].index
    kernel = kernel[kernel.index.isin(demographics['Age'].index)]
    kernel = kernel.drop(columns=idx_missing)
    demographics = demographics[demographics.index.isin(kernel.index)]

    # Compute SVM
    # --------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    svm_dir = experiment_dir / 'voxel_SVM'
    svm_dir.mkdir(exist_ok=True, parents=True)
    cv_dir = svm_dir / 'cv'
    cv_dir.mkdir(exist_ok=True, parents=True)

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    age = demographics['Age'].values

    # Cross validation variables
    cv_r2_scores = []
    cv_mae = []
    cv_rmse = []
    cv_age_error_corr = []

    # Feature important
    coefs = []

    # Create Dataframe to hold actual and predicted ages
    age_predictions = pd.DataFrame(demographics['Age'])

    n_folds = 10
    n_nested_folds = 5

    # Load the mask image
    brain_mask = PROJECT_ROOT / 'imaging_preprocessing_ANTs' / 'mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
    mask_img = nib.load(str(brain_mask))

    # Create new empty column in age_predictions df to save age predictions of this repetition
    repetition_column_name = 'Prediction repetition {:02d}'.format(i_repetition)
    age_predictions[repetition_column_name] = np.nan

    # Create 10-fold cross-validation scheme stratified by age
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
    for i_fold, (train_index, test_index) in enumerate(skf.split(kernel, age)):
        print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))

        x_train = kernel.iloc[train_index, train_index].values
        x_test = kernel.iloc[test_index, train_index].values
        y_train, y_test = age[train_index], age[test_index]

        # Systematic search for best hyperparameters
        svm = SVR(kernel='precomputed')

        search_space = {'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                              2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]}

        nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True,
                                     random_state=i_repetition)

        gridsearch = GridSearchCV(svm,
                                  param_grid=search_space,
                                  scoring='neg_mean_absolute_error',
                                  refit=True, cv=nested_skf,
                                  verbose=3, n_jobs=1)

        gridsearch.fit(x_train, y_train)

        best_svm = gridsearch.best_estimator_

        params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                          'params': gridsearch.cv_results_['params']}

        predictions = best_svm.predict(x_test)

        absolute_error = mean_absolute_error(y_test, predictions)
        root_squared_error = sqrt(mean_squared_error(y_test, predictions))
        r2_score = best_svm.score(x_test, y_test)
        age_error_corr, _ = stats.spearmanr(np.abs(y_test - predictions), y_test)
        # TODO: not sure if this will work when we have all subjects
        # Calculate feature importance
        training_subj_index = kernel.iloc[train_index, train_index].index
        # Load subject's data
        images = []
        for subject in training_subj_index:
            img = nib.load(str(dataset_path /
                               '{}_ses-bl_T1w_Warped.nii.gz'.format(subject)))
            img = apply_mask(img, mask_img)
            img = np.asarray(img, dtype='float64')
            img = np.nan_to_num(img)
            images.append(img)
        images = np.array(images)
        coef = np.dot(best_svm.dual_coef_, images[best_svm.support_])

        cv_r2_scores.append(r2_score)
        cv_mae.append(absolute_error)
        cv_rmse.append(root_squared_error)
        cv_age_error_corr.append(age_error_corr)
        coefs.append(coef)

        # Save scaler, model and model parameters
        model_filename = '{:02d}_{:02d}_regressor.joblib'.format(i_repetition, i_fold)
        params_filename = '{:02d}_{:02d}_params.joblib'.format(i_repetition, i_fold)

        dump(params_results, cv_dir / params_filename)
        dump(best_svm, cv_dir / model_filename)

        # Save model scores r2, MAE, RMSE
        scores_array = np.array([r2_score, absolute_error,
                                 root_squared_error, age_error_corr])

        scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)

        np.save(cv_dir / scores_filename, scores_array)

        # Add predictions per test_index to age_predictions
        for row, value in zip(test_index, predictions):
            age_predictions.iloc[row, age_predictions.columns.get_loc(repetition_column_name)] = value

        # Print results of the CV fold
        print('Repetition {:02d}, Fold {:02d}, R2: {:0.3f}, MAE: {:0.3f}, RMSE: {:0.3f}, CORR: {:0.3f}'
              .format(i_repetition, i_fold, r2_score, absolute_error, root_squared_error, age_error_corr))

    # Save feature importance for the different repetitions'
    weights_filename = '{:02d}_weights.npy'.format(i_repetition)
    np.save(cv_dir / weights_filename, coefs)

    # Save predictions
    age_predictions_filename = '{:02d}_age_predictions.csv'.format(i_repetition)
    age_predictions.to_csv(cv_dir / age_predictions_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('i_repetition', help='index of the repetition to run', type=int)
    args = parser.parse_args()

    main(args.i_repetition)
