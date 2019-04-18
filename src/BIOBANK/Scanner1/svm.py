"""Script to implement SVM in BIOBANK Scanner1 freesurfer data to predict brain age

Step 1: Set global random seed
Step 2: Normalise by TiV
Step 3: Prepare CV variables
Step 4: Create loops for repetitions and folds
Step 5: Split into training and test sets
Step 6: Scaling
Step 7: Declare search space
Step 8: Perform search with nested CV
Step 9: Retrain best model with whole training set
Step 10: Predict test set
Step 11: Print R_squared, mean absolute error (MAE), root mean squared error (RMSE)
Step 12: Save model file, scaler file, predictions file
Step 13: Print CV results"""

from math import sqrt
from pathlib import Path
import random
import warnings
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def run_svm(input_dataset='/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/homogeneous_dataset_freesurferData.h5',
         output_dir='/home/lea/PycharmProjects/predicted_brain_age/outputs/total'):
    warnings.filterwarnings('ignore')

    # Load hdf5 dataset
    dataset = pd.read_hdf(input_dataset, key='table')
    dataset = dataset[1:50] # remove after testing

    # Create output_dir if necessary
    if not os.path.exists(str(output_dir)):
        os.makedirs(str(output_dir))

    # Initialise random seed
    np.random.seed = 42
    random.seed = 42

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[dataset.columns[4:-2]].values
    tiv = dataset.EstimatedTotalIntraCranialVol.values
    tiv = tiv.reshape(len(dataset), 1)
    regions_norm = np.true_divide(regions, tiv)  # Independent vars X
    age = dataset[dataset.columns[1]].values  # Dependent var Y

    # Create variable to hold CV variables
    cv_r2_scores = []
    cv_mae = []
    cv_rmse = []

    # Create dataframe to hold actual and predicted ages + df for loop to add predictions to
    age_predictions = pd.DataFrame(dataset[['Participant_ID', 'Age']])
    age_predictions['Index'] = age_predictions.index

    n_repetitions = 10
    n_folds = 10
    n_nested_folds = 5

    # Loop to repeat 10-fold CV 10 times
    for i_repetition in range(n_repetitions):

        # Create new empty column in age_predictions df to save age predictions of this repetition
        age_predictions['Prediction repetition %02d' % i_repetition] = np.nan

        # Create 10-fold cross-validator, stratified by age
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)

        for i_fold, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
            print('Running repetition %02d, fold %02d' % (i_repetition, i_fold))

            x_train, x_test = regions_norm[train_index], regions_norm[test_index]
            y_train, y_test = age[train_index], age[test_index]

            # Scaling in range [-1, 1]
            scaling = RobustScaler()
            x_train = scaling.fit_transform(x_train)
            x_test = scaling.transform(x_test)

            # Systematic search for best hyperparameters
            svm = LinearSVR(loss='epsilon_insensitive')

            c_range = [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0, 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7]
            search_space = [{'C': c_range}]
            nested_skf = StratifiedKFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

            gridsearch = GridSearchCV(svm, param_grid=search_space, scoring='neg_mean_absolute_error',
                                      refit=True, cv=nested_skf, verbose=3, n_jobs=1)

            gridsearch.fit(x_train, y_train)

            best_svm = gridsearch.best_estimator_

            params_results = {'means': gridsearch.cv_results_['mean_test_score'],
                              'params': gridsearch.cv_results_['params']}

            predictions = best_svm.predict(x_test)

            absolute_error = mean_absolute_error(y_test, predictions)
            root_squared_error = sqrt(mean_squared_error(y_test, predictions))
            r2_score = best_svm.score(x_test, y_test)

            cv_r2_scores.append(r2_score)
            cv_mae.append(absolute_error)
            cv_rmse.append(root_squared_error)

            # Save scaler, model and model parameters
            scaler_file_name = str(i_repetition) + '_' + str(i_fold) + '_scaler.joblib'
            model_file_name = str(i_repetition) + '_' + str(i_fold) + '_svm.joblib'
            params_file_name = str(i_repetition) + '_' + str(i_fold) + '_svm_params.joblib'
            dump(scaling, str(output_dir + '/' + scaler_file_name))
            dump(params_results, str(output_dir + '/' + params_file_name))
            dump(best_svm, str(output_dir + '/' +  model_file_name))

            # Save model scores r2, MAE, RMSE
            scores_array = np.array([r2_score, absolute_error, root_squared_error])
            scores_file_name = str(i_repetition) + '_' + str(i_fold) + '_svm_scores.npy'
            filepath_scores = str(output_dir + '/' + scores_file_name)
            np.save(filepath_scores, scores_array)

            # Create new df to hold test_index and corresponding age prediction
            new_df = pd.DataFrame()
            new_df['index'] = test_index
            new_df['predictions'] = predictions

            # Add predictions per test_index to age_predictions
            for index, row in new_df.iterrows():
                col_index = i_repetition + 3
                sub_index = int(row['index'])
                age_predictions.iloc[[sub_index], [col_index]] = row['predictions']

            # Print results of the CV fold
            print('Repetition %02d, Fold %02d, R2: %0.3f, MAE: %0.3f, RMSE: %0.3f'
                  % (i_repetition, i_fold, r2_score, absolute_error, root_squared_error))

    # Save predictions
    age_predictions = age_predictions.drop('Index', axis=1)
    age_predictions.to_csv(str(output_dir + '/age_predictions.csv'), index=False)

    # Variables for CV means across all repetitions
    cv_r2_mean = np.mean(cv_r2_scores)
    cv_mae_mean = np.mean(cv_mae)
    cv_rmse_mean = np.mean(cv_rmse)
    print('Mean R2: %0.3f, MAE: %0.3f, RMSE: %0.3f' % (cv_r2_mean, cv_mae_mean, cv_rmse_mean))


if __name__ == "__main__":
    run_svm()
