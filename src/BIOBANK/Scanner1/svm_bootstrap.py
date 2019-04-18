"""Script to run SVM on bootstrap datasets of UK BIOBANK Scanner1
IMPORTANT NOTE: This script is adapted from svm.py but uses KFold instead of StratifiedKFold
to account for the bootstrap samples with few participants"""

from math import sqrt
from pathlib import Path
import random
import warnings
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():
    # Disable warnings
    warnings.filterwarnings('ignore') # there's a deprecation warning for model_selection iid

    # Loop over bootstrap samples
    for file_path in glob.iglob(
            '/home/lea/PycharmProjects/predicted_brain_age/data/BIOBANK/Scanner1/bootstrap/h5_datasets/homogeneous_bootstrap_*.h5',
            recursive=True):

        # Load hdf5 dataset for that bootstrap sample
        input_dataset = file_path
        dataset = pd.read_hdf(input_dataset, key='table')

        # Create output_dir for that specific bootstrap sample
        file_name = os.path.basename(file_path)
        output_dir = str('home/lea/PycharmProjects/predicted_brain_age/outputs/bootstrap/svm/' + file_name)
        if not os.path.exists(str(output_dir)):
            os.makedirs(str(output_dir))

        # Initialise random seed
        np.random.seed = 42
        random.seed = 42

        # Normalise regional volumes by total intracranial volume (tiv)
        regions = dataset[dataset.columns[5:-2]].values # note: first regional col is 5 now because of gender added to the dataset
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

            # Create 10-fold cross-validator
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)

            for i_fold, (train_index, test_index) in enumerate(kf.split(regions_norm, age)):
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
                nested_kf = KFold(n_splits=n_nested_folds, shuffle=True, random_state=i_repetition)

                gridsearch = GridSearchCV(svm, param_grid=search_space, scoring='neg_mean_absolute_error',
                                          refit=True, cv=nested_kf, verbose=3, n_jobs=1)

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
    main()
