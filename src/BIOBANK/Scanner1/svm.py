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
Step 11: Print R_squared, MAE, RMSE
Step 12: Save model file, scaler file, predictions file
Step 13: Print CV results"""

from pathlib import Path
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():
    # Load hdf5 file
    dataset = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table')

    # Initialise random seed
    np.random.seed = 42
    random.seed = 42

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[dataset.columns[4:]].values
    region_labels = dataset.columns[4:]  # for future reference, if needed
    tiv = dataset.EstimatedTotalIntraCranialVol.values
    tiv = tiv.reshape(len(dataset), 1)
    regions_norm = np.true_divide(regions, tiv) # Independent vars X
    age = dataset[dataset.columns[1]].values # Dependent var Y

    # Create variable to hold CV variables
    cv_r2_scores = []

    # Create 10-fold cross-validator, stratified by age
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(regions_norm, age)
    for i, (train_index, test_index) in enumerate(skf.split(regions_norm, age)):
        print(i)
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = regions_norm[train_index], regions_norm[test_index]
        y_train, y_test = age[train_index], age[test_index]

        # Scaling in range [-1, 1]
        scaling = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaling.fit_transform(x_train)
        x_test = scaling.transform(x_test)

        # Svc using default hyper-parameters
        svm = SVR(kernel='linear')
        svm_train = svm.fit(x_train, y_train)
        params = svm.get_params()
        predictions = svm.predict(x_test)
        mean_absolute_error(y_test, predictions)
        r2_score = svm_train.score(x_test, y_test)
        cv_r2_scores.append(r2_score)

    cv_r2_mean = np.mean(cv_r2_scores)

    print('CV accuracy score: %0.3f,'
          ' test accuracy score: %0.3f'
          % (cv_performance_mean, test_performance))

    # Systematic search for better hyper-parameters
    learning_algo = SVR(kernel='linear')
    search_space = [{'kernel': ['linear'],
                     'C': np.logspace(-3, 3, 7)},
                    {'kernel': ['rbf'],
                     'C': np.logspace(-3, 3, 7),
                     'gamma': np.logspace(-3, 2, 6)}]
    gridsearch = GridSearchCV(learning_algo, param_grid=search_space, refit=True, cv=skf)
    gridsearch.fit(X_train, y_train)
    print('Best parameter: %s' % str(gridsearch.best_params_))
    cv_performance_grid = gridsearch.best_score_
    test_performance_grid = gridsearch.score(X_test, y_test)
    print('CV accuracy score: %0.3f,'
          ' test accuracy score: %0.3f'
          % (cv_performance_grid, test_performance_grid))


if __name__ == "__main__":
    main()
