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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.cross_validation import cross_val_score
# from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():
    # Load hdf5 file
    dataset = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table')

    # Initialise random seed
    np.random.seed(42)
    random.seed(42)

    # Normalise regional volumes by total intracranial volume (tiv)
    regions = dataset[dataset.columns[4:]].values
    region_labels = dataset.columns[4:]  # for future reference, if needed
    tiv = dataset.EstimatedTotalIntraCranialVol.values
    tiv = tiv.reshape(len(dataset), 1)
    regions_norm = np.true_divide(regions, tiv) # Independent vars X
    age = dataset[dataset.columns[1]].values # Dependent var Y

    # 10-fold cross-validator - how can this be linked to the below steps? Does it apply to SVC?
    kf = KFold(n_splits=10)
    kf.get_n_splits(regions_norm)
    for train_index, test_index in kf.split(regions_norm):
        print("TRAIN:", train_index, "TEST:", test_index)

    # Split into training and test set with 70-30 ratio
    X_train, X_test, y_train, y_test = train_test_split(regions_norm, age, test_size=0.3)

    # Scaling in range [-1, 1]
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)

    # # Example code for svc using default hyper-parameters
    # svm = SVC()
    # cv_performance=cross_val_score(svm, X_train, y_train, cv=10)
    # test_performance = svm.fit(X_train, y_train).score(X_test, y_test)
    # print('CV accuracy score: %0.3f,'
    #       ' test accuracy score: %0.3f'
    #       % (np.mean(cv_performance), test_performance))
    #
    # # Example code for systematic search for better hyper-parameters
    # learning_algo = SVC(kernel='linear', random_state = 101)
    # search_space = [{'kernel': ['linear'],
    #                  'C': np.logspace(-3, 3, 7)},
    #                 {'kernel': ['rbf'],
    #                  'C': np.logspace(-3, 3, 7),
    #                  'gamma': np.logspace(-3, 2, 6)}]
    # gridsearch = GridSearchCV(learning_algo, param_grid=search_space, refit=True, cv=10)
    # gridsearch.fit(X_train, y_train)
    # print('Best parameter: %s' % str(gridsearch.best_params_))
    # cv_performance = gridsearch.best_score_
    # test_performance = gridsearch.score(X_test, y_test)
    # print('CV accuracy score: %0.3f,'
    #       ' test accuracy score: %0.3f'
    #       % (cv_performance, test_performance))


if __name__ == "__main__":
    main()
