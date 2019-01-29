"""Script to implement SVM in BIOBANK Scanner1 freesurfer data to predict brain age

Step 1: generate dataset in hdf5 format
Step 2: Set global random seed
Step 3: Normalise by TiV
Step 4: Prepare CV vaoran;es
Step 5: Create loops for repetitions and folds
Step 6: Split into training and test sets
Step 7: Scaling
Step 8: Declare search space
Step 9: Perform search with nested CV
Step 10: Retrain best model with whole training set
Step 11: Predict test set
Step 12: Print R_squared, MAE, RMSE
Step 13: Save model file, scaler file, predictions file
Step 16: Print CV results"""

from pathlib import Path
import pandas as pd
import numpy as np


def main():

    # Initialise random number generator with random seed
    np.random.seed(30)

if __name__ == "__main__":
    main()