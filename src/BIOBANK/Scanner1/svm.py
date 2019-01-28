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
import random

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def normalise_region_df(df, region_name): # to do: to be adapted for h5
    """Normalise regional volume within df"""

    normalised_df["Norm_vol_" + region_name] = df[region_name] / df['EstimatedTotalIntraCranialVol'] * 100

    return normalised_df


def main():

    # Load freesurfer data as hdf5
    dataset = pd.read_csv(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.csv')
    dataset_hdf = dataset.to_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table', mode='w')
    dataset_test = pd.read_hdf(PROJECT_ROOT / 'data/BIOBANK/Scanner1/freesurferData.h5', key='table')
    # question: what's the difference between these?

    # Initialise random number generator with random seed
    random.seed(30)
    # question: should I use np.random instead (depending on what we will use later)?
    # do we have a specific seed value to choose?

if __name__ == "__main__":
    main()