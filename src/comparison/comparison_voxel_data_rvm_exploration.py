"""Script to explore voxel-based RVM models (without PCA) that
converged to the mean age of the training sample"""
#TODO: delete this script once this issue is fixed

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    # Get iteration number for models that converged to the training sample mean age
    experiment_name = 'biobank_scanner2'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    # Load list of failed models for voxel_RVM model;
    # The list was obtained in generalisation_get_corrcoef.py
    model_name = 'voxel_RVM'
    model_dir = experiment_dir / model_name
    failed_model_df = pd.read_csv(model_dir / 'failed_model.csv', index_col=0)
    failed_model_ls = failed_model_df['0'].tolist()

    # ---------------------------------
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    # Check if there are any repeat predictions in site 1 to see if any
    # models predicted mean ages there
    age_predictions = pd.read_csv(model_dir / 'age_predictions_test.csv', index_col = 0)

    for i_col in age_predictions.columns[1:]:
        boolean = age_predictions[i_col].duplicated().any()
        print(i_col, boolean)

    # --------------------------------
    # Check how many vectors the models had that performed badly on site 2 data
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    model_name = 'voxel_RVM'
    model_dir = experiment_dir / model_name
    cv_dir = model_dir / 'cv'

    n_repetitions = 10
    n_folds = 10

    new_df = pd.DataFrame()

    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            prefix = f'{i_repetition:02d}_{i_fold:02d}'

            vector_filename = prefix + '_relevance_vectors.npz'
            loaded_vector = np.load(cv_dir / vector_filename)
            vectors = loaded_vector['relevance_vectors_']
            nr_vectors = vectors.shape
            print(prefix, nr_vectors)
            new_df[prefix] = nr_vectors

    new_df.to_csv(model_dir / 'relevance_vectors.csv')


    if __name__ == '__main__':
    main()