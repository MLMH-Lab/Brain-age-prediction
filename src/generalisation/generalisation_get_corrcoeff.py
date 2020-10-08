"""Script to calculate Pearson's r correlation coefficient
for all model predictions per subject and chronological age
in the independent test set;

This script is adapted from comparison_get_corrcoeff"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    experiment_name = 'biobank_scanner2'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # Create df to hold r values for all models
    r_val_df = pd.DataFrame()

    # Loop over all models to obtain r values
    for model_name in model_ls:
        model_dir = experiment_dir / model_name
        file_name = model_dir / 'age_predictions_test.csv'

        try:
            model_data = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f'No age prediction file for {model_name}.')
            raise

        # Access variable names for all 100 models
        repetition_cols = model_data.loc[:,
                        'Prediction 00_00' : 'Prediction 09_09']
        repetition_col_names = repetition_cols.columns

        # Loop over all models to obtain r value for each
        r_val_ls = []

        # Initialise variable 'skipped' that counts how often r_val could not be
        # calculated for a model (possibly due to model non-convergence)
        skipped = 0

        for i_col in repetition_col_names:
            r_val, _ = pearsonr(model_data['Age'],model_data[i_col])
            if not np.isnan(r_val):
                r_val_ls.append(r_val)
            else:
                skipped += 1

        r_mean = np.mean(r_val_ls)
        r_std = np.std(r_val_ls)
        print(model_name, skipped)
        print(model_name, r_mean, r_std)

        r_val_df[model_name] = [r_mean, r_std]

    # Export age_predictions_all and r_val_df as csv
    r_val_df.to_csv(experiment_dir / 'r_values_test_allmodels.csv')


if __name__ == '__main__':
    main()