"""Script to calculate Pearson's r correlation coefficient
for all model predictions per subject and chronological age
in the independent test set;

The script also creates a summary file of mean model predictions
for all subjects across model repetitions;

This script is adapted from comparison_get_corrcoeff"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    experiment_name = 'biobank_scanner2'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # Create df with subject IDs and chronological age
    # All mean model predictions will be added to this df in the loop
    # Based on an age_predictions csv file from model training to have the
    # same order of subjects
    example_file = pd.read_csv(experiment_dir / 'SVM' / 'age_predictions_test.csv')
    age_predictions_all = pd.DataFrame(example_file.loc[:, 'image_id':'Age'])

    # Create df to hold r values for all models
    r_val_df = pd.DataFrame()

    # Loop over all models, calculate mean predictions across repetitions,
    # calculate r
    for model_name in model_ls:
        model_dir = experiment_dir / model_name
        file_name = model_dir / 'age_predictions_test.csv'

        try:
            model_data = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f'No age prediction file for {model_name}.')
            raise

        # TODO: check where the standard deviation in previous manuscript version comes from
        # is it correlation values for each of the 100 models then averaged (option 1 below)
        # or are all models averaged before correlation (option 2) and std is obtained differently?

        # OPTION 1
        # # Get mean predicted age across the 10 folds for each repetition
        # n_repetitions = 10
        # n_folds = 10
        #
        # repetition_mean_ls = []
        #
        # for i_repetition in range(n_repetitions):
        #     col_first_fold = 'Prediction ' + str(f'{i_repetition:02}') + '_00'
        #     col_last_fold = 'Prediction ' + str(f'{i_repetition:02}') + '_09'
        #     fold_cols = model_data.loc[:, col_first_fold : col_last_fold]
        #     new_col_name = 'Mean_repetition_' + str(f'{i_repetition:02}')
        #     model_data[new_col_name] = fold_cols.mean(axis=1)
        #     repetition_mean_ls.append(new_col_name)
        #
        # # get r value
        # for repetition_col in repetition_mean_ls:
        #     r_val = model_data['Age'].corr(model_data[repetition_col])
        #     print(model_name, r_val)

        # # Add r value to r_val_df
        # r_val_df[model_name] = [r_val]


        #OPTION 2
        repetition_cols = model_data.loc[:,
                        'Prediction 00_00' : 'Prediction 09_09']

        # get mean predictions across repetitions
        model_data['prediction_mean'] = repetition_cols.mean(axis=1)

        # get those into one file for all models
        age_predictions_all[model_name] = model_data['prediction_mean']

        # get r value
        r_val = model_data['Age'].corr(model_data['prediction_mean'])
        print(model_name, r_val)

        # Add r value to r_val_df
        r_val_df[model_name] = [r_val]

    # Export age_predictions_all and r_val_df as csv
    age_predictions_all.to_csv(experiment_dir / 'age_predictions_test_allmodels.csv')
    r_val_df.to_csv(experiment_dir / 'r_values_test_allmodels.csv')


if __name__ == '__main__':
    main()