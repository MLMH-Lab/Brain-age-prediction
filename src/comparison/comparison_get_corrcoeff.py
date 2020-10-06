"""Script to calculate Pearson's r correlation coefficient
for all model predictions per subject and chronological age;

The script also creates a summary file of mean model predictions
for all subjects across model repetitions"""

import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    experiment_name = 'biobank_scanner1'
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # Create df with subject IDs and chronological age
    # All mean model predictions will be added to this df in the loop
    # Based on an age_predictions csv file from model training to have the
    # same order of subjects
    example_file = pd.read_csv(experiment_dir / 'SVM' / 'age_predictions.csv')
    age_predictions_all = pd.DataFrame(example_file.loc[:, 'image_id':'Age'])

    # Create df to hold r values for all models
    r_val_df = pd.DataFrame()

    # Loop over all models, calculate mean predictions across repetitions,
    # calculate r
    for model_name in model_ls:
        model_dir = experiment_dir / model_name
        file_name = model_dir / 'age_predictions.csv'

        try:
            model_data = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f'No age prediction file for {model_name}.')
            raise

        repetion_cols = model_data.loc[:,
                        'Prediction repetition 00' : 'Prediction repetition 09']

        # get mean predictions across repetitions
        model_data['prediction_mean'] = repetion_cols.mean(axis=1)

        # get those into one file for all models
        age_predictions_all[model_name] = model_data['prediction_mean']

        # get r value
        r_val = model_data['Age'].corr(model_data['prediction_mean'])
        print(model_name, r_val)

        # Add r value to r_val_df
        r_val_df[model_name] = [r_val]

    # Export age_predictions_all and r_val_df as csv
    age_predictions_all.to_csv(experiment_dir / 'age_predictions_allmodels.csv')
    r_val_df.to_csv(experiment_dir / 'r_values_allmodels.csv')

    # ----------------------------------------
    # Calculate age-BrainAGE correlation
    # #TODO: correct in comparison_statistical_analysis.py
    for model_name in model_ls:
        age_error_corr, _ = stats.spearmanr(
            (age_predictions_all[model_name] - age_predictions_all['Age']),
            age_predictions_all['Age'])
        print(model_name, age_error_corr)


if __name__ == '__main__':
    main()