"""Script to create csv file with mean predictions across model repetitions"""

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

    # Loop over all models, calculate mean predictions across repetitions
    for model_name in model_ls:
        model_dir = experiment_dir / model_name
        file_name = model_dir / 'age_predictions_test.csv'
        try:
            model_data = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f'No age prediction file for {model_name}.')
            raise

        repetition_cols = model_data.loc[:,
                        'Prediction 00_00' : 'Prediction 09_09']

        # get mean predictions across repetitions
        model_data['prediction_mean'] = repetition_cols.mean(axis=1)

        # get those into one file for all models
        age_predictions_all[model_name] = model_data['prediction_mean']

    # Calculate brainAGE for all models and add to age_predictions_all df
    # brainAGE = predicted age - chronological age
    for model_name in model_ls:
        brainage_model = age_predictions_all[model_name] - \
                         age_predictions_all['Age']
        brainage_col_name = model_name + '_brainAGE'
        age_predictions_all[brainage_col_name] = brainage_model

    # Export age_predictions_all as csv
    age_predictions_all.to_csv(experiment_dir / 'age_predictions_test_allmodels.csv')

if __name__ == '__main__':
    main()