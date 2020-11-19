"""Analysis of BrainAGER differences between men and women"""

import argparse
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.api as sm
from pathlib import Path
from utils import load_demographic_data

PROJECT_ROOT = Path.cwd()

parser = argparse.ArgumentParser()

parser.add_argument('-E', '--experiment_name',
                    dest='experiment_name',
                    help='Name of the experiment.')

parser.add_argument('-S', '--scanner_name',
                    dest='scanner_name',
                    help='Name of the scanner.')

args = parser.parse_args()


def main(experiment_name, scanner_name):
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    # Load file containing brainAGE and brainAGER scores for all subjects
    # from all models
    brainage_allmodels = pd.read_csv(
        experiment_dir / 'age_predictions_allmodels_brainager.csv',
        index_col = 0)

    # Access demographics file and add sex variable to brainage_allmodels
    participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / \
                        scanner_name / 'participants.tsv'

    ids_path = experiment_dir / 'homogenized_ids.csv'
    demographics = load_demographic_data(participants_path, ids_path)
    demographics_sex = demographics[['image_id', 'Gender']]
    brainage_sex = brainage_allmodels.merge(demographics_sex, how='left',
                                            on='image_id')

    # List of models to loop over
    model_ls = ['SVM', 'RVM', 'GPR',
                'voxel_SVM', 'voxel_RVM',
                'pca_SVM', 'pca_RVM', 'pca_GPR']

    # Get mean brainAGER per sex per model
    # and test for significant differences using a t-test
    male_code = 1
    female_code = 0
    male_subjects = brainage_sex.groupby('Gender').get_group(male_code)
    female_subjects = brainage_sex.groupby('Gender').get_group(female_code)

    for model in model_ls:
        brainager_model = model + '_brainAGER'
        male_pred = male_subjects[brainager_model]
        female_pred = female_subjects[brainager_model]
        male_pred_mean = male_pred.mean()
        female_pred_mean = female_pred.mean()
        tstat, pval = ttest_ind(male_pred, female_pred)
        print(model, male_pred_mean, female_pred_mean, tstat, pval)


if __name__ == '__main__':
    main(args.experiment_name, args.scanner_name)