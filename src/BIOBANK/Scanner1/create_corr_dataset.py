"""Script to create dataset for var_corr and lsoa_corr"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def main():
    # Define what subjects dataset should contain: total, male or female
    subjects = 'total'

    # Define output subdirectory
    output_dir = PROJECT_ROOT / 'outputs' / subjects

    # Load SVR age predictions
    age_pred = pd.read_csv(output_dir / 'age_predictions.csv')

    # Add new columns as mean, median, std of age predictions + difference between actual age and mean, median
    pred_repetition = 10
    last_col = pred_repetition + 2
    age_pred['Mean_predicted_age'] = age_pred.iloc[:, 2:last_col].mean(axis=1)
    age_pred['Median_predicted_age'] = age_pred.iloc[:, 2:last_col].median(axis=1)
    age_pred['Std_predicted_age'] = age_pred.iloc[:, 2:last_col].std(axis=1)
    age_pred['BrainAGE_predmean'] = age_pred['Mean_predicted_age'] - age_pred['Age']
    age_pred['BrainAGE_predmedian'] = age_pred['Median_predicted_age'] - age_pred['Age']
    age_pred['Abs_BrainAGE_predmean'] = abs(age_pred['BrainAGE_predmean'])
    age_pred['Abs_BrainAGE_predmedian'] = abs(age_pred['BrainAGE_predmedian'])

    # Add new columns for BrainAGER (Brain Age Gap Estimate Residualized)
    brainager_model_predmean = sm.OLS(age_pred['Age'], age_pred['Mean_predicted_age'])
    brainager_results_predmean = brainager_model_predmean.fit()
    brainager_residuals_predmean = brainager_results_predmean.resid
    age_pred['BrainAGER_predmean'] = brainager_residuals_predmean

    brainager_model_predmedian = sm.OLS(age_pred['Age'], age_pred['Median_predicted_age'])
    brainager_results_predmedian = brainager_model_predmedian.fit()
    brainager_residuals_predmedian = brainager_results_predmedian.resid
    age_pred['BrainAGER_predmedian'] = brainager_residuals_predmedian

    age_pred['Abs_BrainAGER_predmean'] = abs(age_pred['BrainAGER_predmean'])
    age_pred['Abs_BrainAGER_predmedian'] = abs(age_pred['BrainAGER_predmedian'])

    # Extract participant ID
    age_pred['ID'] = age_pred['Participant_ID'].str.split('-', expand=True)[1]
    age_pred['ID'] = pd.to_numeric(age_pred['ID'])

    # Loading demographic data to access variables
    dataset_dem = pd.read_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'),
                              usecols=['eid',
                                       '6138-2.0', '6138-2.1', '6138-2.2', '6138-2.3', '6138-2.4',
                                       '22702-0.0', '22704-0.0',
                                       '24005-0.0',
                                       '24009-0.0', '24010-0.0', '24014-0.0',
                                       '24500-0.0', '24501-0.0', '24502-0.0', '24506-0.0'])
    dataset_dem.columns = ['ID',
                           'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
                           'East_coordinate', 'North_coordinate',
                           'Air_pollution',
                           'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
                           'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Create new education cols to simulate ordinal scale
    education_dict = {1: 4, 2: 2, 3: 1, 4: 1, 5: 3, 6: 3}
    dataset_dem['Education_1'] = dataset_dem['Education_1'].map(education_dict)
    dataset_dem['Education_2'] = dataset_dem['Education_2'].map(education_dict)
    dataset_dem['Education_3'] = dataset_dem['Education_3'].map(education_dict)
    dataset_dem['Education_4'] = dataset_dem['Education_4'].map(education_dict)
    dataset_dem['Education_5'] = dataset_dem['Education_5'].map(education_dict)

    # Create col for maximum of education level per respondent
    dataset_dem['Education_highest'] = dataset_dem[['Education_1', 'Education_2', 'Education_3',
                                                    'Education_4', 'Education_5']].apply(max, axis=1)

    # Merge age_pred and dataset_dem datasets
    dataset = pd.merge(age_pred, dataset_dem, on='ID')


    # output csv for polr in R
    dataset.to_csv(str(PROJECT_ROOT / 'outputs'/'age_predictions_demographics.csv'),
                   columns=['Participant_ID', 'Age', 'East_coordinate', 'North_coordinate',
                            'Mean_predicted_age', 'Median_predicted_age',
                            'Abs_BrainAGE_predmean', 'Abs_BrainAGE_predmedian',
                            'Abs_BrainAGER_predmean', 'Abs_BrainAGER_predmedian',
                            'BrainAGE_predmean', 'BrainAGE_predmean',
                            'BrainAGER_predmean', 'BrainAGER_predmedian',
                            'Std_predicted_age',
                            'Education_highest',
                            'Air_pollution',
                            'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
                            'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc'],
                   index=False)

    # output csv with actual age, mean predicted age, median, std
    dataset.to_csv(str(PROJECT_ROOT / 'outputs'/'age_predictions_stats.csv'),
                   columns=['Participant_ID', 'Age',
                            'Mean_predicted_age', 'Median_predicted_age',
                            'Abs_BrainAGE_predmean', 'Abs_BrainAGE_predmedian',
                            'Abs_BrainAGER_predmean', 'Abs_BrainAGER_predmedian',
                            'BrainAGE_predmean', 'BrainAGE_predmean',
                            'BrainAGER_predmean', 'BrainAGER_predmedian',
                            'Std_predicted_age'],
                   index=False)

if __name__ == "__main__":
    main()