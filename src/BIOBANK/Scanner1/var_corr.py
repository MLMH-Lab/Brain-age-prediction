"""Script to assess correlations between difference in actual and
predicted age with demographic variables in UK BIOBANK Scanner1

Variables to assess [variable code - variable name, code names (where applicable)]:
6138-2.0, 6138-2.1, 6138-2.2, 6138-2.3, 6138-2.4 - Which of the following qualifications do you have? (up to 5 selections possible)
1	College or University degree
2	A levels/AS levels or equivalent
3	O levels/GCSEs or equivalent
4	CSEs or equivalent
5	NVQ or HND or HNC or equivalent
6	Other professional qualifications eg: nursing, teaching
-7	None of the above
-3	Prefer not to answer
Researcher notes:
- CSE is the predecessor of GCSE, so can be treated the same
- NVQ/HND/HNC are work-based qualifications/degrees, comparable to short undergraduate degrees

24005-0.0 - Particulate matter air pollution (pm10); 2010
24009-0.0 - Traffic intensity on the nearest road
24010-0.0 - Inverse distance to the nearest road
24014-0.0 - Close to major road (binary)
24500-0.0 - Greenspace percentage
24501-0.0 - Domestic garden percentage
24502-0.0 - Water percentage
24506-0.0 - Natural environment percentage

Note: Baseline assessment data chosen for 24500-24506 because data is not available for all subjects at next assessment;
no data available for imaging assessment

Variable information available at https://biobank.ctsu.ox.ac.uk/crystal/label.cgi;
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')

def main():
    # Load SVR age predictions
    age_pred = pd.read_csv(PROJECT_ROOT / 'outputs/age_predictions.csv')

    # Add new columns as mean, median, std of age predictions + difference between actual age and mean, median
    pred_repetition = 10
    last_col = pred_repetition + 2
    age_pred['Mean predicted age'] = age_pred.iloc[:, 2:last_col].mean(axis=1)
    age_pred['Median predicted age'] = age_pred.iloc[:, 2:last_col].median(axis=1)
    age_pred['Std predicted age'] = age_pred.iloc[:, 2:last_col].std(axis=1)
    age_pred['Diff age-mean'] = age_pred['Age'] - age_pred['Mean predicted age']
    age_pred['Diff age-median'] = age_pred['Age'] - age_pred['Median predicted age']

    # Loading demographic data to access variables
    dataset_dem = pd.read_csv(str(PROJECT_ROOT / 'data' / 'BIOBANK'/ 'Scanner1' / 'ukb22321.csv'),
        usecols=['eid',
                 '6138-2.0', '6138-2.1', '6138-2.2', '6138-2.3', '6138-2.4',
                 '24005-0.0',
                 '24009-0.0', '24010-0.0', '24014-0.0',
                 '24500-0.0', '24501-0.0', '24502-0.0', '24506-0.0'])
    dataset_dem.columns = ['ID',
                           'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
                           'Air_pollution',
                           'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
                           'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Create new education col to simulate ordinal scale
    education_dict = {1:4, 2:2, 3:1, 4:1, 5:3, 6:3}
    dataset_dem['Education_1'] = dataset_dem['Education_1'].map(education_dict)
    dataset_dem['Education_2'] = dataset_dem['Education_2'].map(education_dict)
    dataset_dem['Education_3'] = dataset_dem['Education_3'].map(education_dict)
    dataset_dem['Education_4'] = dataset_dem['Education_4'].map(education_dict)
    dataset_dem['Education_5'] = dataset_dem['Education_5'].map(education_dict)
    dataset_dem['Education_highest'] = dataset_dem[['Education_1', 'Education_2', 'Education_3',
                                                   'Education_4', 'Education_5']].apply(max, axis=1)

    # Use maximum of education level per respondent

    # Spearman correlation per variable

    # Linear regression per variable

    # output csv with actual age, mean predicted age, median, std


if __name__ == "__main__":
    main()
