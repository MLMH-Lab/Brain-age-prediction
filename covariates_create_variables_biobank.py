#!/usr/bin/env python3
"""Create dataset with demographic variables from UK Biobank for correlation analysis;

The following UK Biobank variables are assessed [variable data field - variable name]:
6138-2.0, 6138-2.1, 6138-2.2, 6138-2.3, 6138-2.4 - Which of the following qualifications do you have? (up to 5 selections)
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

Variable information available at https://biobank.ctsu.ox.ac.uk/crystal/label.cgi"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    demographic_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / 'ukb22321.csv'
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    correlation_dir = experiment_dir / 'correlation_analysis'
    correlation_dir.mkdir(exist_ok=True)

    # Loading demographic data in UK Biobank to access variables
    variables_df = pd.read_csv(demographic_path,
                               usecols=['eid',
                                        '6138-2.0', '6138-2.1', '6138-2.2', '6138-2.3', '6138-2.4',
                                        '24005-0.0',
                                        '24009-0.0', '24010-0.0', '24014-0.0',
                                        '24500-0.0', '24501-0.0', '24502-0.0', '24506-0.0'])
    variables_df.columns = ['ID',
                            'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
                            'Air_pollution',
                            'Traffic_intensity', 'Inverse_dist_road', 'Close_road_bin',
                            'Greenspace_perc', 'Garden_perc', 'Water_perc', 'Natural_env_perc']

    # Create new education cols to simulate ordinal scale
    # The original 6-point education scale was reduced to a 4-point scale using the following assumptions:
    # Codes 3 "O levels / GCSEs or equivalent" and 4 "CSEs or equivalent" are equivalent
    # Codes 5 "NVQ or HND or HNC or equivalent" and 6 "Other professional qualifications" are equivalent
    # Codes -7	"None of the above" and -3	"Prefer not to answer" are treated as missing data
    education_dict = {1: 4, 2: 2, 3: 1, 4: 1, 5: 3, 6: 3}
    variables_df['Education_1'] = variables_df['Education_1'].map(education_dict)
    variables_df['Education_2'] = variables_df['Education_2'].map(education_dict)
    variables_df['Education_3'] = variables_df['Education_3'].map(education_dict)
    variables_df['Education_4'] = variables_df['Education_4'].map(education_dict)
    variables_df['Education_5'] = variables_df['Education_5'].map(education_dict)

    # Create col for maximum of education level per respondent and drop original variables
    variables_df['Education_highest'] = variables_df[['Education_1',
                                                      'Education_2',
                                                      'Education_3',
                                                      'Education_4',
                                                      'Education_5']].apply(max, axis=1)
    variables_df = variables_df.drop(['Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5'],
                                     axis=1)

    # output csv with age variables and demographic variables
    variables_df.to_csv(correlation_dir / 'variables_biobank.csv', index=False)


if __name__ == '__main__':
    main()
