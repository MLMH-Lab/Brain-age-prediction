"""
Helper functions.
"""
import pandas as pd


def load_demographic_data(demographic_path, id_path):
    """Load dataset using selected ids."""

    if demographic_path.suffix == '.tsv':
        demographic_df = pd.read_csv(demographic_path, sep='\t')
    else:
        demographic_df = pd.read_csv(demographic_path)

    # if using UK Biobank supplementary data
    if 'eid' in demographic_df.columns:
        demographic_df = demographic_df[['eid', '31-0.0', '21000-0.0', '21003-2.0']]
        demographic_df.columns = ['ID', 'Gender', 'Ethnicity', 'Age']

        # Labeling data
        ethnicity_dict = {
            1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
            2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
            3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
            4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
            5: 'Chinese',
            6: 'Other',
            -1: 'Not known', -3: 'Not known'
        }

        gender_dict = {
            0: 'Female',
            1: 'Male'
        }

        demographic_df = demographic_df.replace({'Gender': gender_dict})
        demographic_df = demographic_df.replace({'Ethnicity': ethnicity_dict})

    # if using participants.tsv file
    else:
        demographic_df['ID'] = demographic_df['Participant_ID'].str.split('-').str[1]

    demographic_df['ID'] = pd.to_numeric(demographic_df['ID'])
    demographic_df = demographic_df.dropna()

    id_df = pd.read_csv(id_path)
    # Create a new 'ID' column to match supplementary demographic data
    if 'Participant_ID' in id_df.columns:
        # For create_homogeneous_data.py output
        id_df['ID'] = id_df['Participant_ID'].str.split('-').str[1]
    else:
        # For freesurferData dataframe
        id_df['ID'] = id_df['Image_ID'].str.split('_').str[0]
        id_df['ID'] = id_df['ID'].str.split('-').str[1]

    id_df['ID'] = pd.to_numeric(id_df['ID'])

    # Merge supplementary demographic data with ids
    dataset = pd.merge(id_df, demographic_df, on='ID')

    if 'Participant_ID_y' in dataset.columns:
        dataset['Participant_ID'] = dataset['Participant_ID_x']
        dataset = dataset.drop(['Participant_ID_x', 'Participant_ID_y'], axis=1)

    return dataset

COLUMNS_NAME = ['Left-Lateral-Ventricle',
                'Left-Inf-Lat-Vent',
                'Left-Cerebellum-White-Matter',
                'Left-Cerebellum-Cortex',
                'Left-Thalamus-Proper',
                'Left-Caudate',
                'Left-Putamen',
                'Left-Pallidum',
                '3rd-Ventricle',
                '4th-Ventricle',
                'Brain-Stem',
                'Left-Hippocampus',
                'Left-Amygdala',
                'CSF',
                'Left-Accumbens-area',
                'Left-VentralDC',
                'Right-Lateral-Ventricle',
                'Right-Inf-Lat-Vent',
                'Right-Cerebellum-White-Matter',
                'Right-Cerebellum-Cortex',
                'Right-Thalamus-Proper',
                'Right-Caudate',
                'Right-Putamen',
                'Right-Pallidum',
                'Right-Hippocampus',
                'Right-Amygdala',
                'Right-Accumbens-area',
                'Right-VentralDC',
                'CC_Posterior',
                'CC_Mid_Posterior',
                'CC_Central',
                'CC_Mid_Anterior',
                'CC_Anterior',
                'lh_bankssts_volume',
                'lh_caudalanteriorcingulate_volume',
                'lh_caudalmiddlefrontal_volume',
                'lh_cuneus_volume',
                'lh_entorhinal_volume',
                'lh_fusiform_volume',
                'lh_inferiorparietal_volume',
                'lh_inferiortemporal_volume',
                'lh_isthmuscingulate_volume',
                'lh_lateraloccipital_volume',
                'lh_lateralorbitofrontal_volume',
                'lh_lingual_volume',
                'lh_medialorbitofrontal_volume',
                'lh_middletemporal_volume',
                'lh_parahippocampal_volume',
                'lh_paracentral_volume',
                'lh_parsopercularis_volume',
                'lh_parsorbitalis_volume',
                'lh_parstriangularis_volume',
                'lh_pericalcarine_volume',
                'lh_postcentral_volume',
                'lh_posteriorcingulate_volume',
                'lh_precentral_volume',
                'lh_precuneus_volume',
                'lh_rostralanteriorcingulate_volume',
                'lh_rostralmiddlefrontal_volume',
                'lh_superiorfrontal_volume',
                'lh_superiorparietal_volume',
                'lh_superiortemporal_volume',
                'lh_supramarginal_volume',
                'lh_frontalpole_volume',
                'lh_temporalpole_volume',
                'lh_transversetemporal_volume',
                'lh_insula_volume',
                'rh_bankssts_volume',
                'rh_caudalanteriorcingulate_volume',
                'rh_caudalmiddlefrontal_volume',
                'rh_cuneus_volume',
                'rh_entorhinal_volume',
                'rh_fusiform_volume',
                'rh_inferiorparietal_volume',
                'rh_inferiortemporal_volume',
                'rh_isthmuscingulate_volume',
                'rh_lateraloccipital_volume',
                'rh_lateralorbitofrontal_volume',
                'rh_lingual_volume',
                'rh_medialorbitofrontal_volume',
                'rh_middletemporal_volume',
                'rh_parahippocampal_volume',
                'rh_paracentral_volume',
                'rh_parsopercularis_volume',
                'rh_parsorbitalis_volume',
                'rh_parstriangularis_volume',
                'rh_pericalcarine_volume',
                'rh_postcentral_volume',
                'rh_posteriorcingulate_volume',
                'rh_precentral_volume',
                'rh_precuneus_volume',
                'rh_rostralanteriorcingulate_volume',
                'rh_rostralmiddlefrontal_volume',
                'rh_superiorfrontal_volume',
                'rh_superiorparietal_volume',
                'rh_superiortemporal_volume',
                'rh_supramarginal_volume',
                'rh_frontalpole_volume',
                'rh_temporalpole_volume',
                'rh_transversetemporal_volume',
                'rh_insula_volume']
