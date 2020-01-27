"""Helper functions and constants."""
import pandas as pd
import numpy as np
from scipy import stats


def load_pca_dataset(participants_path, ids_path, pca_path):
    pass


def load_freesurfer_dataset(participants_path, ids_path, freesurfer_path):
    """Load dataset."""
    demographic_data = load_demographic_data(participants_path, ids_path)

    freesurfer_df = pd.read_csv(freesurfer_path)

    dataset_df = pd.merge(freesurfer_df, demographic_data, on='Image_ID')

    return dataset_df


def load_demographic_data(participants_path, ids_path):
    """Load dataset using selected ids."""
    participants_df = pd.read_csv(participants_path, sep='\t')
    participants_df = participants_df.dropna()

    ids_df = pd.read_csv(ids_path, usecols=['Image_ID'])

    ids_df['participant_id'] = ids_df['Image_ID'].str.split('_').str[0]

    dataset_df = pd.merge(ids_df, participants_df, on='participant_id')

    return dataset_df


def ttest_ind_corrected(performance_a, performance_b, k=10, r=10):
    """Corrected repeated k-fold cv test.
     The test assumes that the classifiers were evaluated using cross validation.

    Ref:
        Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning
         algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004

    Args:
        performance_a: performances from classifier A
        performance_b: performances from classifier B
        k: number of folds
        r: number of repetitions

    Returns:
         t: t-statistic of the corrected test.
         prob: p-value of the corrected test.
    """
    df = k * r - 1

    x = performance_a - performance_b
    m = np.mean(x)

    sigma_2 = np.var(x, ddof=1)
    denom = np.sqrt((1 / k * r + 1 / (k - 1)) * sigma_2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(m, denom)

    prob = stats.t.sf(np.abs(t), df) * 2

    return t, prob


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
