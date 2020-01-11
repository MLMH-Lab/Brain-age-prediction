#!/usr/bin/env python3
"""Script to compare the CORR score (correlation between the prediction error and the subject age) from a
classifier and from a regressor (using an uniform sample with 50 pairs)."""
from pathlib import Path

import numpy as np

from utils import ttest_ind_corrected

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    i_n_subjects = 50
    ids_with_n_subjects_dir = experiment_dir / 'bootstrap_analysis' / f'{i_n_subjects:02d}'

    scores_classifier_dir = ids_with_n_subjects_dir / 'scores_classifier'
    scores_regressor_dir = ids_with_n_subjects_dir / 'scores'

    n_bootstrap = 1000

    regressor_corr_list = []
    classifier_corr_list = []
    for i_bootstrap in range(n_bootstrap):
        scores_filename = f'boot_scores_{i_bootstrap:04d}.npy'
        regressor_scores = np.load(str(scores_regressor_dir / scores_filename))
        classifier_scores = np.load(str(scores_classifier_dir / scores_filename))

        # Store CORR score (position 3 in list of scores)
        regressor_corr_list.append(regressor_scores[3])
        classifier_corr_list.append(classifier_scores[3])

    # Check if CORR score is significantly different between regressor and classifier.
    _, pvalue = ttest_ind_corrected(np.array(regressor_corr_list), np.array(classifier_corr_list))
    print(f'CORR from classifier vs. CORR from regressor - pvalue: {pvalue:6.3}')


if __name__ == '__main__':
    main()
