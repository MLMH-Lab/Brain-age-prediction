""""""
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    i_n_subjects = 50
    n_bootstrap = 500
    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name

    ids_with_n_subjects_dir = experiment_dir / 'bootstrap_analysis' / ('{:02d}'.format(i_n_subjects))

    scores_classifier_dir = ids_with_n_subjects_dir / 'scores_classifier'
    scores_regressor_dir = ids_with_n_subjects_dir / 'scores'


    for i_bootstrap in range(n_bootstrap):
        scores_filename = ('boot_scores_{:04d}.npy'.format(i_bootstrap))

        regressor_scores = np.load(str(scores_regressor_dir / scores_filename))
        classifier_scores = np.load(str(scores_classifier_dir / scores_filename))




    pass



if __name__ == "__main__":
    main()
