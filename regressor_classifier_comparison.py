""""""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path.cwd()



def ttest_ind_corrected(a, b, k=10, r=10):
    """
        Corrected repeated k-fold cv test

                The test assumes that the classifiers were evaluated using cross
    validation. The number of folds is determined from the length of the vector
    of differences, as `len(diff) / runs`. The variance includes a correction
    for underestimation of variance due to overlapping training sets, as
    described in `Inference for the Generalization Error
    <http://link.springer.com/article/10.1023%2FA%3A1024068626366>`_,
    C. Nadeau and Y. Bengio, Mach Learning 2003.)


    n1 is the number of instances used  for training, and n2 the number of instances used for testing

    n2 / n1 = 1/ (nfolds-1) = 1/(k-1)
    r-times k-fold cross-validations there are r,r>1, runs and k,k>1, folds.

    Ref:
        Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004.


    Args:
        a: performances from classifier A
        b: performances from classifier A
        k: number of folds
        r: number of repetitions

    Returns:

    """
    df = k * r - 1

    x = a - b
    m = np.mean(x)

    sigma_2 = np.var(x, ddof=1)
    denom = np.sqrt((1 / k * r + 1 / (k - 1)) * sigma_2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(m, denom)

    prob = stats.t.sf(np.abs(t), df) * 2

    return t, prob


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

    regressor_corr_list = []
    classifier_corr_list = []
    for i_bootstrap in range(n_bootstrap):
        scores_filename = ('boot_scores_{:04d}.npy'.format(i_bootstrap))

        regressor_scores = np.load(str(scores_regressor_dir / scores_filename))
        classifier_scores = np.load(str(scores_classifier_dir / scores_filename))

        regressor_corr_list.append(regressor_scores[3])
        classifier_corr_list.append(classifier_scores[3])

    stat, pvalue = ttest_ind_corrected(np.array(regressor_corr_list), np.array(classifier_corr_list))


if __name__ == "__main__":
    main()
