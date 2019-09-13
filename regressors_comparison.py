"""Comparing classifiers using a version of the paired Student’s t-test that is
corrected for the violation of the independence assumption from repeated k-fold cross-validation
when training the model

Based on:
https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/

Bouckaert, Remco R., and Eibe Frank. "Evaluating the replicability of significance tests for comparing learning algorithms." Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, Berlin, Heidelberg, 2004.
https://github.com/BayesianTestsML/tutorial/blob/9fb0bf75b4435d61d42935be4d0bfafcc43e77b9/Python/bayesiantests.py
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path.cwd()


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


def main():
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    list_of_classifiers = ['SVM', 'RVM', 'GPR']

    # ----------------------------------------------------------------------------------------
    combinations = list(itertools.combinations(list_of_classifiers, 2))

    # Bonferroni correction for multiple comparisons
    corrected_alpha = 0.05 / len(combinations)

    results_df = pd.DataFrame(columns=['regressors', 'p-value', 'stats'])

    for classifier_a, classifier_b in combinations:
        classifier_a_dir = PROJECT_ROOT / 'outputs' / experiment_name / classifier_a
        classifier_b_dir = PROJECT_ROOT / 'outputs' / experiment_name / classifier_b

        mae_a = []
        mae_b = []

        n_repetitions = 10
        n_folds = 10

        for i_repetition in range(n_repetitions):
            for i_fold in range(n_folds):
                scores_filename = '{:02d}_{:02d}_scores.npy'.format(i_repetition, i_fold)

                performance_a = np.load(classifier_a_dir / 'cv' / scores_filename)[1]
                performance_b = np.load(classifier_b_dir / 'cv' / scores_filename)[1]

                mae_a.append(performance_a)
                mae_b.append(performance_b)

        statistic, pvalue = ttest_ind_corrected(np.asarray(mae_a), np.asarray(mae_b), k=n_folds, r=n_repetitions)

        print('{} vs. {} pvalue: {:6.3}'.format(classifier_a, classifier_b, pvalue), end='')
        if pvalue <= corrected_alpha:
            print('*')
        else:
            print('')

        results_df = results_df.append({'regressors': '{} vs. {}'.format(classifier_a, classifier_b),
                                        'p-value': pvalue,
                                        'stats': statistic},
                                       ignore_index=True)

        results_df.to_csv(PROJECT_ROOT / 'outputs' / experiment_name / 'regressors_comparison.csv', index=False)


if __name__ == "__main__":
    main()