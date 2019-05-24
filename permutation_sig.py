"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np
import pandas as pd
import os

from sklearn.externals import joblib

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()




def main():
    # Create output subdirectory
    experiment_name = 'total'

    # Define number of repetitions and folds used in SVM models
    n_repetitions = 10
    n_folds = 10

    n_perm = 1000
    n_features = len(COLUMNS_NAME)
    n_scores = 3 # r2, MAE, RMSE

    # Directory where SVM models are saved
    model_output_dir = PROJECT_ROOT / 'outputs' / experiment_name

    # Directory where permutation scores and coefficients are saved (2 files per permutation)
    perm_dir = PROJECT_ROOT / 'outputs' / experiment_name / 'permutations'

    # Significance level alpha with Bonferroni correction
    bonferroni_alpha = 0.05 / n_perm

    # Assessing significance
    perm_coefs = []
    perm_scores = []
    for i_perm in range(n_perm):

        filepath_coef = perm_dir / ('perm_coef_{:04d}.npy'.format(i_perm))
        filepath_scores = perm_dir / ('perm_scores_{:04d}.npy'.format(i_perm))

        try:
            perm_coefs.append(np.load(filepath_coef))
        except FileNotFoundError:
            print('File not found: {:}'.format(filepath_coef))

        try:
            perm_scores.append(np.load(filepath_scores))
        except FileNotFoundError:
            print('File not found: {:}'.format(filepath_scores))

    perm_scores = np.asarray(perm_scores, dtype='float32')
    perm_coefs = np.asarray(perm_coefs, dtype='float32')

    # TODO: Read assessed model values
    # Calculate proportion of permutation scores higher than model scores out of all permutations (p value)
    scores_pval = []
    for i in range(1, model_scores_mean.shape[0]):
        print(i)
        pval = (np.sum(model_scores_mean[i] >= perm_scores[:,i]) + 1.0) / (perm_scores.shape[0] + 1)
        scores_pval.append(pval)
        print(pval)

    # Assess significance with Bonferroni correction
    scores_pval_array = np.array(scores_pval)
    scores_sig = scores_pval_array < bonferroni_alpha

    # Save as csv
    scores_csv = pd.DataFrame([model_scores_mean, scores_pval, scores_sig],
                              columns=['R2', 'MAE', 'RMSE'],
                              index=['score', 'p', 'significance'])
    scores_csv.to_csv(str(model_output_dir / 'scores_sig.csv'))


    # ASSESSING SIGNIFICANCE OF FEATURE COEFFICIENTS

    # Create a list of all SVM models that were run to be able to access coefficients
    list_model_files = []
    for rep in range(n_repetitions):
        for fold in range(n_folds):
            list_model_files.append(str(rep) + '_' + str(fold) + '_svm.joblib')

    # Load SVM model coef into array model_coef
    model_coef = np.zeros([n_models, n_features])

    index = 0
    for mod in list_model_files:
        if os.path.isfile(str(model_output_dir / mod)):
            model = joblib.load(PROJECT_ROOT / 'outputs' / subjects / mod)
            model_coef[index] = model.coef_
        else:
            print("File not found: %s" % mod)
        index += 1

    # Get mean of absolute SVM model coefficients
    model_coef_abs = np.abs(model_coef)
    model_coef_mean = np.mean(model_coef_abs, axis=0)

    # Load permutation coefficients into array perm_coef
    perm_coef = []
    for i in range(n_perm):
        coef_file_name = 'perm_coef_%04d.npy' % i
        if os.path.isfile(str(perm_output_dir / coef_file_name)):
            perm_coef.append(np.load(perm_output_dir / coef_file_name))
        else:
            print('File not found: perm_coef_%04d.npy' % i)

    perm_coef = np.asarray(perm_coef, dtype='float32')

    # Calculate p-value per feature and store in coef_pval
    coef_pval = []
    for i in range(model_coef_mean.shape[0]):
        pval = (np.sum(perm_coef[:, i] >= model_coef_mean[i]) + 1.0) / (882 + 1)
        coef_pval.append(pval)

    # Assess significance with Bonferroni correction
    coef_p_array = np.array(coef_pval)
    coef_sig_array = coef_p_array < bonferroni_alpha

    # Save as csv
    coef_csv = pd.DataFrame([model_coef_mean, coef_p_array, coef_sig_array],
                            columns=COLUMNS_NAME,
                            index=['coefficient', 'pval', 'significance'])
    coef_csv = coef_csv.transpose()
    coef_csv = coef_csv.sort_values(by=['pval'])
    coef_csv.to_csv(model_output_dir / 'coef_sig.csv')


if __name__ == "__main__":
    main()