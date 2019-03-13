"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np
import pandas as pd

import os
import glob

from sklearn.externals import joblib

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')

def main():
    # Define what subjects were modeled: total, male or female
    subjects = 'test'

    # Define number of subjects, number of permutations run, number of features, number of output scores
    n_subjects = 12190
    n_perm = 1000 # for testing
    n_features = 101
    n_scores = 3 # r2, MAE, RMSE

    # Directory where permutation scores and coefficients are saved (2 files per permutation)
    perm_output_dir = Path(str(PROJECT_ROOT / 'outputs' / 'permutations' / subjects))

    # Significance level alpha with Bonferroni correction
    bonferroni_alpha = 0.05 / n_perm

    # Load permutation scores into array perm_scores
    perm_scores = np.zeros([n_perm, n_scores])

    for i in range(n_perm):
        score_file_name = 'perm_scores_%04d.npy' % i
        if os.path.isfile(str(perm_output_dir / score_file_name)):
            perm_scores[i] = np.load(perm_output_dir / score_file_name)
        else:
            print('File not found: perm_scores_%04d.npy' % i)

    # Load permutation coefficients into array perm_coef
    perm_coef = np.zeros([n_perm, n_features]) # why does it have to be +1?

    for i in range(n_perm):
        coef_file_name = 'perm_coef_%04d.npy' % i
        if os.path.isfile(str(perm_output_dir / coef_file_name)):
            perm_coef[i] = np.load(perm_output_dir / coef_file_name)
        else:
            print('File not found: perm_coef_%04d.npy' % i)

    # Define number of repetitions and folds used in SVM models
    n_repetitions = 10
    n_folds = 10
    n_models = n_repetitions * n_folds

    # Directory where models are saved
    model_output_dir = Path(str(PROJECT_ROOT / 'outputs' / subjects))

    # Create a list of all SVM models that were run
    list_model_files = []
    for rep in range(n_repetitions):
        for fold in range(n_folds):
            list_model_files.append(str(rep) + '_' + str(fold) + '_svm.joblib')

    # Load SVM model scores into array model_scores
    model_scores = np.zeros([n_models, n_scores])

    # TODO: svm files don't contain score output, so below not yet working

    model_scores_abs = np.abs(model_scores)
    model_scores_mean = np.mean(model_scores_abs, axis=0)
    scores_pval = (np.sum(perm_scores >= model_scores_mean) + 1.0) / (n_perm + 1)

    # Assess significance with Bonferroni correction
    scores_sig_array = scores_pval < bonferroni_alpha

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

    # Calculate p-value per feature #TODO: is not calculating pvalue per feature but per permutation
    coef_p_list = []

    for p in perm_coef: # should be referring to sth else
        print(p >= model_coef_mean)
        pval = (np.sum(p >= model_coef_mean) + 1.0) / (n_perm + 1)
        coef_p_list.append(pval)

    # Assess significance with Bonferroni correction
    coef_p_array = np.array(coef_p_list)
    coef_sig_array = coef_p_array < bonferroni_alpha

    # TODO: save scores and their pval as csv file

    # Load FS data to access feature names
    fs_file_name = 'freesurferData_' + subjects + '.h5'
    dataset = pd.read_hdf(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / fs_file_name, key='table')
    feature_names = np.array(dataset.columns[5:-1])

    coef_csv = pd.DataFrame([model_coef_mean, coef_p_array, coef_sig_array], columns=feature_names)
    coef_csv.to_csv(model_output_dir + 'coef_sig.csv')

    # p_list_df = pd.DataFrame(p_list, columns=['Feature', 'N_permcoef_higher', 'p_val'])

    # TODO: save as two separate csv files, include the feature names, model coef, sig level (bonferroni correction), sort by pval



if __name__ == "__main__":
    main()