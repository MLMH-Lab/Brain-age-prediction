"""Significance testing of SVM permutations for BIOBANK Scanner1"""

from pathlib import Path
import numpy as np
import pandas as pd
import os

from sklearn.externals import joblib

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')

def main():
    # Define what subjects were modeled: total, male or female
    subjects = 'total'

    # Define number of repetitions and folds used in SVM models
    n_repetitions = 10
    n_folds = 10
    n_models = n_repetitions * n_folds

    # Define number of subjects, number of permutations run, number of features, number of output scores
    n_subjects = 12190
    n_perm = 1000
    n_features = 101 # freesurfer regions
    n_scores = 3 # r2, MAE, RMSE

    # Directory where SVM models are saved
    model_output_dir = Path(str(PROJECT_ROOT / 'outputs' / subjects))

    # Directory where permutation scores and coefficients are saved (2 files per permutation)
    perm_output_dir = Path(str(PROJECT_ROOT / 'outputs' / 'permutations' / subjects))

    # Significance level alpha with Bonferroni correction
    bonferroni_alpha = 0.05 / n_perm


    # ASSESSING SIGNIFICANCE OF MODEL SCORES (R2, MAE, RMSE)

    # Create a list of all model score files
    list_model_score_files = []
    for rep in range(n_repetitions):
        for fold in range(n_folds):
            list_model_score_files.append(str(rep) + '_' + str(fold) + '_svm_scores.npy')

    # Load SVM model scores into array model_scores
    model_scores = np.zeros([n_models, n_scores])

    index = 0
    for score_file in list_model_score_files:
        if os.path.isfile(str(model_output_dir / score_file)):
            model_score = np.load(model_output_dir / score_file)
            model_scores[index] = model_score
        else:
            print("File not found: %s" % score_file)
        index += 1

    model_scores_abs = np.abs(model_scores)
    model_scores_mean = np.mean(model_scores_abs, axis=0)

    # Load permutation scores into array perm_scores
    perm_scores = np.zeros([n_perm, n_scores])

    for i in range(n_perm):
        score_file_name = 'perm_scores_%04d.npy' % i
        if os.path.isfile(str(perm_output_dir / score_file_name)):
            perm_scores[i] = np.load(perm_output_dir / score_file_name)
        else:
            print('File not found: perm_scores_%04d.npy' % i)

    # Calculate proportion of permutation scores higher than model scores out of all permutations (p value)
    scores_pval = []

    ind = 0
    for i_score in perm_scores.T:
        pval = (np.sum(i_score >= model_scores_mean[ind]) + 1.0) / (n_perm + 1)
        scores_pval.append(pval)
        ind += 1

    # Assess significance with Bonferroni correction
    scores_pval_array = np.array(scores_pval)
    scores_sig = scores_pval_array < bonferroni_alpha

    # Save as csv
    scores_csv = pd.DataFrame([model_scores_mean, scores_pval, scores_sig], columns=['R2', 'MAE', 'RMSE'],
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
    perm_coef = np.zeros([n_perm, n_features])

    for i in range(n_perm):
        coef_file_name = 'perm_coef_%04d.npy' % i
        if os.path.isfile(str(perm_output_dir / coef_file_name)):
            perm_coef[i] = np.load(perm_output_dir / coef_file_name)
        else:
            print('File not found: perm_coef_%04d.npy' % i)

    # Calculate p-value per feature and store in coef_p_list
    coef_p_list = []

    ind = 0
    for i_feat in perm_coef.T:
        pval = (np.sum(i_feat >= model_coef_mean[ind]) + 1.0) / (n_perm + 1)
        coef_p_list.append(pval)
        ind += 1

    # Assess significance with Bonferroni correction
    coef_p_array = np.array(coef_p_list)
    coef_sig_array = coef_p_array < bonferroni_alpha

    # Load FS data to access feature names
    fs_file_name = 'freesurferData_' + subjects + '.h5'
    dataset = pd.read_hdf(PROJECT_ROOT / 'data' / 'BIOBANK' / 'Scanner1' / fs_file_name, key='table')
    feature_names = np.array(dataset.columns[5:-1])

    # Save as csv
    coef_csv = pd.DataFrame([model_coef_mean, coef_p_array, coef_sig_array], columns=feature_names,
                            index=['coefficient', 'pval', 'significance'])
    coef_csv = coef_csv.transpose()
    coef_csv = coef_csv.sort_values(by=['pval'])
    coef_csv.to_csv(model_output_dir / 'coef_sig.csv')


if __name__ == "__main__":
    main()