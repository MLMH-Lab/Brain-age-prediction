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
    n_perm = 882 # 118 of the 1000 planned failed
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
    perm_scores = []
    counter = 0
    for i in range(n_perm):
        score_file_name = 'perm_scores_%04d.npy' % i
        if os.path.isfile(str(perm_output_dir / score_file_name)):
            perm_scores.append(np.load(perm_output_dir / score_file_name))
        else:
            counter = counter +1
            print('File not found: perm_scores_%04d.npy' % i)

    perm_scores = np.asarray(perm_scores, dtype='float32')

    # Calculate proportion of permutation scores higher than model scores out of all permutations (p value)
    scores_pval = []
    print('R_square')
    pval = (np.sum(model_scores_mean[0] <= perm_scores[0,i]) + 1.0) / (perm_scores.shape[0] + 1)
    scores_pval.append(pval)
    print(pval)

    for i in range(1, model_scores_mean.shape[0]):
        print(i)
        pval = (np.sum(model_scores_mean[i] >= perm_scores[:,i]) + 1.0) / (perm_scores.shape[0] + 1)
        scores_pval.append(pval)
        print(pval)

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