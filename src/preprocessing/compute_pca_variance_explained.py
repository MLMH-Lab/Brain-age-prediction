"""Script to calculate the % variance explained from the PCA models"""

import pandas as pd
from joblib import load
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    pca_path = PROJECT_ROOT / 'outputs' / 'pca' / 'models'

    # Get list of file names for pca models
    n_repetitions = 10
    n_folds = 10

    pca_name_ls = []
    for i_repetition in range(n_repetitions):
        for i_fold in range(n_folds):
            pca_name = f'{i_repetition:02d}_{i_fold:02d}_pca.joblib'
            pca_name_ls.append(pca_name)

    # Loop over pca model file names, load models and get variance explained
    pca_var_ls = []
    for i_model in pca_name_ls:
        print(i_model)
        pca_model = load(pca_path / i_model)
        var_explained = pca_model.explained_variance_ratio_.sum()
        pca_var_ls.append(var_explained)

    # Create df for % variance explained per model iteration
    pca_var_df = pd.DataFrame({'variance_explained':pca_var_ls})

    # Get mean and standard deviation for % variance explained across iterations
    var_mean = pca_var_df['variance_explained'].mean()
    var_std = pca_var_df['variance_explained'].stdev()
    print(var_mean, var_std)

    # Save % variance explained per model
    file_name = 'pca_variance_explained.csv'
    file_path = PROJECT_ROOT / 'outputs' / 'pca'
    pca_var_df.to_csv(file_path / file_name)



if __name__ == '__main__':
    main()