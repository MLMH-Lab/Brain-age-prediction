"""Script to calculate the % variance explained from the PCA models"""

import pandas as pd
from joblib import load
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def main():
    pca_path = PROJECT_ROOT / 'outputs' / 'pca' / 'models'

    pca_name_ls = []
    for i_repetition in range(10):
        for i_fold in range(10):
            pca_name = f'{i_repetition:02d}_{i_fold:02d}_pca.joblib'
            pca_name_ls.append(pca_name)

    pca_var_ls = []
    for i_model in pca_name_ls:
        print(i_model)
        pca_model = load(pca_path / i_model)
        var_explained = pca_model.explained_variance_ratio_.sum()
        pca_var_ls.append(var_explained)

    pca_var_df = pd.DataFrame({'variance_explained':pca_var_ls})
    var_mean = pca_var_df['variance_explained'].mean()
    var_std = pca_var_df['variance_explained'].stdev()
    print(var_mean, var_std)

    file_name = 'pca_variance_explained.csv'
    file_path = PROJECT_ROOT / 'outputs' / 'pca'
    pca_var_df.to_csv(file_path / file_name)



if __name__ == '__main__':
    main()