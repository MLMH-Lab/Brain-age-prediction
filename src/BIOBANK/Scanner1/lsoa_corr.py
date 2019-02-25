"""Script to assess correlations between age prediction error
and English index of multiple deprivation (IMD) variables"""

from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm

PROJECT_ROOT = Path('/home/lea/PycharmProjects/predicted_brain_age')


def ols_reg(df, x, y):
    """Perform linear regression using ordinary least squares (OLS) method"""

    endog = np.asarray(df[x], dtype=float)
    exog = np.asarray(sm.add_constant(df[y]), dtype=float)
    OLS_model = sm.OLS(endog, exog)
    OLS_results = OLS_model.fit()
    OLS_p = OLS_results.pvalues[1]
    OLS_coeff = OLS_results.params[1]

    alpha = 0.05
    n = len(df)
    if OLS_p < alpha:
        print('n=%s, %s and %s - reject H0: p = %.3f, rho = %.3f'
              % (n, x, y, OLS_p, OLS_coeff))
    # elif OLS_p >= alpha:
    #     print('%s and %s - fail to reject H0: p = %.3f, rho = %.3f'
    #           % (x, y, OLS_p, OLS_coeff))
    # else:
    #     print('Error with %s and %s' % (x, y))


def main():
    # Loading LSOA data merged with BIOBANK demographic data
    dataset = pd.read_csv(PROJECT_ROOT / 'data' / 'BIOBANK'/ 'Scanner1' / 'IMD_data.csv')
    col = list(dataset.columns[23:])

    for var in col:
        dataset_var = dataset.dropna(subset=[var])
        ols_reg(dataset_var, 'Diff_age-m', var)


if __name__ == "__main__":
    main()