# Correlation between model predictions and covariates

In brain age prediction studies, it is usually interesting to investigate the
correlation between the model prediction error with other variables of interest.
These variables can include demographic (e.g., education level), clinical (e.g.,
clinical groups and clinical scores), or even genetic information. In our study,
we verified the correlation between two covariates (i.e., demographic and
environmental characteristics) with the predicted age.

To do so, we followed the following method:
1. We collected the prediction on the
test sets used in the comparison analysis.
2. We built an ensemble using the mean
of the ten predictions made on the [comparison analysis](https://github.com/MLMH-Lab/Brain-age-prediction/tree/master/src/comparison) for each subject. We only
had 10 predictions because each subject was present in the test set just once.
As we performed a 10 times 10-fold-cross validation, we had had 10 predictions
per subject.
3. It is known that machine learning regressors have their
predictions error influenced by the mean age of the training sample. Therefore
to avoid biased results, we regressed out the chronological from the predicted age and
used the residualised values (known as
[brainAGER](https://www.frontiersin.org/articles/10.3389/fnagi.2018.00317/full))
in further analysis.
4. We computed the correlation coefficient between the
brainAGER and the covariates.
