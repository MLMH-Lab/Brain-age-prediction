# Comparison between machine learning methods

Here, we assessed the difference of the prediction performance between different machine learning approaches
trained using voxel-based or region-based morphometric MRI data.
In our analysis, we included the most commonly used methods in the brain age literature:

Using voxel-based data:
1. [Support Vector Machine]()
2. [Relevance Vector Machine]()

Using principal components from voxel-based data:
1. [Support Vector Machine]()
2. [Relevance Vector Machine]()
3. [Gaussian process model]()

Using region-based data:
1. [Support Vector Machine]()
2. [Relevance Vector Machine]()
3. [Gaussian process model]()

Each approach has their advantages and weaknesses.
Voxel-based data preserve most information of the raw data with minimal preprocessing.
However, this minimal preprocessing might include noise and irrelevant information for 
the task of brain age prediction. The unnecessary features can be
harmful for the performance of the models (as implied in the common machine learning saying: 
"garbage in, garbage out"). The feature are especially harmful in shallow machine learning methods,
 as is the case in this study.
For this reason, some feature engineering steps are commonly applied. This feature
engineering can include feature selection, dimensionality reduction, feature extraction, etc.
Here, we performed a dimensionality reduction using the Principal Component Analysis (PCA). Besides
this approach, we transformed our raw data using the surface-based morphometry analysis that
FreeSurfer software offers and worked with the features of the 101 selected regions of interest.

In order to compare our approaches, we assess all methods using the same individuals in the training
set and in the test set. These sets were defined using an resampling method called 10 times 10-fold 
cross-validation (CV) that resulted in each model being evaluated 100 times. We chose this resampling method
to have a better estimate of each approach and avoid influences caused by chance (lucky selection of training set).

The metrics of performance were obtained from the test set (to avoid biased results, a problem known as double dipping),
and we used the most common brain age prediction metrics from the literature:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared
- Correlation between prediction error and age (or 'age bias')

Finally, we assess if these performance metrics significantly differ between approaches through
statistical testing. We used the corrected paired t-test to perform the hypothesis tests.
 