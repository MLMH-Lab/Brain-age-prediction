# Comparison between machine learning methods

Here we verified the difference of the prediction performance between different machine learning approaches.
In our analysis, we included the most common methods used in the literature to predict the
 brain age following methods:

Using voxel-based data:
1. [Support Vector Machine]()
2. [Relevance Vector Machine]()

Using principal components from voxel-based data:
1. [Relevance Vector Machine]()

Using surface-based data:
1. [Support Vector Machine]()
2. [Relevance Vector Machine]()
3. [Gaussian process model]()

Each approach had their strong aspects and weakness.
Voxel-based data preserve most information of the raw data with minimal preprocessing.
However, this minimal preprocessing might include noise and unnecesary information for 
our task among the relevant features of the data. This unnecessary features can be
harmful for the performance of the models ("garbage in, garbage out"). This characteristic 
is specially notice in shalows machine learning methods (which is our case in this study).
For this reason, some feature engineering steps is commonly applied in studies. This feature
engineering can include feature selection, dimensionality reduction, feature extraction, etc.
Here, we performed a dimensionality reduction using the Principal Componets Analysis. Besides
this approach, we transformed our raw data using the surface-based morphometry analysis that
FeeSurfer software offers and worked with the features of the selected regions of interest.

In order to compared our approaches, we assessed all methods using the same individuals in the training
set and in the test set. These sets were defined using an resampling method callled 10 times 10-fold 
cross-validation that resulted in each model beeing evaluated 100 times. We chose this resampling method
to have a better estimate of each approach and avoid influences caused by chance (lucky selection of training set).

The metrics of performance were obtained from the test set (to avoid biased results (problem known as double dipping)),
and we used most common metrics from the literature:
- Mean Absolute Error
- Root Mean Squared Error   
- R-squared
- Correlaion between prediction error and age

Finally, we verify if these performance metrics significantly differ between approaches by using 
statistical test. Here, we used the corrected paired t-test to perform the hypothesis tests.
 