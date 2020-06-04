# Correlation between model predictions and covariates

Usually in brain age prediciton studies, it is interesting to investigated 
the correlation with the model prediction error with other variables of interest.
These variables could be demographic characteristics (e.g. education level), clinical
characteristics (e.g. clinical groups and clinical scores), or even geneteic characteristics.
In our study, we verified the correlation between our prediction and the demographic and 
enviromental characteristcs present in the UK Biobank data.

In order to do so, we followed the following method:

*FIGURE*

First, we collected the prediction on the test sets that was 
performed in the [comparison analysis](../comparison). Then we 
calculated the prediction used in this analysis, as the resulting
prediction of an ensemble of methods. In our case this ensemble was performed as
the mean of the ten predictions made about a particular subject. Note, for each 
subject we had only ten prediction because during each 10-fold cross-validation
a participant was present in the test set just once. So we have ten prediction about 
the subject beacause we used a 10 times 10-fold cross-validation.

As mention in previous studies, some machine learning regressors have
their predictions error influenced by the age. In order to remove this component 
to avoid biased results, we rgressed out the age from the error prediction 
and used the residualized values (known as brainager) in further analysis.

Finally, we computed the correlation coefficieent between the brainager and the 
covariates.
