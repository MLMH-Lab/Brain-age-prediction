# Exploratory data analysis

The script for exploratory data analysis (EDA) can be applied to both training and independent test set.
It is useful to visually assess the age distribution in the sample as well as the sex distribution across ages
before further cleaning and preprocessing of the datasets.

Based on EDA, we decided to exclude ages with fewer than 99 subjects and also that it was necessary 
to process the dataset to be sex-homogeneous across ages to avoid introducing bias.
These data cleaning steps are performed in preprocessing/clean_data.py 
and preprocessing/homogenize_gender.py, respectively.