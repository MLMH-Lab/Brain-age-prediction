# Brain age prediction: A comparison between machine learning models using region- and voxel-based morphometric data
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/MLMH-Lab/Brain-age-prediction/blob/master/LICENSE)

Official script for the paper "Brain age prediction: A comparison between machine learning models using region- and voxel-based morphometric data".

## Abstract
Brain age prediction can be used to detect abnormalities in the ageing trajectory of an individual and their associated health issues. Existing studies on brain age vary widely in terms of their methods and type of data, so at present the most accurate and generalisable methodological approach is unclear. We used the UK Biobank dataset (N = 10,814) to compare the performance of the machine learning models support vector regression (SVR), relevance vector regression (RVR), and Gaussian process regression (GPR) on whole-brain region-based or voxel-based structural Magnetic Resonance Imaging data with or without dimensionality reduction through principal component analysis (PCA). Performance was assessed in the validation set through cross-validation as well as an independent test set. The models achieved mean absolute errors between 3.7 and 4.7 years, with those trained on voxel-level data with PCA performing best. There was little difference in performance between models trained on the same data type, indicating that the type of input data has greater impact on performance than model choice. Furthermore, dataset size analysis revealed that RVR required around half the sample size than SVR and GPR to yield generalisable results (approx. 120 subjects). Our results illustrated that the most suitable methodological approach for a brain age study depends on the sample size and the available computational and time resources. We are making all of our scripts open source in the hope that this will aid future research.

## Test our models online

## Citation
If you find this code useful for your research, please cite:

    Baecker L, Dafflon J, da Costa PF, Garcia-Dias R, Vieira S, Scarpazza C, Calhoun VD, Sato JR, Mechelli A*, Pinaya WHL* (in press). Brain age prediction: A comparison between machine learning models using region- and voxel-based morphometric data. Human Brain Mapping. * These authors contributed equally to this work
