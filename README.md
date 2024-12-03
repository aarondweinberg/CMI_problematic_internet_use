# Predicting Problematic Internet Use

## Team members
[Aaron Weinberg](https://github.com/aarondweinberg)  
[Emilie Wiesner](https://github.com/ewiesner)  
[Daniel Visscher](https://github.com/danielvisscher)

## Introduction
Internet use has been identified by researchers as having the potential to rise to the level of addiction, with associated increased rates of anxiety and depression. Identifying cases of problematic internet usage currently requires evaluation by an expert, however, which is a significant impediment to screening children and adolescents across society. One potential solution is to rely on an assortment of data that is more easily and uniformly collected: the kind collected by a family physician or by a smartwatch. The research question this project sets out to answer is: “Can we predict the level of problematic internet usage exhibited by children and adolescents, based on their physical activity and survey responses?” 

## Dataset
The data comes from a research study conducted by the Child-Mind Institute, with data for 3960 participants. The target variable is a severity impairment index (SII) that measures problematic internet use on a scale from 0 (no impairment) to 3 (severe impairment). There are 33 predictor variables, including demographics (e.g., age, sex), physical measurements often taken by a family physician (e.g., height, weight, blood pressure), results of a fitness test (e.g., sit & reach, endurance time), survey responses and scales (e.g., internet usage in hours per day, sleep disturbance, children’s global assessment), and measures from a bio-electric impedance analysis (e.g., bone mineral content, fat mass index). Additionally, about one month of accelerometer data was provided for almost one thousand of the participants in 5-second intervals. Significantly, participants often completed some parts of the study but not others, so any given participant will have entire groups of variables missing. 

## Preprocessing
Participants were dropped if they did not have an SII score (about a third of the participants). The distribution of remaining SII scores is very skewed: over half of participants had an SII score of 0, while only 30 (about 1% of) participants who were measured had an SII score of 3. For each input variable, we found benchmarks to identify extreme values and removed these. Because many variables had high correlation with other variables, we engaged in several rounds of feature reduction and identified important features using a random forest.

## Model Selection and Results
We compared models using Cohen’s kappa function. This function measures the accuracy of prediction for ordinal variables, with random guessing producing a score of 0, scores of 0.4-0.6 indicating moderate or fair agreement levels, and scores above 0.75 producing excellent agreement.
We designed an iterative imputer for missing predictor values, a function transformer to compute activity zones for various predictors, and an algorithm to sequentially apply classifiers to ordinal data. We then ran a pipe that first imputed and computed predictors, oversampled the sii=3 class using a Synthetic Minority Oversampling TEchnique, and made predictions using a variety of models. Cohen’s kappa indicated that using gradient boost to predict PCIAT scores and then modified bins to convert to sii scores had the best kappa score of 0.448.
For our final model we used a tuned gradient boosting regressor. This took our test data and used it to predict PCIAT values and then used custom-tuned bins to compute SII values. This resulted in a kappa value of 0.456, suggesting that our model avoided overfitting but was only able to come up with a moderate amount of agreement.

## Files
### Notebooks:
-	[`Accelerometer_Computations.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Accelerometer_Computations.ipynb) computes predictors from the actigraphy data
-	[`Data_Cleaning.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Data_Cleaning.ipynb) cleans the data
-	[`Feature_Importance.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Feature_Importance.ipynb) identifies a list of “key features” to use in multiple linear regression
-	[`Feature_Reduction.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Feature_Reduction.ipynb) removes highly-correlated and problematic predictors
-	[`Model_Selection.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Model_Selection.ipynb) tests out-of-the-box performance for a collection of models, tunes, and re-tests the models
-	[`Model_Final_Performance.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Model_Selection.ipynb) runs and analyzes our final model, a tuned gradient boosting regressor, on the reserved testing data
-	[`Outcome_Imputing.ipynb`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/Outcome_Imputing.ipynb) imputes missing values of PCIAT scores
###	Custom classes:
-	[`CustomImputers.py`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/CustomImputers.py) includes classes for doing our iterative imputing and computation of Zone predictors
-	[`OrdinalClassifier.py`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/OrdinalClassifier.py) includes a class that “wraps” a classifier in an algorithm that performs ordinal classification based on a method proposed by Frank and Hal (2001)
###	CSV files:
-	[`train_original.csv`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/train_original.csv) is the data file downloaded from Kaggle
-	`train` and `test` are an 80/20% split of train_original
-	[`train_cleaned.csv`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/CSV_Files_For_Modeling/train_cleaned.csv) is train after being processed by Data_Cleaning
-	[`train_cleaned_outcome_imputed.csv`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/CSV_Files_For_Modeling/train_cleaned_outcome_imputed.csv) is train_cleaned after being processed by Outcome_Imputing
-	[`train_cleaned_outcom_imputed_feature_selected.csv`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/CSV_Files_For_Modeling/train_cleaned_outcome_imputed_feature_selected.csv) is the result of processing by Feature_Reduction
-	[`Accelerometer_enmo_anglez_daily_averages.csv`](https://github.com/aarondweinberg/CMI_problematic_internet_use/blob/main/CSV_Files_For_Modeling/Accelerometer_enmo_anglez_daily_averages.csv) is the predictors generated from the actigraphy data