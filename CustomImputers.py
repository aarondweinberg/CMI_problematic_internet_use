#To use the objects below, need the following modules:
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin



#Define two custom imputer objects.

## Define KNN custom imputer
class Custom_KNN_Imputer(BaseEstimator, TransformerMixin):
    # Class Constructor 
    # This allows you to initiate the class when you call Custom_KNN_Imputer
    def __init__(self):
        # I want to initiate each object with both a KNNImputer and StandardScaler object/method
        self.KNNImputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        self.StandardScaler = StandardScaler()

    
    # For my fit method I'm just going to "steal" KNNImputers's fit method using a curated collection of predictors
    def fit(self, X, y = None ):
        feature_list = X.columns.tolist()
        if 'id' in feature_list:
            feature_list.remove('id')
        if 'sii' in feature_list:
            feature_list.remove('sii')
        feature_list = [x for x in feature_list if 'PCIAT' not in x]
        feature_list = [x for x in feature_list if 'Zone' not in x]
        feature_list = [x for x in feature_list if 'Season' not in x]
        # Reset the index in X
        X = X.reset_index(drop=True)
        self.StandardScaler.fit(X[feature_list])
        # I'm never sure if we need the .values and/or .reshape(-1,1)
        #self.KNNImputer.fit(X[feature_list].values.reshape(-1,1))
        self.KNNImputer.fit(X[feature_list])
        return self
    
    # Now I want to transform the columns in feature list and return it with imputed values that have been un-transformed
    def transform(self, X, y = None):
        feature_list = X.columns.tolist()
        if 'id' in feature_list:
            feature_list.remove('id')
        if 'sii' in feature_list:
            feature_list.remove('sii')
        feature_list = [x for x in feature_list if 'PCIAT' not in x]
        feature_list = [x for x in feature_list if 'Zone' not in x]
        feature_list = [x for x in feature_list if 'Season' not in x]
        copy_X = X.copy().reset_index(drop=True)
        copy_X[feature_list] = self.KNNImputer.transform(copy_X[feature_list])
        copy_X2 = self.StandardScaler.inverse_transform(copy_X[feature_list])
        df2 = pd.DataFrame(copy_X2, columns=feature_list)
        copy_X[feature_list]=copy_X[feature_list].fillna(df2[feature_list])
        return copy_X




## Define MICE custom imputer
class Custom_MICE_Imputer(BaseEstimator, TransformerMixin):
    # Class Constructor 
    # This allows you to initiate the class when you call Custom_KNN_Imputer
    def __init__(self):
        # I want to initiate each object with both a KNNImputer and StandardScaler object/method
        self.MICEImputer = IterativeImputer(max_iter=10, random_state=497)

    
    # For my fit method I'm just going to "steal" IterativeImputers's fit method using a curated collection of predictors
    def fit(self, X, y = None, Z ):
        feature_list = Z.columns.tolist()
        if 'id' in feature_list:
            feature_list.remove('id')
        if 'sii' in feature_list:
            feature_list.remove('sii')
        feature_list = [x for x in feature_list if 'PCIAT' not in x]
        feature_list = [x for x in feature_list if 'Zone' not in x]
        feature_list = [x for x in feature_list if 'Season' not in x]
        Z = Z.reset_index(drop=True)
        self.MICEImputer.fit(Z[feature_list])
        return self
    
    # Now I want to transform the columns in feature list and return it with imputed values that have been un-transformed
    def transform(self, X, y = None, Z):
        feature_list = Z.columns.tolist()
        if 'id' in feature_list:
            feature_list.remove('id')
        if 'sii' in feature_list:
            feature_list.remove('sii')
        feature_list = [x for x in feature_list if 'PCIAT' not in x]
        feature_list = [x for x in feature_list if 'Zone' not in x]
        feature_list = [x for x in feature_list if 'Season' not in x]
        copy_Z = Z.copy()
        copy_Z = copy_Z.reset_index(drop=True)
        df2 = self.MICEImputer.transform(copy_Z[feature_list])
        df3 = pd.DataFrame(df2, columns=feature_list)
        copy_Z[feature_list]=copy_Z[feature_list].fillna(df3[feature_list])
        return copy_Z
    

####Now defining zone functions.

# Compute values for the 'FGC-FGC_SR_Zone' that is equal to 1 if any of the following are true:
# Basic_Demos-Sex==0 and FGC-FGC_SR >= 8
# Basic_Demos-Sex==1 and FGC-FGC_SR >= 9 and Basic_Demos-Age is between 5 and 10
# Basic_Demos-Sex==1 and FGC-FGC_SR >= 10 and Basic_Demos-Age is between 11 and 14
# Basic_Demos-Sex==1 and FGC-FGC_SR >= 12 and Basic_Demos-Age is at least 15
# Note that Basic_Demos-Sex is coded as 0=Male and 1=Female

def sitreachzone(sex, age, sr):
    try:
        if np.isnan(sr) or np.isnan(sex) or np.isnan(age):
            return np.nan
        elif sex == 0 and sr>=8:
            return 1
        elif sex == 1 and age >= 15 and sr >= 12:
            return 1
        elif sex == 1 and age >= 11 and sr >= 10:
            return 1
        elif sex == 1 and age >= 5 and sr >= 9:
            return 1
        else:
            return 0
    except:
        return np.nan

# Compute values for the 'FGC-FGC_CU_Zone' that is equal to 1 if any of the following are true:
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 2 and Basic_Demos-Age is between 5 and 6
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 4 and Basic_Demos-Age is 7
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 6 and Basic_Demos-Age is 8
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 9 and Basic_Demos-Age is 9
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 12 and Basic_Demos-Age is 10
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 15 and Basic_Demos-Age is 11
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 18 and Basic_Demos-Age is 12
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 21 and Basic_Demos-Age is 13
# Basic_Demos-Sex==0 and FGC-FGC_CU >= 24 and Basic_Demos-Age is at least 14
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 2 and Basic_Demos-Age is between 5 and 6
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 4 and Basic_Demos-Age is 7
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 6 and Basic_Demos-Age is 8
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 9 and Basic_Demos-Age is 9
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 12 and Basic_Demos-Age is 10
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 15 and Basic_Demos-Age is 11
# Basic_Demos-Sex==1 and FGC-FGC_CU >= 18 and Basic_Demos-Age is at least 12

def curlupzone(sex, age, cu):
    try:
        if np.isnan(sex) or np.isnan(age) or np.isnan(cu):
            return np.nan
        elif sex == 0:
            if (age >= 14 and cu >= 24) or (age == 13 and cu >= 21) or (age == 12 and cu >= 18) or (age == 11 and cu >= 15) or (age == 10 and cu >= 12) or (age == 9 and cu >= 9) or (age == 8 and cu >= 6) or (age == 7 and cu >= 4) or (age <= 6 and cu >= 2):
                return 1
            else:
                return 0
        elif sex == 1:
            if (age >= 12 and cu >= 18) or (age == 11 and cu >= 15) or (age == 10 and cu >= 12) or (age == 9 and cu >= 9) or (age == 8 and cu >= 6) or (age == 7 and cu >= 4) or (age <= 6 and cu >= 2):
                return 1
            else:
                return 0
    except:
        return np.nan

# Compute values for the 'FGC-FGC_PU_Zone' that is equal to 1 if any of the following are true:
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 3 and Basic_Demos-Age is between 5 and 6
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 4 and Basic_Demos-Age is 7
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 5 and Basic_Demos-Age is 8
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 6 and Basic_Demos-Age is 9
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 7 and Basic_Demos-Age is 10
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 8 and Basic_Demos-Age is 11
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 10 and Basic_Demos-Age is 12
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 12 and Basic_Demos-Age is 13
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 14 and Basic_Demos-Age is 14
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 16 and Basic_Demos-Age is 15
# Basic_Demos-Sex==0 and FGC-FGC_PU >= 18 and Basic_Demos-Age is at least 16
# Basic_Demos-Sex==1 and FGC-FGC_PU >= 3 and Basic_Demos-Age is between 5 and 6
# Basic_Demos-Sex==1 and FGC-FGC_PU >= 4 and Basic_Demos-Age is 7
# Basic_Demos-Sex==1 and FGC-FGC_PU >= 5 and Basic_Demos-Age is 8
# Basic_Demos-Sex==1 and FGC-FGC_PU >= 6 and Basic_Demos-Age is 9
# Basic_Demos-Sex==1 and FGC-FGC_PU >= 7 and Basic_Demos-Age is at least 10

def pullupzone(sex, age, pu):
    try:
        if np.isnan(sex) or np.isnan(age) or np.isnan(pu):
            return np.nan
        elif sex == 0:
            if (age >= 16 and pu >= 18) or (age == 15 and pu >= 16) or (age == 14 and pu >= 14) or (age == 13 and pu >= 12) or (age == 12 and pu >= 10) or (age == 11 and pu >= 8) or (age == 10 and pu >= 7) or (age == 9 and pu >= 6) or (age == 8 and pu >= 5) or (age == 7 and pu >= 4) or (age <= 6 and pu >= 2):
                return 1
            else:
                return 0
        elif sex == 1:
            if (age >= 10 and pu >= 7) or (age == 9 and pu >= 6) or (age == 8 and pu >= 5) or (age == 7 and pu >= 4) or (age <= 6 and pu >= 3):
                return 1
            else:
                return 0
    except:
        return np.nan

# Comtlte values for the 'FGC-FGC_TL_Zone' that is equal to 1 if any of the following are true:
# FGC-FGC_TL >= 6 and Basic_Demos-Age is between 5 and 9
# FGC-FGC_TL >= 9 and Basic_Demos-Age is at least 10

def tlzone(age, tl):
    try:
        if np.isnan(tl) or np.isnan(age):
            return np.nan
        elif (age >= 10 and tl >= 9) or (age <= 9 and tl >= 6):
            return 1
        else:
            return 0
    except:
        return np.nan

# Comtlte values for the 'PAQ_MVPA' that is equal to 1 if any of the following are true:
# PAQ_Total >= 2.73 and Basic_Demos-Age is between 5 and 13
# PAQ_Total >= 2.75 and Basic_Demos-Age is at least 14

def paqzone(age, paq):
    try:
        if np.isnan(paq) or np.isnan(age):
            return np.nan
        elif (age >= 14 and paq >= 2.75) or (age <= 13 and paq >= 2.73):
            return 1
        else:
            return 0
    except:
        return np.nan

###Custom encoder function
# 
def zone_encoder(df):
    df_copy = df.copy()

    if 'FGC-FGC_SR_Zone' in df_copy.columns:
        if 'Basic_Demos-Age' in df_copy.columns and 'Basic_Demos-Sex' in df_copy.columns and 'FGC-FGC_SR' in df_copy.columns:
            df_copy['FGC-FGC_SR_Zone'] = df_copy.apply(lambda x: sitreachzone(x['Basic_Demos-Sex'], x['Basic_Demos-Age'], x['FGC-FGC_SR']), axis=1)
        else:
            df_copy['FGC-FGC_SR_Zone'] = df_copy['FGC-FGC_SR_Zone'].fillna(df_copy['FGC-FGC_SR_Zone'].mean())
    if 'FGC-FGC_CU_Zone' in df_copy.columns:
        if 'Basic_Demos-Age' in df_copy.columns and 'Basic_Demos-Sex' in df_copy.columns and 'FGC-FGC_CU' in df_copy.columns:
            df_copy['FGC-FGC_CU_Zone'] = df_copy.apply(lambda x: curlupzone(x['Basic_Demos-Sex'], x['Basic_Demos-Age'], x['FGC-FGC_CU']), axis=1)
        else:
            df_copy['FGC-FGC_CU_Zone'] = df_copy['FGC-FGC_CU_Zone'].fillna(df_copy['FGC-FGC_CU_Zone'].mean())
    if 'FGC-FGC_PU_Zone' in df_copy.columns:
        if 'Basic_Demos-Age' in df_copy.columns and 'Basic_Demos-Sex' in df_copy.columns and 'FGC-FGC_PU' in df_copy.columns:
            df_copy['FGC-FGC_PU_Zone'] = df_copy.apply(lambda x: pullupzone(x['Basic_Demos-Sex'], x['Basic_Demos-Age'], x['FGC-FGC_PU']), axis=1)
        else:
            df_copy['FGC-FGC_PU_Zone'] = df_copy['FGC-FGC_PU_Zone'].fillna(df_copy['FGC-FGC_PU_Zone'].mean())
    if 'FGC-FGC_TL_Zone' in df_copy.columns:
        if 'Basic_Demos-Age' in df_copy.columns and 'FGC-FGC_TL' in df_copy.columns:
            df_copy['FGC-FGC_TL_Zone'] = df_copy.apply(lambda x: tlzone(x['Basic_Demos-Age'], x['FGC-FGC_TL']), axis=1)
        else:
            df_copy['FGC-FGC_TL_Zone'] = df_copy['FGC-FGC_TL_Zone'].fillna(df_copy['FGC-FGC_TL_Zone'].mean())
    if 'PAQ_Zone' in df_copy.columns:
        if 'Basic_Demos-Age' in df_copy.columns and 'PAQ_Total' in df_copy.columns:
            df_copy['PAQ_Zone'] = df_copy.apply(lambda x: tlzone(x['Basic_Demos-Age'], x['PAQ_Total']), axis=1)
        else:
            df_copy['PAQ_Zone'] = df_copy.apply(lambda x: paqzone(x['Basic_Demos-Age'], x['PAQ_Total']), axis=1)
    return df_copy    
