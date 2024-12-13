#First we load basic packages and import data.

import numpy as np
import pandas as pd
import os

#This is the starting data.

train=pd.read_csv('CSV_Files_For_Modeling/train_original.csv')
test=pd.read_csv('CSV_Files_For_Modeling/test_original.csv')

####################################
#Next we add in the accelerometer data.

# Specify cutoffs

#ENMO cutoffs in mg for MVPA
mvpa_cutoff1 = 0.192
mvpa_cutoff2 = 0.110

# Number of 'active' bouts required for a day to count as 'active'
active_bout_cutoff = 150

# Specify the length of the bouts
boutlength = '5min'

# Maximum number of 5-minute bouts that can be imputed as zeroes to account for the accelerometer not collected data when at rest
impute_max = 6

# Minimum number of 5-second intervals (within a 5-minute bout) that need to have data for the bout to be counted
impute_sec_min = 29

# Create a new data frame with columns 'ID', 'ENMO_Avg_Active_Days_MVPA192', 'ENMO_Avg_Active_Days_MVPA110', and 'Positive_Anglez_Active_Days
accel = pd.DataFrame(columns=['ID', 'ENMO_Avg_Active_Days_MVPA192', 'ENMO_Avg_Active_Days_MVPA110', 'Positive_Anglez_Active_Days'])


# Walk through the files in the test directory
for dirname, _, filenames in os.walk('/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        #Check to see if filename is a parquet file; if it is, read the file and extract the ID from the directory name
        if filename.endswith('.parquet'):
            #Read the parquet file at dirname/filename
            data = pd.read_parquet(os.path.join(dirname, filename))
            id = dirname[-8:]

            # Remove any rows where the variable non-wear_flag is nonzero
            data = data[data['non-wear_flag'] == 0]

            # Change the time_of_day variable to a datetime and make it into the index
            data['dt'] = pd.to_datetime(data['time_of_day'])
            data['dt_mod'] = data['dt'] + pd.to_timedelta(data['relative_date_PCIAT'], unit='D')
            data.set_index('dt_mod', inplace=True)

            # Create a new data frame that counts the number of valid data points within each 5-minute ('boutlength') interval
            # This will later be used to exclude intervals that had fewer than 30 (out of 60) valid data points
            data['count'] = 1
            number_of_data_points = data.resample(boutlength).agg({'count':'sum'})
            data.drop('count', axis=1, inplace=True)

            # Create 5-minute "bouts" of averaged data and incorporate the number of valid data points within each interval as a new variable 'count'
            data_resampled_5min = data.resample(boutlength).mean()
            data_resampled_5min = data_resampled_5min.merge(number_of_data_points, left_index=True, right_index=True)

            # Some of the accelerometers stopped collecting data if they were stationary (but still on/worn)
            # This next section is an attempt to identify and fill in these seemingly missing values with "0" for the enmo value
            # It does this by identifying the length of each sequence of NaN values and filling them with 0 if thery are at most 30 minutes long
            # This also restricts this process to 5-minute bouts that had data for at least 30 of the 5-second-intervals within the bout
            data_resampled_5min['enmogroup'] = data_resampled_5min['enmo'].notna().cumsum()
            enmogroupcount = data_resampled_5min.groupby(by=["enmogroup"]).size().to_frame()
            enmogroupcount = enmogroupcount.rename(columns={0: 'enmogroupsize'})
            data_resampled_5min = data_resampled_5min.merge(enmogroupcount, how='left', left_on='enmogroup', right_index=True)
            data_resampled_5min['smallinterval'] = (data_resampled_5min['enmogroupsize'] < impute_max+2) & (data_resampled_5min['count']>impute_sec_min)
            data_resampled_5min['filled_enmo'] = np.where(data_resampled_5min.smallinterval, data_resampled_5min.enmo.fillna(0), data_resampled_5min.enmo)

            # Also fill in only anglez values where the count is large enough
            data_resampled_5min['filled_anglez'] = np.where(data_resampled_5min['count']>impute_sec_min, data_resampled_5min.anglez, np.nan)

            # The next code chunk will create a new data frame that lists the total number of valid bouts for the participant
            # and will count the number of bouts with filled_enmo values over a particular threshold
            # and then count the number of bouts with positive anglez values

            # Start by counting the number of valid bouts in each day as a data frame
            boutcount_filled = data_resampled_5min.groupby(data_resampled_5min.index.date).count()['filled_enmo'].to_frame()
            boutcount_filled = boutcount_filled.rename(columns={'filled_enmo': 'valid_bouts'})

            # Count the number of bouts in each day with filled_enmo at least mvpa_cutoff1
            boutcount_MVPA1 = data_resampled_5min[data_resampled_5min['filled_enmo'] >= mvpa_cutoff1].groupby(data_resampled_5min[data_resampled_5min['filled_enmo'] >= mvpa_cutoff1].index.date).count()['filled_enmo'].to_frame()
            boutcount_MVPA1 = boutcount_MVPA1.rename(columns={'filled_enmo': 'MVPA_bouts_over_cutoff1'})
            boutcount = boutcount_filled.merge(boutcount_MVPA1, how='left', left_index=True, right_index=True)

            # Count the number of bouts in each day with filled_enmo at least mvpa_cutoff2
            boutcount_MVPA2 = data_resampled_5min[data_resampled_5min['filled_enmo'] >= mvpa_cutoff2].groupby(data_resampled_5min[data_resampled_5min['filled_enmo'] >= mvpa_cutoff2].index.date).count()['filled_enmo'].to_frame()
            boutcount_MVPA2 = boutcount_MVPA2.rename(columns={'filled_enmo': 'MVPA_bouts_over_cutoff2'})
            boutcount = boutcount.merge(boutcount_MVPA2, how='left', left_index=True, right_index=True)

            # Count the number of bouts in each day with anglez at least 0
            boutcount_anglez = data_resampled_5min[data_resampled_5min['filled_anglez'] > 0].groupby(data_resampled_5min[data_resampled_5min['filled_anglez'] > 0].index.date).count()['filled_anglez'].to_frame()
            boutcount_anglez = boutcount_anglez.rename(columns={'filled_anglez': 'Positive_Anglez_Bouts'})
            boutcount = boutcount.merge(boutcount_anglez, how='left', left_index=True, right_index=True)

            # Compute a new variable 'included_day' to be True if valid_bouts is at least active_bout_cutoff
            boutcount['included_day'] = boutcount['valid_bouts'] >= active_bout_cutoff

            # Compute the mean of MVPA bouts over each cutoff
            # Note: We are only using the "included day" data in our final analysis, so we'll restrict the output accordingly
            MVPA_mean1 = boutcount[boutcount['included_day'] == True]['MVPA_bouts_over_cutoff1'].mean()
            MVPA_mean2 = boutcount[boutcount['included_day'] == True]['MVPA_bouts_over_cutoff2'].mean()
            Anglez_mean1 = boutcount[boutcount['included_day'] == True]['Positive_Anglez_Bouts'].mean()

            #Create a new row in the accel dataframe where 'ID'=id, 'ENMO_Avg_Active_Days_MVPA192'=MVPA_mean1, 'ENMO_Avg_Active_Days_MVPA110'=MVPA_mean2, and 'Positive_Anglez_Active_Days'=Anglez_mean1
            accel = pd.concat([accel, pd.DataFrame([id, MVPA_mean1, MVPA_mean2, Anglez_mean1])], ignore_index=True)

    
# Join train/test and accel  on the 'id' column and accel on the 'ID' column
#train = train.join(accel.set_index('ID'), on='id', how='left')
test = test.join(accel.set_index('ID'), on='id', how='left')

###################################
#Now we do a variety of data cleaning.
# Create a new variable 'FGC-FGC_SR' that is the mean of FGC-FGC_SRL and FGC-FGC_SRR
#train['FGC-FGC_SR'] = train[['FGC-FGC_SRL', 'FGC-FGC_SRR']].mean(axis=1)
test['FGC-FGC_SR'] = test[['FGC-FGC_SRL', 'FGC-FGC_SRR']].mean(axis=1)

# Remove the old sit & reach variables
#train = train.drop(columns=['FGC-FGC_SRL', 'FGC-FGC_SRR', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone'])
test = test.drop(columns=['FGC-FGC_SRL', 'FGC-FGC_SRR', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone'])

# Create a new variable 'FGC-FGC_SR_Zone' that is equal to 1 if any of the following are true:
# Basic_Demos-Sex==0 and FGC-FGC_SR >= 8
# Basic_Demos-Sex==1 and FGC-FGC_SR_Zone >= 9 and Basic_Demos-Age is between 5 and 10
# Basic_Demos-Sex==1 and FGC-FGC_SR_Zone >= 10 and Basic_Demos-Age is between 11 and 14
# Basic_Demos-Sex==1 and FGC-FGC_SR_Zone >= 12 and Basic_Demos-Age is at least 15

# One way to do this is to define a function that would take sex, age, and SR value as inputs and output 1 or 0
def sitreachzone(sex, age, sr):
    try:
        if np.isnan(sr):
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

# Apply sitreachzone to create a new column using the columns Basic_Demos-Sex, Basic_Demos-Age, and FGC-FGC_SR as inputs
#train['FGC-FGC_SR_Zone'] = train.apply(lambda x: sitreachzone(x['Basic_Demos-Sex'], x['Basic_Demos-Age'], x['FGC-FGC_SR']), axis=1)
test['FGC-FGC_SR_Zone'] = test.apply(lambda x: sitreachzone(x['Basic_Demos-Sex'], x['Basic_Demos-Age'], x['FGC-FGC_SR']), axis=1)

# Create a new variable that is 1 when PAQA/C Total is at least 2.75/2.73, 0 if it's less than these cutoffs, and NaN if PAQA/C is NaN
#train['PAQA_Zone'] = np.where(train['PAQ_A-PAQ_A_Total']>=2.75, 1, 0)
#train['PAQA_Zone'] = np.where(train['PAQ_A-PAQ_A_Total'].isnull(), np.nan, train['PAQA_Zone'])
test['PAQA_Zone'] = np.where(test['PAQ_A-PAQ_A_Total']>=2.75, 1, 0)
test['PAQA_Zone'] = np.where(test['PAQ_A-PAQ_A_Total'].isnull(), np.nan, test['PAQA_Zone'])

#train['PAQC_Zone'] = np.where(train['PAQ_C-PAQ_C_Total']>=2.73, 1, 0)
#train['PAQC_Zone'] = np.where(train['PAQ_C-PAQ_C_Total'].isnull(), np.nan, train['PAQC_Zone'])
test['PAQC_Zone'] = np.where(test['PAQ_C-PAQ_C_Total']>=2.73, 1, 0)
test['PAQC_Zone'] = np.where(test['PAQ_C-PAQ_C_Total'].isnull(), np.nan, test['PAQC_Zone'])

# Create new variables that merge the three PAQA/C variables
#train['PAQ_Total']=train['PAQ_C-PAQ_C_Total']
#train.loc[train['PAQ_Total'].isnull(),'PAQ_Total']=train['PAQ_A-PAQ_A_Total']
test['PAQ_Total']=test['PAQ_C-PAQ_C_Total']
test.loc[test['PAQ_Total'].isnull(),'PAQ_Total']=test['PAQ_A-PAQ_A_Total']

#train['PAQ_Season']=train['PAQ_C-Season']
#train.loc[train['PAQ_Season'].isnull(),'PAQ_Season']=train['PAQ_A-Season']
test['PAQ_Season']=test['PAQ_C-Season']
test.loc[test['PAQ_Season'].isnull(),'PAQ_Season']=test['PAQ_A-Season']

#train['PAQ_Zone']=train['PAQC_Zone']
#train.loc[train['PAQ_Zone'].isnull(),'PAQ_Zone']=train['PAQA_Zone']
test['PAQ_Zone']=test['PAQC_Zone']
test.loc[test['PAQ_Zone'].isnull(),'PAQ_Zone']=test['PAQA_Zone']

# Drop the PAQ variables we no longer need
#train=train.drop(columns=['PAQ_C-PAQ_C_Total', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season', 'PAQ_A-Season', 'PAQA_Zone', 'PAQC_Zone'])
test=test.drop(columns=['PAQ_C-PAQ_C_Total', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season', 'PAQ_A-Season', 'PAQA_Zone', 'PAQC_Zone'])

# Combine the minutes and seconds of Fitness_Endurance into a single number (total number of seconds)
#train['Fitness_Endurance_Total_Time_Sec'] = train['Fitness_Endurance-Time_Mins'] * 60 + train['Fitness_Endurance-Time_Sec']
test['Fitness_Endurance_Total_Time_Sec'] = test['Fitness_Endurance-Time_Mins'] * 60 + test['Fitness_Endurance-Time_Sec']

#train=train.drop(columns=['Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec'])
test=test.drop(columns=['Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec'])

# Remove the SDS-SDS_Total_T variable from train
#train=train.drop(columns=['SDS-SDS_Total_T'])
test=test.drop(columns=['SDS-SDS_Total_T'])

# Remove the FGC-FGC_GSND, FGC-FGC_GSND_Zone, FGC-FGC_GSD, and FGC-FGC_GSD_Zone variables
#train=train.drop(columns=['FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone'])
test=test.drop(columns=['FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone'])

# Create a list of numerical columns of type float. Note that these columns include the "Zone" variables which are really categorical/ordinal:
float_columns = train.select_dtypes(include=['float']).columns

# Change negative values to NaN
#train[train[float_columns] < 0] = np.nan
test[test[float_columns] < 0] = np.nan

# For each variable that starts with 'Physical-' replace any values that are 0 with NaN
#for column in train.columns:
#    if column.startswith('Physical-'):
#        train[column] = train[column].replace(0, np.nan)
for column in test.columns:
    if column.startswith('Physical-'):
        test[column] = test[column].replace(0, np.nan)

# For each column in float_columns, identify entries that are 5 standard deviations above or below the mean and replace them with NaN
for column in float_columns:
    #train[column] = train[column].mask(train[column] > train[column].mean() + 5 * train[column].std())
    #train[column] = train[column].mask(train[column] < train[column].mean() - 5 * train[column].std())
    test[column] = test[column].mask(test[column] > train[column].mean() + 5 * train[column].std())
    test[column] = test[column].mask(test[column] < train[column].mean() - 5 * train[column].std())

##Completed up through the cleaning.