# CMI_problematic_internet_use
Collaboration to investigate the Problematic Internet Use dataset from the Child Mind Institute

## Initial Project Description

The problem and data set we are working on are based on a data competition posted to the competition platform Kaggle.com.  The data comes from the Child Mind Institute, with the goal of developing a mechanism to predict and identify problematic internet usage among adolescents without the need for highly specialized evaluation by professionals.

The data provided includes 82 variables for 3960 participants. Variables include 
-	Demographic data (age, sex)
-	Physical characteristics (height, weight, waist, BMI, blood pressure, pulse rate)
-	Results of a fitness test (for example: grip strength, an endurance time measure)
-	Bio-electric Impedance Analysis results (for example: bone mineral content, body fat percentage)
-	Physical activity score (in response to a questionnaire)
-	Responses to an internet addiction survey 
-	Sleep disturbance score
-	Internet time usage score
-	Target variable: severity impairment index. (This categorical variable is based on the total score on the internet addiction survey.)

Additionally, some participants wore accelerometer data for one month. The data provided from these participants includes ENMO and light data at 5-second intervals.

There is considerable missing data in the set. We only have accelerometer data for XX participants. For non-demographic variables, we only have data from  740 to ~3000 participants (with wide variation.)

The primary stakeholder for this project is the Child Mind Institute. Secondary stakeholders could include parents, professionals (family doctors, teachers, counselors), and policy-makers who seek to identify or predict problematic internet usage among adolescents. 

## Key Performance Indicators
The competition has withheld a portion of the data collected to serve as a final test set. After submitting our model(s), they will evaluate the model on this withheld data using an internally defined fit measure. **I don’t have a good understanding of the measure to say much more than this, beyond just copying over the rather long description of this measure.**  

Our models will first predict the PCIAT score (between 0 and 100) for each participant. We will preliminarily use the MSE (mean squared error) to determine the goodness of fit for these models, calculated for the participants that have a PCIAT score (only about 2/3 of the data comes labeled).

The PCIAT score determines the sii score (between 0 and 3) by bins; this sii score is officially what we are trying to predict. In line with the Kaggle competition, we will use the "quadratic weighted kappa" score to determine goodness of fit for an sii-predicting model. This is a score that "measures agreement" between predicted and actual outcomes, with 0 denoting random agreement, 1 total agreement, and negative numbers worse than random.
