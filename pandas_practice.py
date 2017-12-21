import pandas as pd
import numpy as np

# pd?
# pd.__version__
# pd.show_versions()

oo = pd.read_csv('data/olympics.csv', skiprows=4)

# Series and Dataframes 

# print(oo.head())
# print(oo.City)
# print(oo.Athlete)
# print(oo[['City','Edition', 'Athlete']])
# print(oo)
# print(oo['NOC'])
# print(oo.NOC)
# print(type(oo.NOC))
# print(type(oo[['Edition','City','Athlete', 'Medal']]))


# Data Input and Validation


# Head and Tail


# print(oo.shape)
# print(oo.shape[0])

# head and tail give first 5 rows is no "n" is specified
# print(oo.head())
# print(oo.tail())

# print(oo.head(3))
# print(oo.tail())

# Info
# print(oo.info())

# VALUE COUNTS


 # By default, you will not get a count of the na values. 
 # Remember those are the missing data values. 
 # If your data set has a significant number of na values, 
 # this can be misleading. 
 # So we will not see any difference in our data set as we don't have any missing data. 
 # So if I hit shift and tab, dropna, I need to just change dropna to False, 
 # and there will be no difference in what we see.


# print(oo.Edition.value_counts())
# print(oo.Gender.value_counts(ascending=True, dropna=False))




# SORT VALUES

# sorts values in a Series
ath = oo.Athlete.sort_values()
# print(ath)

print(oo.sort_values(by=['Edition', 'Athlete']))


























