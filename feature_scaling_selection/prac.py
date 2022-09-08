# feature scaling & selection

################################################
###########Feature Scaling######################
################################################
import pandas as pd
from sklearn import preprocessing

# 1. min-max normalization
height_weight_people = {'height' : [6.1, 5.9, 5.2], 'weight' : [140, 175, 115]}
df = pd.DataFrame(height_weight_people, index = ['A', 'B', 'C'])
#print(df)

minmax_scaler = preprocessing.MinMaxScaler()
minmax_scaler.fit(df)
#print(minmax_scaler.transform(df))

# 2. Standardization
standard_scaler = preprocessing.StandardScaler()

standard_scaler.fit(df)
#print(standard_scaler.transform(df))

################################################
###########Feature Selection####################
################################################
from sklearn.linear_model import Lasso

regression = Lasso()
print(regression)
regression.fit(features, labels)

regression.coef_
