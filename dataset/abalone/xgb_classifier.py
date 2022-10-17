import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                              'Viscera weight', 'Shell weight', 'age']
df_data['Sex'] = df_data['Sex'].replace('M',0)
df_data['Sex'] = df_data['Sex'].replace('F',1)
df_data['Sex'] = df_data['Sex'].replace('I',2)

train_col = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
X_features = np.array(df_data[train_col])
y_label = np.array(df_data['age'])

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=156)

model = XGBClassifier()
model.fit(X_features, y_label)

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
X = cross_validate(model, X_features, y_label, cv = StratifiedKFold(n_splits=3, shuffle = True))
Y = np.mean(X['test_score']), np.std(X['test_score'])
print(Y)

