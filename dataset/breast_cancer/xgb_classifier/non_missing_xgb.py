import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                              'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0)
df_data['Bare Nuclei'] = df_data['Bare Nuclei'].astype(int)
df_data['Class'] = df_data['Class'].replace(2,0)
df_data['Class'] = df_data['Class'].replace(4,1)

train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
             'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

# df_data[train_col]을 array 형태로 변경
X_features = np.array(df_data[train_col])
# df_data['Class'] : 판다스 series type을 array형태로 변경
y_label = np.array(df_data['Class'])

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=156)

model = XGBClassifier()
model.fit(X_features, y_label)
X = cross_validate(model, X_features, y_label, cv = StratifiedKFold(n_splits=3, shuffle = True))
Y = np.mean(X['test_score']), np.std(X['test_score'])
print(Y)

