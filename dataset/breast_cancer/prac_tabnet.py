import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                              'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
df_data['Class'] = df_data['Class'].replace({2:0, 4:1})

train = pd.DataFrame({'Clump Thickness' : df_data['Clump Thickness'],
                      'Uniformity of Cell Size' : df_data['Uniformity of Cell Size'],
                      'Uniformity of Cell Shape' : df_data['Uniformity of Cell Shape'],
                      'Marginal Adhesion' : df_data['Marginal Adhesion'],
                      'Single Epithelial Cell Size' : df_data['Single Epithelial Cell Size'],
                      'Bare Nuclei' : df_data['Bare Nuclei'],
                      'Bland Chromatin' : df_data['Bland Chromatin'],
                      'Normal Nucleoli' : df_data['Normal Nucleoli'],
                      'Mitoses' : df_data['Mitoses'],
                     'target' : df_data['Class']
                     })
#print(train.info())

l_enc = LabelEncoder()

if "Set" not in train.columns:
    train["Set"] = np.random.choice(["train", "valid"], p =[.8, .2,], size=(train.shape[0],))

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index

nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object':
        print(col, train[col].nunique())
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음.
unused_feat = ['Set']
features = [ col for col in train.columns if col not in unused_feat+['target']]  # 7 target
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]



X_train = train[features].values[train_indices]
y_train = train['target'].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train['target'].values[valid_indices]


# make y data
y = train["target"]
print(y)

