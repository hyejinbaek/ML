'''
tabnet_breast caner dataset
'''

import os
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

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

df_data[train_col].loc[:139] = np.nan

#print("3333333333",df_data.info())

if "Set" not in df_data.columns:
    df_data["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(df_data.shape[0],))

train_indices = df_data[df_data.Set=="train"].index
valid_indices = df_data[df_data.Set=="valid"].index
test_indices = df_data[df_data.Set=="test"].index

# simple preprocessing
nunique = df_data.nunique()
types = df_data.dtypes

categorical_columns = []
categorical_dims =  {}

for col in df_data.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, df_data[col].nunique())
        l_enc = LabelEncoder()
        df_data[col] = df_data[col].fillna("VV_likely")
        df_data[col] = l_enc.fit_transform(df_data[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        df_data.fillna(df_data.loc[train_indices, col].mean(), inplace=True)

target = 'Class'
# Define categorical feature for categorical embeddings
unused_feat = ['Set']

features = [ col for col in df_data.columns if col not in unused_feat+[target]]
cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
# define your embedding sizes : here just a random choice
cat_emb_dim = [5, 4, 3, 6, 2, 2, 1, 10]

# check that pipeline accepts strings
#df_data.loc[df_data[target]==0, target] = "wealthy"
#df_data.loc[df_data[target]==1, target] = "not_wealthy"



# network parameters
tabnet_params = {"cat_idxs":cat_idxs,
                 "cat_dims":cat_dims,
                 "cat_emb_dim":1,
                 "optimizer_fn":torch.optim.Adam,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax' # "sparsemax"
                 #"gamma" : 1.3 # coefficient for feature reusage in the masks
                }

clf = TabNetClassifier(**tabnet_params)

# training
X_train = df_data[features].values[train_indices]
#print("====X_train====", X_train)
y_train = df_data[target].values[train_indices]
#print("====y_train====", y_train)

X_valid = df_data[features].values[valid_indices]
y_valid = df_data[target].values[valid_indices]

X_test = df_data[features].values[test_indices]
y_test = df_data[target].values[test_indices]

max_epochs = 100 if not os.getenv("CI", False) else 2


# This illustrates the warm_start=False behaviour
save_history = []

for _ in range(2):
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs , patience=20,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    save_history.append(clf.history["valid_auc"])


# plot losses
plt.plot(clf.history['loss'])

# plot auc
plt.plot(clf.history['train_auc'])
plt.plot(clf.history['valid_auc'])

# plot learning rates
plt.plot(clf.history['lr'])

# prediction
preds = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)
test_score_std = np.std(y_score, y_true)


preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

dataset_name = 'breast-cancer'
print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc} ± {test_score_std: .3f}")


# global
feat_importances = pd.Series(clf.feature_importances_, index=features)
feat_importances.plot(kind='barh')

#local
explain_matrix, masks = clf.explain(X_test)
fig, axs = plt.subplots(1, 3, figsize=(20,20))

for i in range(3):
    axs[i].imshow(masks[i][30:50])
    axs[i].set_title(f"mask {i}")

