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

l_enc = LabelEncoder()


def cross_valid(
    X: np.array, y: np.array,
    
    model=TabNetClassifier(cat_idxs=[],  # cat_idxs: List[int] = field(default_factory=list)
                       cat_dims=[],  # cat_dims: List[int] = field(default_factory=list)
                       cat_emb_dim=10,     # int = 1
                       optimizer_fn=torch.optim.Adam,
                       optimizer_params=dict(lr=1e-3),
                       scheduler_params={"step_size":50,
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='sparsemax' # "sparsemax", entmax
                      ),
    cv=StratifiedKFold(n_splits=5, shuffle = True, random_state=42),
    scoring=['accuracy', 'f1', 'recall', 'precision'],
    **kwargs,
):
    
    # model.fit(X_features, y_label) # !! cross_validate 에서 train 동작이 있으므로 지워야 하는 코드 !!
    cv_result = cross_validate(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle = True, random_state=42), scoring=scoring, **kwargs)
    
    for score_name in cv_result:
        if 'test' in score_name:
            test_score_mean, test_score_std = np.mean(cv_result[score_name]), np.std(cv_result[score_name])
            print(f'{score_name}: {test_score_mean:.4f} ± {test_score_std:.4f}') # 유효숫자 소수점 아래 4 자리까지 표시
            
from typing import Tuple

train = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                              'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
                     
def set_missing_value(df: pd.DataFrame, ratio: float) -> Tuple[np.array, np.array]:

    missing_length = int(len(df) * ratio)

    print(missing_length)
    
    df = df.copy()
    df.loc[:missing_length-1, train] = np.nan
    print("=========df=========", df)
    #df = df.fillna(0)
    
    # df_data[train_col]을 array 형태로 변경
    X = df[train].to_numpy()
    # df_data['Class'] : 판다스 series type을 array형태로 변경
    y = df['Class'].to_numpy()
    
    # X: features matrix, y: label vector
    return X, y

# xgboost classifier 을 이용할 때 결측치가 없을 때 성능 예측
X, y = set_missing_value(df_data, 0.2)
print(cross_valid(X, y))