'''
adaboost regressor knn imputation
'''

from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.impute import KNNImputer

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                              'Viscera weight', 'Shell weight', 'Rings']

df_data['Sex'] = df_data['Sex'].replace({'M':0, 'F':1, 'I':2})
df_data.info()



# define train code
def cross_valid(
    X : np.array, y : np.array,
    model = AdaBoostRegressor(n_estimators = 50, learning_rate=1),
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42),
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error'],
    **kwargs,
):
    # model.fit(X_features, y_label) # !! cross_validate 에서 train 동작이 있으므로 지워야 하는 코드 !!
    cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, **kwargs)
    for score_name in cv_result:
        if 'test' in score_name:
            test_score_mean, test_score_std = np.mean(cv_result[score_name]), np.std(cv_result[score_name])
            print(f'{score_name}: {test_score_mean:.4f} ± {test_score_std:.4f}')

# define missing value setting function
from typing import Tuple


train_col = ['Sex','Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']

def set_missing_value(df: pd.DataFrame, ratio: float) -> Tuple[np.array, np.array]:

    missing_length = int(len(df) * ratio)

    print('{missing_length=}', missing_length)

    df = df.copy()
    df.loc[:missing_length-1, train_col] = np.nan

    imputer = KNNImputer(n_neighbors=2)
    df_im = imputer.fit_transform(df)
    df_im = pd.DataFrame(df_im, columns = df_data.columns)

    # df_data[train_col]을 array 형태로 변경
    X = df_im[train_col].to_numpy()
    # df_data['Class'] : 판다스 series type을 array형태로 변경
    y = df_im['Rings'].to_numpy()

    # X: features matrix, y: label vector
    return X, y

X, y = set_missing_value(df_data, 0.0)
print("missing data 0% ===", cross_valid(X,y))


# xgboost classifier 을 이용할 때 결측치 20% 성능 예측
X, y = set_missing_value(df_data, 0.2)
#cross_valid(X, y)
print("missing data 20% === ", cross_valid(X, y))

# xgb classifier을 이용하여 결측치 40%
X, y = set_missing_value(df_data, 0.4)
print("missing data 40% === ", cross_valid(X, y))


# xgb classifier을 이용하여 결측치 60%
X, y = set_missing_value(df_data, 0.6)
print("missing data 60% === ", cross_valid(X, y))

# xgb classifier을 이용하여 결측치 80%
X, y = set_missing_value(df_data, 0.8)
print("missing data 80% === ", cross_valid(X, y))

