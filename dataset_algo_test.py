import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get dataset:liver-disorders(uci)
def non_missing_value():
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df_data = pd.read_csv(data_url)
    col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                                  'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    #print(df_data)
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0)
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].astype(int)
    #print(df_data.info())
   
    
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    
    train_col = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    X_train, X_test, y_train, y_test = train_test_split(df_data[train_col], df_data['Class'], test_size = 0.25, random_state = 32)
    #print(X_train)

    model = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4, random_state = 32)
    print("model === ", model)
    model.fit(X_train, y_train)
    print("model fit 완료")
    
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    print("y_pred ====", y_pred)
    print(accuracy_score(y_pred, y_test))
    
def missing_value10():
    # create missing value
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df_data = pd.read_csv(data_url)
    col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                                  'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0)
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].astype(int)
    
    print(df_data[['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']].describe())
    #print(df_data.describe())



if __name__ == '__main__':
    #non_missing_value()
    missing_value10()

