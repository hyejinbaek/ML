import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get dataset:adult(uci)
def get_data():
    train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    #test_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    
    df_train_data = pd.read_csv(train_data_url)

    df_data = df_train_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                                    'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country','and-so-on']

    print(df_train_data.info())
    

    



if __name__ == '__main__':
    get_data()

