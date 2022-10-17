'''
전체 데이터셋에서 몇 % 추출할 건지에 대한 코드
'''

import numpy as np
import pandas as pd

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
                              'Viscera weight', 'Shell weight', 'age']
df_data['Sex'] = df_data['Sex'].replace('M',0)
df_data['Sex'] = df_data['Sex'].replace('F',1)
df_data['Sex'] = df_data['Sex'].replace('I',2)

random_df_data = df_data.sample(frac = 0.8, random_state=200)
random_df_data
