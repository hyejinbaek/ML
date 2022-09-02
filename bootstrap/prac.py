import pandas as pd
import numpy as np
np.random.seed(104)


df = pd.read_csv('coffee_dataset.csv')

df_sample = df.sample(200)
print(df_sample.head())
