import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    with open('adult.data') as f:
        df = pd.DataFrame(f)
        print(df)



if __name__ == '__main__':
    get_data()

