import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

def bootstrap():
    np.random.seed(104)

    df = pd.read_csv('coffee_dataset.csv')

    df_sample = df.sample(200)
    #print(df_sample.head())

    iterationNum = 1000
    diffHeightList = []
    for _ in range(iterationNum):
        bootSample = df_sample.sample(200, replace = True)
        nonCoffeeHeightMean = bootSample[bootSample['drinks_coffee'] == False].height.mean()
        #print("noncoffee =====", nonCoffeeHeightMean)
        coffeeHeightMean = bootSample[bootSample['drinks_coffee'] == True].height.mean()
        #print("coffee =====", coffeeHeightMean)
        diff = nonCoffeeHeightMean - coffeeHeightMean
        #print("diff ====", diff)
        diffHeightList.append(diff)
    #print(diffHeightList)
    a = np.percentile(diffHeightList, 0.5), np.percentile(diffHeightList, 99.5)
    #print(a)


    diffHeightListByAge = []
    for _ in range(iterationNum):
        bootSample = df_sample.sample(200, replace = True)
        over21HeightMean = bootSample[bootSample['age'] == '>=21'].height.mean()
        under21HeightMean = bootSample[bootSample['age'] == '<21'].height.mean()
        diff = over21HeightMean - under21HeightMean
        diffHeightListByAge.append(diff)

    b = np.percentile(diffHeightListByAge, 0.5), np.percentile(diffHeightListByAge, 99.5)
    #print(b)

    diffHeightListUnder21 = []
    for _ in range(iterationNum):
        bootSample = df_sample.sample(200, replace=True)
        nonCoffeeHeightMeanUnder21 = bootSample.query("age == '<21' and drinks_coffee == False").height.mean()
        coffeeHeightMeanUnder21 = bootSample.query("age == '<21' and drinks_coffee == True").height.mean()
        diff = nonCoffeeHeightMeanUnder21 - coffeeHeightMeanUnder21
        diffHeightListUnder21.append(diff)

    c = np.percentile(diffHeightListUnder21, 0.5), np.percentile(diffHeightListUnder21, 99.5)
    #print(c)

    diffHeightListOver21 = []
    for _ in range(iterationNum):
        bootSample = df_sample.sample(200, replace=True)
        nonCoffeeHeightMeanOver21 = bootSample.query("age != '<21' and drinks_coffee == False").height.mean()
        coffeeHeightMeanOver21 = bootSample.query("age != '<21' and drinks_coffee == True").height.mean()

        diff = nonCoffeeHeightMeanOver21 - coffeeHeightMeanOver21
        diffHeightListOver21.append(diff)

    d = np.percentile(diffHeightListOver21, 0.5), np.percentile(diffHeightListOver21, 99.5)
    #print(d)

def sample():
    np.random.seed(42)

    full_data = pd.read_csv('coffee_dataset.csv')

    sample1 = full_data.sample(5)
    #print(sample1)




if __name__ == '__main__':
    sample()
