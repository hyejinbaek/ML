# naive bayes

def label_single():

    #Assigning features and label variables
    weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']
    temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
    play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


    # Import LabelEncoder
    from sklearn import preprocessing

    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers.
    weather_encoded=le.fit_transform(weather)
    #print(weather_encoded)

    temp_encoded = le.fit_transform(temp)

    label = le.fit_transform(play)

    features = zip(weather_encoded, temp_encoded)
    features = list(features)
    #print(features)

    ### create model ###
    # import Gaussian Naive Bayes model
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()

    model.fit(features, label)

    predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild
    print(predicted)



def label_multi():
    from sklearn import datasets

    wine = datasets.load_wine()
    #print(wine.feature_names, wine.target_names)
    
    # import train_test_split function
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

    #Import Gaussian Naive Bayes model
    from sklearn.naive_bayes import GaussianNB
    #Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    label_multi()
