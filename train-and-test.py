x_train, x_test, y_train, y_test  =   train_test_split(predictors, targets, test_size=.3)
print( "Predictor - Training : ", x_train.shape, "Predictor - Testing : ", x_test.shape )

# Use Naive Bayes Classifier in our model

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier=classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)

# Analyze accuracy of predictions
sklearn.metrics.confusion_matrix(y_test,predictions)

# Apply an accuracy score to our model
sklearn.metrics.accuracy_score(y_test, predictions)
