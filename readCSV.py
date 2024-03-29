import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

#Read data from CSV file
train = pd.read_csv(r"C:\Users\jaykhan\Desktop\Data Science\customer_propensity_score.csv")

#Understand data types in train dataframe
train.dtypes
print(train.describe())
print(train.info())
train.head()


import seaborn as sns
corr = train.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr, vmax=0.5, center=0, square=True, linewidths=2, cmap='Blues')
plt.savefig("heatmap.png")
plt.show()

train.corr()['ordered']

# Drop columns with low correlation
predictors = train.drop(['ordered','UserID','device_mobile'], axis=1)

targets = train.ordered

x_train, x_test, y_train, y_test  =   train_test_split(predictors, targets, test_size=.3)
print( "Predictor - Training : ", x_train.shape, "Predictor - Testing : ", x_test.shape )

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier=classifier.fit(x_train,y_train)

predictions=classifier.predict(x_test)

#Analyze accuracy of predictions
sklearn.metrics.confusion_matrix(y_test,predictions)

sklearn.metrics.accuracy_score(y_test, predictions)

yesterday_prospects = pd.read_csv(r"C:\Users\jaykhan\Desktop\Data Science\customer_propensity_score.csv")
print(yesterday_prospects.info())

userids = yesterday_prospects.UserID
yesterday_prospects = yesterday_prospects.drop(['ordered','UserID','device_mobile'], axis=1)
print(yesterday_prospects.head(10))

yesterday_prospects.shape

yesterday_prospects['propensity'] = classifier.predict_proba(yesterday_prospects)[:,1]
print(yesterday_prospects.head())

pd.DataFrame(userids)
results = pd.concat([userids, yesterday_prospects], axis=1)
print(results.head(30))


results.to_csv(r"C:\Users\jaykhan\Desktop\Data Science\results.csv")
