# Laod data from yesterday's visitors who haven't placed order
yesterday_prospects = pd.read_csv(r"C:\Users\jaykhan\Desktop\Data Science\customer_propensity_score.csv")
print(yesterday_prospects.info())

# Match load data with training set data
userids = yesterday_prospects.UserID
yesterday_prospects = yesterday_prospects.drop(['ordered','UserID','device_mobile'], axis=1)
print(yesterday_prospects.head(10))

# Ensure shape of load data matches with training set data
yesterday_prospects.shape

# Run predictions and insert it into a new field - propensity
yesterday_prospects['propensity'] = classifier.predict_proba(yesterday_prospects)[:,1]
print(yesterday_prospects.head())

# Append userIds back to dataset
pd.DataFrame(userids)
results = pd.concat([userids, yesterday_prospects], axis=1)
print(results.head(30))

results.to_csv(r"C:\Users\jaykhan\Desktop\Data Science\results.csv")






