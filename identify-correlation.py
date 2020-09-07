import seaborn as sns
corr = train.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr, vmax=0.5, center=0, square=True, linewidths=2, cmap='Blues')
plt.savefig("heatmap.png")
plt.show()

# Identify correlation between dependent variable (i.e. ordered) and independent variables (i.e. website interactions)
train.corr()['ordered']

# Remove unecessary variables that don't have any impact on purchase or have low correlations
# Drop columns with low correlation
predictors = train.drop(['ordered','UserID','device_mobile'], axis=1)
targets = train.ordered

# Verify variables
print(predictors.columns)

