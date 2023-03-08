import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read the dataset
df = pd.read_csv('/content/sample_data/heart_data.csv', sep=',')

# Split the dataset into input and output variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a random forest regression model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Evaluate the model on the test set
score = rf.score(X_test, y_test)
print('R^2 score:', score)
