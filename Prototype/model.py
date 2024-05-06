import pandas as pd
import pickle

df = pd.read_csv('Crop_recommendation.csv')

X = df.drop('label', axis=1)
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.32, shuffle = True, random_state = 0)


import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# save model
path = 'model.pkl'
pickle.dump(model, open(path, 'wb'))