"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')
riders = pd.read_csv('data/riders.csv')
train = train.merge(riders, how='left', on='Rider Id')

X_train_drop_lst = ['Order_No','Vehicle_Type','Precipitation_in_millimeters','Time_from_Pickup_to_Arrival']
X_train = train.drop(X_train_drop_lst, axis=1)

# Take log of target feature to cicumvent outlier issue (see output graph for distribution before(left) and after(right))
y_train = train['Time_from_Pickup_to_Arrival']

#Assign test variables
X_test_drop_lst = ['Order_No','Vehicle_Type','Vehicle_Type','Precipitation_in_millimeters']
X_test = test.drop(X_test_drop_lst, axis=1)

# Verify that train and test DFs underwent same transformations (by column name)
# Will print column names that don't match
for pair in [x for x in zip(X_test.columns, X_train.columns)]:
    if pair[0]==pair[1]:
        pass
    else:
        print(pair[0], pair[1])

encoder = preprocessing.LabelEncoder()
X_train['User_Id'] = encoder.fit_transform(X_train['User_Id'])
#X_train['Rider_Id'] = encoder.fit_transform(X_train['Rider_Id'])
X_train['Personal_or_Business'] = encoder.fit_transform(X_train['Personal_or_Business'])

X_test['User_Id'] = encoder.fit_transform(X_test['User_Id'])
#X_test['Rider_Id'] = encoder.fit_transform(X_test['Rider_Id'])
X_test['Personal_or_Business'] = encoder.fit_transform(X_test['Personal_or_Business'])

drops = ['Platform_Type','Personal_or_Business']
X_train = X_train.drop(drops, axis=1)
X_test = X_test.drop(drops, axis=1)

drops = ['Platform_Type','Personal_or_Business']
X_train = X_train.drop(drops, axis=1)
X_test = X_test.drop(drops, axis=1)

#y_train = train[['Time from Pickup to Arrival']]
#X_train = train[['Pickup Lat','Pickup Long',
                 #'Destination Lat','Destination Long']]

# Fit model
# CatBoost can process categorical variables 
# indicate categorical variables with parameter cat_features and list categ_vars
categ_vars = ['User_Id'] 
CatRegressor = CatBoostRegressor()
print ("Training Model...")
CatRegressor.fit(X_train, y_train, cat_features=categ_vars)

y_pred = CatRegressor.predict(X_test)
y_pred

# Pickle model for use within our API
save_path = '../trained-models/CatRegression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(CatRegression, open(save_path,'wb'))

