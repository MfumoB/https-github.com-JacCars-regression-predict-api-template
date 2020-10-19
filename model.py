"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import Pool
import pickle
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """

    train = pd.read_csv('data/Train.csv')
    test = pd.read_csv('data/Test.csv')
    riders_df = pd.read_csv('data/riders.csv')

    ### Train set:

    pops_train = []

    # day of month
    pops_train.append(train[train['Placement_-_Day_of_Month'].eq(train['Confirmation_-_Day_of_Month']) == False].index[0])
    pops_train.append(train[train['Placement_-_Day_of_Month'].eq(train['Confirmation_-_Day_of_Month']) == False].index[1])

    # Both lines below return no outliers
    train[train['Confirmation_-_Day_of_Month'].eq(train['Arrival_at_Pickup_-_Day_of_Month']) == False].index
    train[train['Arrival_at_Pickup_-_Day_of_Month'].eq(train['Pickup_-_Day_of_Month']) == False].index
        
    #day of week
    train[train['Placement_-_Weekday_(Mo_=_1)'].eq(train['Confirmation_-_Weekday_(Mo_=_1)']) == False].index
    # answ: Int64Index([4024, 9804], dtype='int64') - same as for day of month
    # Both lines below return same 2 outliers
    train[train['Confirmation_-_Weekday_(Mo_=_1)'].eq(train['Arrival_at_Pickup_-_Weekday_(Mo_=_1)']) == False].index
    train[train['Arrival_at_Pickup_-_Weekday_(Mo_=_1)'].eq(train['Pickup_-_Weekday_(Mo_=_1)']) == False].index

    train = train.drop(pops_train)
    col_drops = ['Placement_-_Day_of_Month','Placement_-_Weekday_(Mo_=_1)','Confirmation_-_Day_of_Month','Confirmation_-_Weekday_(Mo_=_1)','Arrival_at_Pickup_-_Day_of_Month','Arrival_at_Pickup_-_Weekday_(Mo_=_1)']
    train = train.drop(col_drops, axis=1)

    ### Test set (no outliers):

    # day of month
    #pops.append(test[test['Placement_-_Day_of_Month'].eq(test['Confirmation_-_Day_of_Month']) == False].index[0]


    # Both lines below return no outliers
    #pops.append(test[test['Confirmation_-_Day_of_Month'].eq(test['Arrival_at_Pickup_-_Day_of_Month']) == False].index)
    #pops.append(test[test['Arrival_at_Pickup_-_Day_of_Month'].eq(test['Pickup_-_Day_of_Month']) == False].index)
        
    #day of week
    #test[test['Placement_-_Weekday_(Mo_=_1)'].eq(test['Confirmation_-_Weekday_(Mo_=_1)']) == False].index
    # answ: Int64Index([4024, 9804], dtype='int64') - same as for day of month
    # Both lines below return no outliers
    #test[test['Confirmation_-_Weekday_(Mo_=_1)'].eq(test['Arrival_at_Pickup_-_Weekday_(Mo_=_1)']) == False].index
    #test[test['Arrival_at_Pickup_-_Weekday_(Mo_=_1)'].eq(test['Pickup_-_Weekday_(Mo_=_1)']) == False].index

    #test = test.drop(col_drops, axis=1)

    # Function converts datetime objects (hours and min) to float (e.g. 10:30 --> 10.50)
    # Circularity of time not important since we only work within one day timeframes

    def converter (column):
    '''
    Function converts datetime objects (hours and min) to float (e.g. 10:30 --> 10.50)
    column: datetime column to be converted to float
    returns: pd.Series of same dimensions that can replace datetime column
    '''
    out = []
        for value in column.values:
            try:
                work = pd.to_datetime(value)
                hour = str(int(work.hour))
                minute = str(int(work.minute/60*100))
                out.append(float(hour+'.'+minute))
            except:
                hour, minute = str(int(value.seconds // 3600)), str(int(value.seconds // 60 % 60))
                out.append(float(hour+'.'+minute))
        return pd.Series(out)

    
    # Convert string objects to datetime objects for every time feature
    train_time_conv_lst = ['Arrival_at_Pickup_-_Time','Placement_-_Time','Confirmation_-_Time','Pickup_-_Time']
    train[train_time_conv_lst] = train[train_time_conv_lst].apply(lambda x: pd.to_datetime(x))

    # Calculate new features (time elsapsed between time markers)
    train['Time_from_conf_to_ArrAtPickup'] = train['Arrival_at_Pickup_-_Time'] - train['Confirmation_-_Time']
    train['Time_from_ArrAtPickup_to_Pickup'] = train['Pickup_-_Time'] - train['Arrival_at_Pickup_-_Time']
    #train['Pickup_Hour_of_Day'] = [time.hour for time in train['Pickup_-_Time']]

    # Convert datetime objects to float for interpretablility (e.g. 10:30 --> 10.50)
    train_time_conv_lst2 = ['Arrival_at_Pickup_-_Time','Placement_-_Time','Confirmation_-_Time', 'Pickup_-_Time','Time_from_conf_to_ArrAtPickup','Time_from_ArrAtPickup_to_Pickup']
    train[train_time_conv_lst2] = train[train_time_conv_lst2].apply(converter, axis=1)
    train_time_conv_lst3 = ['Time_from_conf_to_ArrAtPickup','Time_from_ArrAtPickup_to_Pickup']
    train[train_time_conv_lst3] = train[train_time_conv_lst3].apply(lambda x: x*3600)

     ### Test set:


    # Convert string objects to datetime objects for every time feature
    test_time_conv_lst = ['Arrival_at_Pickup_-_Time','Placement_-_Time','Confirmation_-_Time','Pickup_-_Time']
    test[test_time_conv_lst] = test[test_time_conv_lst].apply(lambda x: pd.to_datetime(x))

    # Calculate new features (time elsapsed between time markers)
    test['Time_from_conf_to_ArrAtPickup'] = test['Arrival_at_Pickup_-_Time'] - test['Confirmation_-_Time']
    test['Time_from_ArrAtPickup_to_Pickup'] = test['Pickup_-_Time'] - test['Arrival_at_Pickup_-_Time']
    #test['Pickup_Hour_of_Day'] = [time.hour for time in test['Pickup_-_Time']]

    # Convert datetime objects to float for interpretablility (e.g. 10:30 --> 10.50)
    test_time_conv_lst2 = ['Arrival_at_Pickup_-_Time','Placement_-_Time','Confirmation_-_Time','Pickup_-_Time','Time_from_conf_to_ArrAtPickup','Time_from_ArrAtPickup_to_Pickup']
    test[test_time_conv_lst2] = test[test_time_conv_lst2].apply(converter, axis=1)
    test_time_conv_lst3 = ['Time_from_conf_to_ArrAtPickup','Time_from_ArrAtPickup_to_Pickup']
    test[test_time_conv_lst3] = test[test_time_conv_lst3].apply(lambda x: x*3600)

    ### Train set
    train['Temperature'] = train['Temperature'].fillna(train['Temperature'].mean())

    ### Test set
    test['Temperature'] = test['Temperature'].fillna(test['Temperature'].mean())

    # Train set
    left = train.copy()
    right = riders_df[['Rider_Id', 'Age','No_of_Ratings','Average_Rating']]
    train = left.merge(right, on='Rider_Id', how='left')
    train = train.drop('Rider_Id', axis=1)

    # Test set
    left2 = test.copy()
    right2 = riders_df[['Rider_Id', 'Age','No_of_Ratings', 'Average_Rating']]
    test = left2.merge(right2, on='Rider_Id', how='left')
    test = test.drop('Rider_Id', axis=1)    

    # Additionaly drop Destination times from train (they aren't present in test)
    dest_cols = ['Arrival_at_Destination_-_Day_of_Month','Arrival_at_Destination_-_Weekday_(Mo_=_1)','Arrival_at_Destination_-_Time']
    train = train.drop(dest_cols, axis=1)  

    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = df.fillna(1)
                        
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
