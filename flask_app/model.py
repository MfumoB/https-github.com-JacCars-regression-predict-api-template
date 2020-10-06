"""

    Helper methods for the pretrained model to be used within our basic API.

    Author: Explore Data Science Academy.
    Note: Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to setup this minimal
    Flask Webserver to serve your developed models within a simple API.

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

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

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define these here.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    2-D numpy array : <class: numpy.narray>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Handle request POST data from multiple sources
    if type(data) == str:
        return json.loads(data)
    # You will need to change this line for your own data preprocessing methods
    return [[np.array(data['exp'])]]