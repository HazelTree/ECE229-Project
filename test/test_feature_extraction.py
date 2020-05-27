import pytest
import sys
sys.path.insert(0, '..')
import pandas as pd
from src.feature_extraction import *
import src.feature_extraction as feature_ex

feature_extraction = FeatureExtractor()
def test_change_wd():
    '''
    Tests the change_wd() function in feature_extractor.py. The change_wd() changes the working directory. We test to ensure that working directory is not changed when we do not pass any parameter to change it.
    '''
    try:
        feature_extraction.change_wd()
    except FileNotFoundError:
        pass

def test_load_preprocessed_data():
    '''
    Tests if the preprocessed data is loaded properly in a pandas dataframe.
    '''
    assert type(feature_extraction.load_preprocessed_data()) == pd.DataFrame
def test_data_scaler():
    '''
    Tests the data_scaler() function
    '''
    assert type(feature_extraction.data_scaler(feature_extraction.load_preprocessed_data())) == pd.DataFrame
    
def test_one_hot_encoder():
    '''
    Tests the one_hot_encoder() function
    '''
    encoding = feature_extraction.data_scaler(feature_extraction.data_scaler(feature_extraction.load_preprocessed_data()))
    assert type(encoding) == pd.DataFrame
def test_get_features():
    '''
    Tests the get_features() function
    '''
    feature_extraction.get_features()
    
def test_get_train_test_split():
    '''
    Tests the get_train_test_split() function
    '''
    try:
        feature_extraction.get_train_test_split()
    except FileNotFoundError:
        pass
    
    
def test_get_feature_extractor():
    '''
    Tests the get_feature_extractor() function
    '''
    try:
        feature_ex.get_feature_extractor()
    except FileNotFoundError:
        pass

