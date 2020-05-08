# Import Python Libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sn

class FeatureExtractor:
    '''
    Feature extractor class for the bank data set.
    
    Calls pre_processing code and pre-processes the data. Then conducts the feature extraction tasks.
    '''
    def __init__(self, change_wd_bool = False, wd = '', filepath = '../data/bank-additional-full.csv'):
        '''
        Initialize the FeatureExtractor class.
            
            change_wd : Default is False, if you will change the working directory set it to True and 
            use the method change_wd.
            
            wd : The default working directory, no need to pass if change_wd_bool is False.
                 It should be set to the location of the src folder. 
                 For example: 'C:/Users/iocak/Desktop/git/ECE229-Project/src'
            
            file_path : Location of the csv data file. Pass the location of 'bank-additional-full.csv' 
            if you would like to change the default data path.
            
        '''
        self.change_wd_bool = change_wd_bool
        self.wd = wd
        self.filepath = filepath
        
    def change_wd(self):
        '''
        Change the working directory.
        This method changes the working directory of the repo. It should always be set to src folder.
        
        '''
        # If the working directory is not 'src' folder in repo, change it
        os.chdir(self.wd)
        
    def load_preprocessed_data(self):
        '''
        Imports pre_processing library from /src folder and loads the preprocessed data.
        
        If the change_wd option is True in the class constructor then imports pre_processing library from the
        new location else uses the default working directory.
        
        Returns: 'df' which is the preprocessed Pandas dataframe.
        
        While reading the data uses the filepath string that was passed in the class constructor.
        '''
        
        if self.change_wd_bool == True:
            self.change_wd()
        
        # import custom pre_processing library
        import pre_processing as pp
            
        # read the data
        bank_data = pp.load_data(self.filepath)
        bank_data.process_all()
        df = bank_data.df
        
        return df
            
    def one_hot_encoder(self, df, encode_time = True):
        '''
        Do one hot encoding on the provided dataframe
        
        Input:
            df: bank data to be one-hot-encoded. (Pandas dataframe)
            
            encode_time: Default is True. If True the columns month and day_of_week are one_hot_encoded too.
                Otherwise they are kept as integers.
            
        Returns:
            one_hot: one hot encoded version of df. (Pandas dataframe)
        '''
        assert isinstance(df, pd.core.frame.DataFrame)
        assert isinstance(encode_time, bool)
        
        if encode_time == True:
            df['month'] = df['month'].astype(str)
            df['day_of_week'] = df['day_of_week'].astype(str)
        
        one_hot = pd.get_dummies(df)
        
        return one_hot
    
    def get_features(self):
        '''
        Get final features.
        
        This method first preprocesses the data and then does the feature extraction.
        Finally it one-hot-encodes the data.
        
        Returns:
            df_features: Final features to be used in machine learning tasks. (Pandas DataFrame)
        '''
        
        # get the preprocessed data
        prep = self.load_preprocessed_data()
        
        
        # call the feature extraction functions
        features = prep.copy()

        # one hot encoding
        features_final = self.one_hot_encoder(features)

        return features_final

def get_feature_extractor(change_wd_bool = False, wd = './src', filepath = '../data/bank-additional-full.csv'):
    '''
    Kickstart the FeatureExtractor class.
                
        change_wd : Default is False, if you will change the working directory set it to True and 
        use the method change_wd.
        
        wd : The default working directory, no need to pass if change_wd_bool is False.
             It should be set to the location of the src folder. 
             For example: 'C:/Users/iocak/Desktop/git/ECE229-Project/src'
        
        file_path : Location of the csv data file. Pass the location of 'bank-additional-full.csv' 
        if you would like to change the default data path.
    
    Returns a FeatureExtractor object.
    '''
    return FeatureExtractor(change_wd_bool, wd, filepath)



