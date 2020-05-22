"""

"""
import pytest
import pandas as pd
import sys
sys.path.insert(0, '..')
import visualization.analysis as analysis
from visualization.analysis import *

csv_path = '../data/bank-additional-full.csv'
my_analysis = Analysis(csv_path)

def test_get_column():
    marital_status = ['married', 'single', 'divorced', 'unknown']
    assert all(i in marital_status for i in my_analysis.get_column(column = 'marital').unique())  == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'age')) == True
    month = [ 5,  6,  7,  8, 10, 11, 12,  3,  4,  9]
    assert all(i in month for i in my_analysis.get_column(column ='month').unique())
#    education_lvl = ['Dropout', 'professional.course', 'unknown', 'university.degree',
#    'illiterate']
    #print(all(i in education_lvl for i in my_analysis.get_column(column ='education').unique))
#    assert all(i in my_analysis.get_column(column ='education').unique() for i in education_lvl) == True
    job_status = ['housemaid', 'services', 'admin', 'blue-collar', 'technician',
    'retired', 'management', 'unemployed', 'self-employed', 'unknown',
    'entrepreneur', 'student']
    assert all(i in job_status for i in my_analysis.get_column('job').unique()) == True
#    loan_housing_status = ['no', 'yes', 'Info Not Available']
#    assert all(i in loan_housing_status for i in my_analysis.get_column(column ='loan').unique()) == True
    contact_type = ['telephone', 'cellular']
    assert all(i in contact_type for i in my_analysis.get_column(column ='contact').unique()) == True
    credit_default = ['no', 'unknown', 'yes']
    assert all(i in credit_default for i in my_analysis.get_column(column ='default').unique()) == True
    day_of_week = [0, 1, 2, 3, 4]
    assert all(i in day_of_week for i in my_analysis.get_column(column ='day_of_week').unique())== True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'duration')) == True
    assert all(isinstance(i, (int)) for i in my_analysis.get_column(column = 'campaign')) == True
    assert all(isinstance(i, (int)) for i in my_analysis.get_column(column = 'pdays')) == True
    assert all(isinstance(i, (int)) for i in my_analysis.get_column(column = 'previous')) == True
    poutcome = ['nonexistent', 'failure', 'success']
    assert all(i  for i in my_analysis.get_column(column = 'poutcome')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'emp.var.rate')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'cons.price.idx')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'cons.conf.idx')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'euribor3m')) == True
    assert all(isinstance(i, (int, float)) for i in my_analysis.get_column(column = 'nr.employed')) == True
     
     
def test_get_probabilities():
    p = my_analysis.get_probabilities('marital')
    assert all(1>=i>=0 for i in p['y']) == True
    assert all(isinstance(i,(int,float)) for i in p['y']) == True

def test_get_success_count():
    p = my_analysis.get_success_count('marital')
    assert type(p) == pd.DataFrame
    assert all(i>=0 for i in p['y']) == True
    assert all(isinstance(i,int) for i in p['y']) == True
    
def test_get_count():
    p = my_analysis.get_count('marital')
    assert type(p) == pd.DataFrame
    assert all(i>=0 for i in p['y']) == True
    assert all(isinstance(i,int) for i in p['y']) == True


#def test_get_yes_no_count():
#    p = my_analysis.get_yes_no_count('marital')
#    assert type(p) == pd.DataFrame()
#    assert all(i>=0 for i in p['y']) == True
#    assert all(isinstance(i,int) for i in p['y'])


def test_percentage_of_population():
    p = my_analysis.percentage_of_population('marital')
    assert all(100>=i>=0 for i in p) == True
    assert all(isinstance(i,(int, float)) for i in p) == True
    
def test_map_age():
    myList, labels = my_analysis.map_age()
    assert isinstance(myList,list) and isinstance(labels, list)
    
def test_get_age_prob_success():
    myList, labels = my_analysis.map_age()
    p = my_analysis.get_age_prob_success(myList)
    assert isinstance(p, list)
    assert all(0<=i<=100 for i in p)
    assert all(isinstance(i, (int, float)) for i in p)

def test_filter_unknown_marital():
    marital_analysis = MaritalAnalysis(csv_path)
    k = marital_analysis.get_count('marital')['marital']
    status = ['married', 'divorced', 'single']
    assert all(i in status for i in k)==True

def test_feature_importance():
    feature = FeatureAnalysis(csv_path)
    importance = feature.get_feature_importance()
    assert type(importance) == pd.DataFrame
    
def test_number_to_day_of_week():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_of_week_probabilities = my_analysis.get_probabilities('day_of_week')
    num_to_day = analysis.number_to_day_of_week(day_of_week_probabilities['day_of_week'])
    assert all(i in days for i in num_to_day)==True

def test_number_to_month():
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_probabilities = my_analysis.get_probabilities('month')
    num_to_month = analysis.number_to_month(month_probabilities['month'])
    assert all(i in months for i in num_to_month)==True


