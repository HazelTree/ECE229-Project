import pytest
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/..')


#print(os.getcwd())
@pytest.fixture(autouse=True)
def teardown():
    d = os.path.dirname(os.path.abspath('test_util.py'))
    d = d.split('/')
    if d[-1]!='test':
       os.chdir('test')

def test_util():
    ''' Tests util.py'''
    os.chdir('..')
    import util
    p = util.dynamic_predict()
    assert type(p) == np.ndarray
    assert isinstance(p[0], (int,float))
