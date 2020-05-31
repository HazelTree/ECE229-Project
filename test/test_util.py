import pytest
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/..')
os.chdir('..')
import util

#print(os.getcwd())


def test_util():
    ''' Tests util.py'''
    
    p = util.dynamic_predict()
    assert type(p) == np.ndarray
    assert isinstance(p[0], (int,float))
