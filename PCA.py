import scipy.io
import scipy as sp
import numpy as np
import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mat = scipy.io.loadmat('./joke_data/joke_train.mat')
    Nan_train = mat['train']
    train = np.nan_to_num(Nan_train)
    u,s,v = np.linalg.svd(train)
    print u.shape
    print s.shape
    print v.shape
    print s
