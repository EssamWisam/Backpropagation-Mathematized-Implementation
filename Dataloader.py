import random
import pickle               #from python objects ro files and vice versa.
import gzip                 #compression and extraction of files.
import numpy as np
from numpy import savetxt


def load_data(mini_batch_size):
    with gzip.open('datasets/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, testing_data = pickle.load(f, encoding="latin1")
            #for any of the data groups, [0][j] has the jth image and [1][j] has its label.
            #latin1 is there due to incompabillity issues between Python 2 (dump) and Python 3 (load).
        re = lambda x: columnize(x)
        return (make_batches(re(training_data), mini_batch_size),  re(validation_data), re(testing_data))
            #we'll use column vectors in order to do math (that's why we columnize the dataset)
    

def columnize(data_group):                                            
    images, labels= data_group[0], data_group[1]
    data_x = [np.reshape(x, (784, 1 )) for x in images]         #x should be a column vector (flattened image).
    data_y = [np.identity(10)[:,[y]] for y in labels]           #y is a one-hot column vector.
    return list(zip(data_x, data_y))                            #returned in a tuple for neatness.


def make_batches(training_data, mini_batch_size):
    n = len(training_data)
    random.shuffle(training_data)
    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
    return mini_batches



