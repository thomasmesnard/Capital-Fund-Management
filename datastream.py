import logging
import random
import numpy

import cPickle

from picklable_itertools import iter_

from fuel.datasets import Dataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding, Transformer

import sys
import os

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# -------------- DATASTREAM SETUP --------------------

def setup_datastream(file_name,target_file, batchsize,proportion_train):
    data = numpy.genfromtxt(file_name,skip_header=1,delimiter=',',filling_values=0)[:,:,None]
    data_pros = data[:,3:,:].astype(numpy.float32)
    data_mean = numpy.mean(numpy.mean(data_pros,1))
    data_std = numpy.std(numpy.std(data_pros,1))
    data_mean_reshape = data_mean*numpy.ones(data_pros.shape[1])
    data_mean_reshape = data_mean_reshape.astype(numpy.float32)
    data_std_reshape = data_std*numpy.ones(data_pros.shape[1])
    data_std_reshape = data_std_reshape.astype(numpy.float32)

    product = data[:,2,0]
    ref = {int(k):i for i, k in enumerate(open('listid.txt'))}
    product = numpy.array([ref[product[i]] for i in range(len(product))])
    
    data = (data_pros - data_mean_reshape[None,:,None])/data_std_reshape[None,:,None]

    target = numpy.genfromtxt(target_file,skip_header=1,delimiter=';')[:,:,None]
    target = target[:,1,:].astype(numpy.float32)

    print "Shuffling train set..."
    order = list(range(data.shape[0]))
    SEED = 448
    random.seed(SEED)
    random.shuffle(order)
    data = numpy.concatenate([data[i:i+1] for i in order], axis=0)
    product = numpy.concatenate([product[i:i+1] for i in order], axis=0) 
    target = numpy.concatenate([target[i:i+1] for i in order], axis=0) 

    n_train  = int(proportion_train*data.shape[0])
    
    ds = IndexableDataset({'input':data[:n_train],'target':target[:n_train],
                        'product':product[:n_train]})
    scheme = SequentialScheme(batch_size=batchsize,examples= ds.num_examples)
    stream = DataStream(ds,iteration_scheme = scheme)

    ds_valid = IndexableDataset({'input':data[n_train:],'target':target[n_train:],
                        'product':product[n_train:]})
    scheme_valid = SequentialScheme(batch_size=batchsize,examples= ds_valid.num_examples)
    stream_valid = DataStream(ds_valid,iteration_scheme = scheme_valid)

    return ds, stream, stream_valid

if __name__ == "__main__":
    # Test
    ds, stream,_ = setup_datastream('training_input.csv',
        'challenge_output_data_training_file_prediction_of_transaction_volumes_in_financial_markets.csv',
        2,0.9)
    it = stream.get_epoch_iterator()

    for i, d in enumerate(stream.get_epoch_iterator()):
        print '--'
        print d
        if i >= 1: break

# vim: set sts=4 ts=4 sw=4 tw=0 et :
