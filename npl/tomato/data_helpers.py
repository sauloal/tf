import numpy as np
import os
import re
import itertools
import random
from collections import Counter
#https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py

from tflearn.data_utils import to_categorical
import tensorflow as tf

import splitter

def read_genome(filename):
  # Dimensions of the images in the GENOME dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  # print "read_genome", filename
  sh  = splitter.sequenceHandler(filename)
  seq = sh.read()
  shd = sh.asDict()
  
  # Convert from a string to a vector of uint8 that is record_bytes long.
  # record = tf.decode_raw( seq, tf.uint8 )
  # record = seq
  record = np.array([ord(s) for s in seq], dtype=np.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  label  = shd['groupId'] #tf.cast( shd['groupId'], tf.int32 )

  return (record, label, shd)


def load_data_and_labels(data_dir, positive_data_file, negative_data_file, test_freq=0.2):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files

    all_examples            = []
    positive_data_file_name = os.path.join(data_dir, positive_data_file)
    print "reading", positive_data_file_name
    with open(positive_data_file_name, "r") as fhd:
        for pfn in fhd:
            pff, pfc = pfn.strip().split(",")
            pff_name = os.path.join(data_dir, pff)
            all_examples.append( read_genome(pff_name) )

    negative_data_file_name = os.path.join(data_dir, negative_data_file)
    print "reading", negative_data_file_name
    with open(negative_data_file_name, "r") as fhd:
        for nfn in fhd:
            nff, pfc           = nfn.strip().split(",")
            nff_name = os.path.join(data_dir, nff)
            all_examples.append( read_genome(nff_name) )

    random.shuffle( all_examples )

    print "total number of sequences {:12,d}".format( len(all_examples) )
    
    train_len    = int(len(all_examples) * (1.0-test_freq))

    print "training sequences        {:12,d}".format( train_len )
    print "test     sequences        {:12,d}".format( len(all_examples) - train_len )

    return (([x[0] for x in all_examples[:train_len]],
             [x[1] for x in all_examples[:train_len]]),
            ([x[0] for x in all_examples[train_len:]],
             [x[1] for x in all_examples[train_len:]]))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data                  = np.array(data)
    data_size             = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data   = data[shuffle_indices]
        else:
            shuffled_data   = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index   = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
