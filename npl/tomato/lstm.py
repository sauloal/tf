#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""
# from __future__ import division, print_function, absolute_import

import sys

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb

import data_helpers

INPUT_DATA_DIM    = 50000
MAX_SEQ_LEN       = 10000
RATE              =     0.001
INPUT_DIM         = 10000
OUTPUT_DIM        =   128
BATCH_SIZE        =    32
DROPOUT           =     0.8
VERBOSE           =     0
OPTIMIZER         = 'adam'
ACTIVATION        = 'softmax'
LOSS              = 'categorical_crossentropy'

# IMDB Dataset loading
# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                 valid_portion=0.1)

def run(data_dir, positive_data_file, negative_data_file, test_freq=0.2):
    print "data {} positive {} negative {}".format( data_dir, positive_data_file, negative_data_file )

    cfg, (trainX, trainY), (testX , testY) = data_helpers.load_data_and_labels(data_dir, positive_data_file, negative_data_file, test_freq=0.2)

    #{'test_len': 3204, 'seg_end': 5125550, 'seqName': u'SL2.50ch01', 'end': 50001, 'ctime': u'Wed Feb  8 22:27:55 2017', 'seg_start': 1, 'seg_serial': 0, 'blockSize': 50000, 'start': 1, 'host': u'assembly', 'version': u'1.0', 'group': u'Euchromatin', 'serial': 1, 'train_len': 12816, 'groupId': 0, 'inputFile': u'/home/aflit001/dev/tf/npl/tomato/S_lycopersicum_chromosomes.2.50.fa'}
    #print cfg
    #quit()

    # Converting labels to binary vectors
    NUMBER_OF_CLASSES = len(set(trainY) | set(testY))
    #trainY = to_categorical(trainY, nb_classes=NUMBER_OF_CLASSES)
    #testY  = to_categorical(testY , nb_classes=NUMBER_OF_CLASSES)

    print "generating input"
    sys.stdout.flush()

    # Network building
    net = tflearn.input_data([None, INPUT_DATA_DIM])

    print "embedding"
    sys.stdout.flush()

    net = tflearn.embedding(      net, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)

    print "creating lstm"
    sys.stdout.flush()

    net = tflearn.lstm(           net, OUTPUT_DIM, dropout=DROPOUT)

    print "connecting"
    sys.stdout.flush()

    net = tflearn.fully_connected(net, NUMBER_OF_CLASSES, activation=ACTIVATION)

    print "running regression"
    sys.stdout.flush()

    net = tflearn.regression(     net, optimizer=OPTIMIZER, learning_rate=RATE, loss=LOSS)

    # Training

    print "training"
    sys.stdout.flush()

    model = tflearn.DNN(net, tensorboard_verbose=VERBOSE)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=BATCH_SIZE)

    print "DONE"
    sys.stdout.flush()


def main():
    data_dir, positive_data_file, negative_data_file = sys.argv[1:]
    
    run(data_dir, positive_data_file, negative_data_file, test_freq=0.2)

if __name__ == '__main__':
    main()
