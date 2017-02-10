#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py

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
import math
import time

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb

import data_helpers

POWER             =     2.5
RATE              =     0.01
OUTPUT_DIM        =   128
BATCH_SIZE        =    50
DROPOUT           =     0.8
VERBOSE           =     3
NUM_EPOCHS        =    10
OPTIMIZER         = 'adam'
ACTIVATION        = 'relu'
ACTIVATION_LAST   = 'softmax'
LOSS              = 'categorical_crossentropy'

# IMDB Dataset loading
# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                 valid_portion=0.1)

def run(data_dir, positive_data_file, negative_data_file, test_freq=0.1):
    print "data {} positive {} negative {}".format( data_dir, positive_data_file, negative_data_file )

    cfg, (trainX, trainY), (testX , testY) = data_helpers.load_data_and_labels(data_dir, positive_data_file, negative_data_file, test_freq=0.2)

    #{'test_len': 3204, 'train_len': 12816,
    #'seg_end': 5125550, 'seqName': u'SL2.50ch01',
    #'end': 50001, 'ctime': u'Wed Feb  8 22:27:55 2017',
    #'seg_start': 1, 'seg_serial': 0, 'blockSize': 50000,
    #'start': 1, 'host': u'assembly', 'version': u'1.0',
    #'group': u'Euchromatin', 'serial': 1,
    #'groupId': 0,
    #'inputFile': u'/home/aflit001/dev/tf/npl/tomato/S_lycopersicum_chromosomes.2.50.fa'}
    #
    #print cfg
    #quit()

    INPUT_DATA_DIM    = cfg['blockSize']
    INPUT_DIM         = INPUT_DATA_DIM
    OUTPUT_DIM        = int(2 ** math.log(INPUT_DIM, 10))

    print "input  data dim           {:12,d}".format( INPUT_DATA_DIM )
    print "input  dim                {:12,d}".format( INPUT_DIM      )
    print "output dim                {:12,d}".format( OUTPUT_DIM     )
    print
    print "training sequences        {:12,d}".format( len(trainX) )
    print "test     sequences        {:12,d}".format( len(testX ) )


    if len(trainX) % BATCH_SIZE != 0:
        d = len(trainX) % BATCH_SIZE
        print "correcting train size {:12,d} to match batch size {:12,d} by {:12,d}".format(len(trainX), BATCH_SIZE, d)
        trainX = trainX[:-d]
        trainY = trainY[:-d]

    if len(testX) % BATCH_SIZE != 0:
        d = len(testX) % BATCH_SIZE
        print "correcting test  size {:12,d} to match batch size {:12,d} by {:12,d}".format(len(testX ), BATCH_SIZE, d)
        testX = testX[:-d]
        testY = testY[:-d]

    print "training sequences        {:12,d}".format( len(trainX) )
    print "test     sequences        {:12,d}".format( len(testX)  )

    # Converting labels to binary vectors
    NUMBER_OF_CLASSES = len(set(trainY) | set(testY))

    print "number of classes", NUMBER_OF_CLASSES
    trainY            = to_categorical(trainY, nb_classes=NUMBER_OF_CLASSES)
    testY             = to_categorical(testY , nb_classes=NUMBER_OF_CLASSES)

    print "generating input"
    sys.stdout.flush()

    # Network building
    imp = tflearn.input_data([None, INPUT_DATA_DIM], name="Input")

    print "embedding"
    emb = tflearn.embedding(imp, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)

    print "connecting"
    sys.stdout.flush()

    if True:
        print "creating conv"
        #http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
        filter_sizes   = [3, 5, 7]
        filter_std_dev =   0.1
        filter_const   =   0.1
        num_filters    =   4
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, OUTPUT_DIM, 1, num_filters]
                W            = tf.Variable(tf.truncated_normal(filter_shape, stddev=filter_std_dev), name="W")
                b            = tf.Variable(tf.constant(        filter_const, shape =[num_filters] ), name="b")
                conv         = tf.nn.conv2d(
                    tf.expand_dims(emb,-1) ,
                    W                      ,
                    strides  = [1, 1, 1, 1],
                    padding  = "VALID"     ,
                    name     = "conv_{}".format(i+1)
                )

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu_{}".format(i+1))
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h                      ,
                    ksize    = [1, INPUT_DIM - filter_size + 1, 1, 1],
                    strides  = [1, 1, 1, 1],
                    padding  = 'VALID'     ,
                    name     = "pool_{}".format(i+1)
                )

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool            = tf.concat(3, pooled_outputs)
        h_pool_flat       = tf.reshape(h_pool, [-1, num_filters_total])

    else:
        print "creating lstm"
        h_pool_flat = tflearn.lstm(emb, OUTPUT_DIM, dropout=DROPOUT)




    min_l=int(math.log(NUMBER_OF_CLASSES*2   , POWER) + 3)
    max_l=int(math.log(INPUT_DATA_DIM / POWER, POWER) + 1)

    print "power {} classes {} dim {} min_l {} max_l {}".format( POWER, NUMBER_OF_CLASSES, INPUT_DATA_DIM, min_l, max_l )

    net = None
    with tf.name_scope("fully_connected"):
        for p,l in enumerate(xrange(max_l, min_l, -1)):
            n = int(POWER**l)

            print " Layer #{:12,d}: power {:12,d} size {:12,d}".format(p, l, n)

            if net is None:
                net = h_pool_flat

            net = tflearn.fully_connected(net, n, activation=ACTIVATION, name="Layer_{}_{:02d}".format(ACTIVATION, p+1))
            net = tflearn.dropout(net, DROPOUT,                          name="Dropout_{:02d}".format(p+1))


    fully = tflearn.fully_connected(net, NUMBER_OF_CLASSES, activation=ACTIVATION_LAST, name="Layer_{}_LAST".format(ACTIVATION_LAST))

    print "running regression"
    sys.stdout.flush()

    reg = tflearn.regression(     fully, optimizer=OPTIMIZER, learning_rate=RATE, loss=LOSS, name="Regression")

    # Training

    print "training"
    sys.stdout.flush()


    model = tflearn.DNN(reg, tensorboard_verbose=VERBOSE)

    model.fit(trainX, trainY,
        validation_set = (testX, testY),
        show_metric    = True          ,
        n_epoch        = NUM_EPOCHS    ,
        shuffle        = True          ,
        batch_size     = BATCH_SIZE)

    outf  = 'my_model{:.0f}.tflearn'.format(time.time())
    print "saving model to {}".format(outf)
    model.save(outf)

    print "DONE"
    sys.stdout.flush()


def main():
    data_dir, positive_data_file, negative_data_file = sys.argv[1:]

    run(data_dir, positive_data_file, negative_data_file, test_freq=0.2)

if __name__ == '__main__':
    main()
