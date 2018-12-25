#coding=utf-8
from __future__ import division

import sys

import tensorflow as tf
#from tensorflow.python import debug as tf_debug

import numpy as np
from QACNN_2 import QACNN
from train import FLAGS



def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def load_data(filename, seq_len, pad_id):
    pad_id = int(pad_id)
    data = []
    f1 = open(filename, 'r')
    for line in f1.readlines():
        dataLine = []
        line = line.strip()
        line = line.split("\t")
        usrq = line[0]
        esqs = line[1:]

        usrq = usrq.strip()
        usrq = usrq.split()
        usrq = usrq[:seq_len]
        usrq = usrq + [pad_id] * (seq_len - len(usrq))
        dataLine.append(usrq)

        for esq in esqs:
            esq = esq.strip()
            esq = esq.split()
            esq = esq[:seq_len]
            esq = esq + [pad_id] * (seq_len - len(esq))
            dataLine.append(esq)

        data.append(dataLine)

    return data


def dev(ckpt, out):
    print "Loading data..."
    data = load_data(FLAGS.dev_data, FLAGS.max_sequence_length, FLAGS.pad_id)
    #assert len(data) == 800
    print "Data Size:", len(data)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        with sess.as_default():
            cnn = QACNN(sequence_length=FLAGS.max_sequence_length,
						vocab_size=FLAGS.vocab_size,
						embedding_size=FLAGS.embedding_dim,
						filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                        filter_sizes2=list(map(int, FLAGS.filter_sizes2.split(","))),
						num_filters=FLAGS.num_filters,
                        dropout_keep_prob=1.0,
						l2_reg_lambda=FLAGS.l2_reg_lambda,
                        is_training=False)

            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            print("*" * 20 + "\nReading model parameters from %s \n" % ckpt + "*" * 20)

            saver.restore(sess, ckpt)

            h = 1
            index = []
            for sample in data:

                usrq = sample[0]
                esqs = sample[1:]
                usrqs = []
                for esq in esqs:
                    usrqs.append(usrq)

                if np.array(usrqs).shape[-1] != FLAGS.max_sequence_length or np.array(esqs).shape[-1] != FLAGS.max_sequence_length:
                    print h
                    index.append(-999)
                    continue


                feed_dict = {
                    cnn.input_x_1: usrqs,
                    cnn.input_x_2: esqs
                }


                score = tf.reshape(cnn.score12, [-1])

                ind = tf.argmax(score, 0)

                i = sess.run(ind, feed_dict)

                index.append(i)

                h += 1

            assert len(index) == len(data)

    f = open(out, 'w')
    for i in index:
        f.write(str(i) + "\n")




if __name__ == "__main__":
    print_args(FLAGS)

    args = sys.argv
    ckpt = args[1]
    out = args[2]
    dev(ckpt, out)



