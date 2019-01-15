#coding=utf-8
from __future__ import division

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import tensorflow as tf
#from tensorflow.python import debug as tf_debug

import numpy as np
from simnetCNN2 import MLPTagCnn
from train2 import FLAGS



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
        usrq, usrq_tag = usrq.split("###")
        usrq = usrq.strip()
        usrq_tag = usrq_tag.strip()

        usrq = usrq.split()
        l1 = len(usrq)
        usrq = usrq[:seq_len]
        usrq = usrq + [pad_id] * (seq_len - len(usrq))

        usrq_tag = usrq_tag.split()
        assert len(usrq_tag) == l1
        usrq_tag = usrq_tag[:seq_len]
        usrq_tag = usrq_tag + [pad_id] * (seq_len - len(usrq_tag))
        dataLine.append((usrq, usrq_tag))

        for esq in esqs:
            esq = esq.strip()
            esq, esq_tag = esq.split("###")
            esq = esq.strip()
            esq_tag = esq_tag.strip()

            esq = esq.split()
            l2 = len(esq)
            esq = esq[:seq_len]
            esq = esq + [pad_id] * (seq_len - len(esq))

            esq_tag = esq_tag.split()
            assert len(esq_tag) == l2
            esq_tag = esq_tag[:seq_len]
            esq_tag = esq_tag + [pad_id] * (seq_len - len(esq_tag))
            dataLine.append((esq, esq_tag))

        data.append(dataLine)

    return data


def dev(ckpt, out):
    print "Loading data..."
    data = load_data(FLAGS.dev_data, FLAGS.max_sequence_length, FLAGS.pad_id)
    #assert len(data) == 800
    print "Data Size:", len(data)

    with tf.device('/gpu:4'):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            with sess.as_default():
                cnn = MLPTagCnn(sequence_length=FLAGS.max_sequence_length,
                                vocab_size=FLAGS.vocab_size,
                                tagVocab_size=FLAGS.tag_vocab_size,
                                embedding_size=FLAGS.embedding_dim,
                                tagEmb_size=FLAGS.tagEmb_dim,
                                window_size=list(map(int, FLAGS.window_sizes.split(","))),
                                num_filters=FLAGS.num_filters,
                                hidden_size=FLAGS.hidden_size,
                                margin=FLAGS.margin,
                                dropout_keep_prob=1.0,
                                l2_reg_lambda=FLAGS.l2_reg_lambda,
                                is_training=False)

                saver = tf.train.Saver(tf.global_variables())

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                print("*" * 20 + "\nReading model parameters from %s \n" % ckpt + "*" * 20)

                saver.restore(sess, ckpt)

                index = []
                for sample in data:

                    usrq, usrq_tag = sample[0]
                    ess = sample[1:]

                    usrqs = []
                    usrq_tags = []
                    esqs = []
                    esq_tags = []

                    for esq in ess:
                        q, tag = esq
                        usrqs.append(usrq)
                        usrq_tags.append(usrq_tag)
                        esqs.append(q)
                        esq_tags.append(tag)

                    feed_dict = {
                        cnn.input_x_1: usrqs,
                        cnn.input_x_2: esqs,
                        cnn.input_x_11: usrq_tags,
                        cnn.input_x_22: esq_tags
                    }


                    score = tf.reshape(cnn.output_prob, [-1])

                    ind = tf.argmax(score, 0)

                    i = sess.run(ind, feed_dict)

                    index.append(i)


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



