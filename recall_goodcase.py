#coding=utf-8
from __future__ import division

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import time

import tensorflow as tf
import numpy as np
#from tensorflow.python import debug as tf_debug

from QACNN_2 import QACNN
import data_loader_2 as data_loader
import datetime
import json



# Data
tf.flags.DEFINE_string("train_file", "data/id_pairwise_data_shuf", "train data (id)")
tf.flags.DEFINE_string("dev_data", "_data/id_goodcase_es", "dev data (id)")
tf.flags.DEFINE_integer("vocab_size", 19423, "vocab.txt")
tf.flags.DEFINE_integer("pad_id", 0, "id for <pad> token in character list")

tf.flags.DEFINE_string("simq_data", "recall/id_allSimWithQEnt_2", "train data (id)")
tf.flags.DEFINE_string("goodcase_data", "recall/id_goodcase", "train data (id)")
tf.flags.DEFINE_string("true_kId", "recall/goodcase_labels", "train data (id)")
tf.flags.DEFINE_string("simqkID", "recall/simQkID", "train data (id)")



# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("window_sizes", "2,3,5", "window size")
tf.flags.DEFINE_float("margin", 0.5, "learning_rate (default: 0.1)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes2", "3,5,7", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 512, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_epoch", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Save Model
tf.flags.DEFINE_string("model_name", "PairwiseCnn", "model name")
tf.flags.DEFINE_integer("num_checkpoints", 2000, "checkpoints number to save")
tf.flags.DEFINE_boolean("restore_model", False, "Whether restore model or create new parameters")
tf.flags.DEFINE_string("model_path", "runs", "Restore which model")
tf.flags.DEFINE_boolean("restore_pretrained_embedding", False, "Whether restore pretrained embedding")
tf.flags.DEFINE_string("pretrained_embeddings_path", "checkpoints/embedding", "Restore pretrained embedding")




FLAGS = tf.flags.FLAGS


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
        line = line.strip()
        ids = line.split(" ")
        one = []
        for id in ids:
            id = id.strip()
            id = int(id)
            one.append(id)

        one = one[:seq_len]
        one = one + [pad_id] * (seq_len - len(one))

        data.append(one)

    return data


def recall(ckpt):
    print "Loading data..."
    simqs = load_data(FLAGS.simq_data, FLAGS.max_sequence_length, FLAGS.pad_id)
    goodcase = load_data(FLAGS.goodcase_data, FLAGS.max_sequence_length, FLAGS.pad_id)
    print "SimQ Size:", len(simqs)
    print "GoodCase Size:", len(goodcase)

    f0 = open(FLAGS.true_kId, "r")
    truekId = f0.readlines()
    truekId = [kid.strip() for kid in truekId]

    f1 = open(FLAGS.simqkID, "r")
    simqkID = f1.readlines()
    simqkID = [d.strip() for d in simqkID]

    assert len(truekId) == len(goodcase)
    assert len(simqkID) == len(simqs)

    with tf.device('/gpu:5'):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
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

                # saver = tf.train.import_meta_graph("{}.meta".format(ckpt))

                saver = tf.train.Saver(tf.global_variables())

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                saver.restore(sess, ckpt)

                print("*" * 20 + "\nReading model parameters from %s \n" % ckpt + "*" * 20)

                encode_simqs = []

                num_batches_per_epoch = int((len(simqs)) / FLAGS.batch_size) + 1
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * FLAGS.batch_size
                    end_index = min((batch_num + 1) * FLAGS.batch_size, len(simqs))
                    simq_batch = simqs[start_index: end_index]


                    feed_dict = {
                        cnn.input_x_1:simq_batch
                    }

                    #enc = tf.reshape(cnn.q, [-1])
                    enc = sess.run(cnn.q, feed_dict)

                    encode_simqs.extend(enc)

                    print batch_num

                assert len(encode_simqs) == len(simqs)

                print encode_simqs

                # encode_simqs = [list(e) for e in encode_simqs]
                #
                #
                # wriTe = json.dumps(encode_simqs)
                #
                # f99 = open("encode_json", "w")
                #
                # f99.write(wriTe)
                #

                num_batches_per_epoch = int((len(encode_simqs)) / FLAGS.batch_size) + 1

                cnt = 0
                cnt60 = 0
                cnt30 = 0
                for i in range(len(goodcase)):
                    gc = goodcase[i]
                    label = truekId[i]
                    scores = []
                    for batch_num in range(num_batches_per_epoch):
                        start_index = batch_num * FLAGS.batch_size
                        end_index = min((batch_num + 1) * FLAGS.batch_size, len(encode_simqs))
                        simq_batch = encode_simqs[start_index : end_index]
                        gc_batch = [gc] * len(simq_batch)

                        feed_dict = {
                            cnn.enc_simq: simq_batch,
                            cnn.input_x_2: gc_batch
                        }

                        score0 = tf.reshape(cnn.score0, [-1])

                        score = sess.run(score0, feed_dict)

                        scores.extend(score)

                    assert len(scores) == len(simqkID)

                    ind60 = sess.run(tf.nn.top_k(scores, 60)[1])
                    ind30 = sess.run(tf.nn.top_k(scores, 30)[1])

                    recall_kid_30 = np.array(simqkID)[ind30]
                    if label in recall_kid_30:
                        cnt30 += 1

                    recall_kid_60 = np.array(simqkID)[ind60]
                    if label in recall_kid_60:
                        cnt60 += 1

                    cnt = i + 1

                    prop_30 = cnt30 / cnt
                    prop_60 = cnt60 / cnt

                    if cnt < 50 or cnt % 100 == 0:
                        print "cnt:", cnt, "recall 30 proportion:", prop_30, "recall 60 proportion:", prop_60




if __name__ == "__main__":
    ckpt = "bigcnn/model-180000"
    recall(ckpt)







