#coding=utf-8
from __future__ import division

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import time

import tensorflow as tf
#from tensorflow.python import debug as tf_debug

from simnetCNN3 import MLPTagPosiCnn
import data_loader_2 as data_loader
import datetime



# Data
tf.flags.DEFINE_string("train_file", "data/id_pairwise_data_xmwjinfu_2.0", "train data (id)")
tf.flags.DEFINE_string("dev_data", "data/xinwang_test/id_xinwang_test_jinfu_2.0", "dev data (id)")
tf.flags.DEFINE_integer("vocab_size", 10000, "vocab.txt")
tf.flags.DEFINE_integer("tag_vocab_size", 3, "vocab.txt")
tf.flags.DEFINE_integer("pad_id", 0, "id for <pad> token in character list")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("window_sizes", "2,3,5", "window size")
tf.flags.DEFINE_float("margin", 0.7, "learning_rate (default: 0.1)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes2", "3,5,7", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 512, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.6, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_epoch", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400000, "Save model after this many steps (default: 100)")

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

def load_dev_data(filename, seq_len, pad_id):
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


def get_result(indexs):
    f2 = open("data/xinwang_test/kID", "r")
    kIDs = f2.readlines()
    kIDs = [kID.strip() for kID in kIDs]
    #print kIDs

    f3 = open("data/xinwang_test/true_kID", 'r')
    labels = f3.readlines()
    labels = [label.strip() for label in labels]
    #print labels

    #print len(labels), len(kIDs), len(indexs)
    assert len(labels) == len(kIDs) == len(indexs) == 800

    count = 0
    right = 0
    j = 0
    for i in range(800):
        label = labels[i]
        es_IDs = kIDs[i]
        es_IDs = es_IDs.strip()
        es_IDs = es_IDs.split("\t")
        #print i
        #assert len(es_IDs) == 12

        if label not in es_IDs:
            #print i, " line es not call back"
            j += 1
            continue

        count += 1
        index = indexs[i]
        ID = es_IDs[index]
        if ID == label:
            right += 1

    print j, "lines are not call back"
    return right / count





def train():
    print "Loading data..."
    data = data_loader.read_data2(FLAGS.train_file, FLAGS.max_sequence_length, FLAGS.pad_id)
    print "Data Size:", len(data)

    print "Loading dev data..."
    dev_data = load_dev_data(FLAGS.dev_data, FLAGS.max_sequence_length, FLAGS.pad_id)
    assert len(dev_data) == 800
    print "Dev data Size:", len(dev_data)

    with tf.device('/gpu:5'):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            with sess.as_default():
                cnn = MLPTagPosiCnn(sequence_length=FLAGS.max_sequence_length,
                            vocab_size=FLAGS.vocab_size,
                            tagVocab_size=FLAGS.tag_vocab_size,
                            embedding_size=FLAGS.embedding_dim,
                            window_size=list(map(int, FLAGS.window_sizes.split(","))),
                            num_filters=FLAGS.num_filters,
                            hidden_size=FLAGS.hidden_size,
                            margin=FLAGS.margin,
                            l2_reg_lambda=FLAGS.l2_reg_lambda)

                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]

                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_name, timestamp))
                print("Writing to {}\n".format(out_dir))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                ############restore embedding##################
                if FLAGS.restore_pretrained_embedding:
                    embedding_var_name = "embedding/embedding_W:0"

                    # 得到该网络中，所有可以加载的参数
                    variables = tf.contrib.framework.get_variables_to_restore()

                    variables_to_resotre = [v for v in variables if v.name == embedding_var_name]

                    saver = tf.train.Saver(variables_to_resotre)

                    saver.restore(sess, FLAGS.pretrained_embeddings_path)
                    print "Restore embeddings from", FLAGS.pretrained_embeddings_path

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                restore = FLAGS.restore_model
                if restore:
                    saver.restore(sess, FLAGS.model_path)
                    print("*" * 20 + "\nReading model parameters from %s \n" % FLAGS.model_path + "*" * 20)
                else:
                    print("*" * 20 + "\nCreated model with fresh parameters.\n" + "*" * 20)


                def train_step(q_batch, pos_batch, neg_batch, q_tag_batch, pos_tag_batch, neg_tag_batch, epoch):

                    """
                    A single training step
                    """

                    feed_dict = {
                        cnn.input_x_1: q_batch,
                        cnn.input_x_2: pos_batch,
                        cnn.input_x_3: neg_batch,
                        cnn.input_x_11: q_tag_batch,
                        cnn.input_x_22: pos_tag_batch,
                        cnn.input_x_33: neg_tag_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        cnn.is_training: True
                    }

                    _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print "{}: Epoch {} step {}, loss {:g}".format(time_str, epoch, step, loss)



                def dev_step():

                    index = []
                    for sample in dev_data:

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
                            cnn.input_x_22: esq_tags,
                            cnn.dropout_keep_prob: 1.0,
                            cnn.is_training: False
                        }

                        score = tf.reshape(cnn.output_prob, [-1])

                        ind = tf.argmax(score, 0)

                        i = sess.run(ind, feed_dict)

                        index.append(i)

                    assert len(index) == len(dev_data)

                    result = get_result(index)

                    return result



                # Generate batches
                batches = data_loader.batch_iter(data, FLAGS.batch_size, FLAGS.max_epoch, True)

                num_batches_per_epoch = int((len(data)) / FLAGS.batch_size) + 1

                # Training loop. For each batch...
                epoch = 0
                max_dev_res = 0
                max_step = 0
                for batch in batches:
                    q_batch = batch[:, 0]
                    pos_batch = batch[:, 1]
                    neg_batch = batch[:, 2]
                    q_tag_batch = batch[:, 3]
                    pos_tag_batch = batch[:, 4]
                    neg_tag_batch = batch[:, 5]

                    train_step(q_batch, pos_batch, neg_batch, q_tag_batch, pos_tag_batch, neg_tag_batch, epoch)
                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % num_batches_per_epoch == 0:
                        epoch += 1

                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        dev_res = dev_step()
                        print "Evaluation result is: ", dev_res
                        if dev_res > max_dev_res:
                            max_dev_res = dev_res
                            max_step = current_step
                        print "Untill now, the max dev result is", max_dev_res, "in", max_step, "step."



if __name__ == "__main__":
    print_args(FLAGS)

    train()

