#coding=utf-8
from __future__ import division

import os
import time

import tensorflow as tf
#from tensorflow.python import debug as tf_debug

from simnetCNN import MLPCnn
import data_loader_2 as data_loader
import datetime



# Data
tf.flags.DEFINE_string("train_file", "data/id_pairwise_data_shuf", "train data (id)")
tf.flags.DEFINE_string("dev_data", "_data/id_goodcase_es", "dev data (id)")
tf.flags.DEFINE_integer("vocab_size", 16458, "vocab.txt")
tf.flags.DEFINE_integer("pad_id", 0, "id for <pad> token in character list")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 10, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("window_sizes", "2,3,5", "window size")
tf.flags.DEFINE_float("margin", 0.5, "learning_rate (default: 0.1)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes2", "3,5,7", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
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


def train():
    print "Loading data..."
    data = data_loader.read_data(FLAGS.train_file, FLAGS.max_sequence_length, FLAGS.pad_id)
    print "Data Size:", len(data)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        with sess.as_default():
            cnn = MLPCnn(sequence_length=FLAGS.max_sequence_length,
						vocab_size=FLAGS.vocab_size,
						embedding_size=FLAGS.embedding_dim,
						window_size=list(map(int, FLAGS.window_sizes.split(","))),
						num_filters=FLAGS.num_filters,
                        hidden_size=FLAGS.hidden_size,
                        margin=FLAGS.margin,
                        dropout_keep_prob=FLAGS.dropout_keep_prob,
						l2_reg_lambda=FLAGS.l2_reg_lambda,
                        is_training=True)

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


            def train_step(q_batch, pos_batch, neg_batch, epoch):

                """
                A single training step
                """

                feed_dict = {
                    cnn.input_x_1: q_batch,
                    cnn.input_x_2: pos_batch,
                    cnn.input_x_3: neg_batch
                }

                _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print "{}: Epoch {} step {}, loss {:g}".format(time_str, epoch, step, loss)

            # Generate batches
            batches = data_loader.batch_iter(data, FLAGS.batch_size, FLAGS.max_epoch, True)

            num_batches_per_epoch = int((len(data)) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            epoch = 0
            for batch in batches:
                q_batch = batch[:, 0]
                pos_batch = batch[:, 1]
                neg_batch = batch[:, 2]
                train_step(q_batch, pos_batch, neg_batch, epoch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % num_batches_per_epoch == 0:
                    epoch += 1

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



if __name__ == "__main__":
    print_args(FLAGS)

    train()

