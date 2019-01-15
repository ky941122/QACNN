#! /usr/bin/env python

import tensorflow as tf
import os
import shutil
from simnetCNN2 import MLPTagCnn

# Parameters
# ==================================================
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Saving Parameters
tf.flags.DEFINE_integer("model_version", 4, "model version")
tf.flags.DEFINE_string("pb_save_path", "./pairwise_model", "pb save file.")
tf.flags.DEFINE_string("checkpoint_path", "xmwjinfu_ckpt/model-2800000", "ckpt path")

# Model Hyperparameters
tf.flags.DEFINE_integer("vocab_size", 10000, "vocab.txt")
tf.flags.DEFINE_integer("tag_vocab_size", 3, "vocab.txt")
tf.flags.DEFINE_integer("max_sequence_length", 20, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("tagEmb_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("window_sizes", "2,3,5", "window size")
tf.flags.DEFINE_float("margin", 0.7, "learning_rate (default: 0.1)")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 512, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")




FLAGS = tf.flags.FLAGS

def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def export():
    checkpoint_file = FLAGS.checkpoint_path
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

            cnn = MLPTagCnn(sequence_length=FLAGS.max_sequence_length,
                            vocab_size=FLAGS.vocab_size,
                            tagVocab_size=FLAGS.tag_vocab_size,
                            embedding_size=FLAGS.embedding_dim,
                            tagEmb_size=FLAGS.tagEmb_dim,
                            window_size=list(map(int, FLAGS.window_sizes.split(","))),
                            num_filters=FLAGS.num_filters,
                            hidden_size=FLAGS.hidden_size,
                            margin=FLAGS.margin,
                            l2_reg_lambda=FLAGS.l2_reg_lambda)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x_1 = graph.get_operation_by_name("input_q").outputs[0]
            input_x_11 = graph.get_operation_by_name("input_q_tag").outputs[0]

            input_x_2 = graph.get_operation_by_name("input_pos").outputs[0]
            input_x_22 = graph.get_operation_by_name("input_pos_tag").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("Out/output_prob").outputs[0]

            model_version = FLAGS.model_version
            version_export_path = os.path.join(FLAGS.pb_save_path, str(model_version))
            if os.path.exists(version_export_path):
                shutil.rmtree(version_export_path)
            print("Exporting trained model to ", version_export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(version_export_path)

            tensor_input_data_query = tf.saved_model.utils.build_tensor_info(input_x_1)
            tensor_input_data_query_tag = tf.saved_model.utils.build_tensor_info(input_x_11)

            tensor_input_data_candidate = tf.saved_model.utils.build_tensor_info(input_x_2)
            tensor_input_data_candidate_tag = tf.saved_model.utils.build_tensor_info(input_x_22)

            tensor_output_scores = tf.saved_model.utils.build_tensor_info(scores)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_data_query': tensor_input_data_query,
                            "input_data_query_tag": tensor_input_data_query_tag,
                            "input_data_candidate": tensor_input_data_candidate,
                            "input_data_candidate_tag": tensor_input_data_candidate_tag
                            },
                    outputs={'output_scores': tensor_output_scores
                             },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'ics-delex-search': prediction_signature
                }
            )
            builder.save()
            print('Done exporting!')


if __name__ == "__main__":
    print_args(FLAGS)
    export()

