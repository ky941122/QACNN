#coding=utf-8
import tensorflow as tf 
import numpy as np 
import  pickle
import time
class QACNN():
    
    def __init__(self, sequence_length,vocab_size, embedding_size,filter_sizes, filter_sizes2, num_filters,
                 dropout_keep_prob=1.0,l2_reg_lambda=0.0, is_training=True):
        self.sequence_length=sequence_length
        self.filter_sizes=filter_sizes
        self.filter_sizes2 = filter_sizes2
        self.num_filters=num_filters
        self.l2_reg_lambda=l2_reg_lambda

        if is_training:
            self.dropout_keep_prob = dropout_keep_prob
        else:
            self.dropout_keep_prob = 1.0

        self.is_training = is_training

        self.embedding_size=embedding_size
        self.num_filters_total=self.num_filters * len(self.filter_sizes2)

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_3")

        initializer = tf.keras.initializers.he_normal()
        
        # Embedding layer
        self.updated_paras=[]
        with tf.variable_scope("embedding"):
            self.Embedding_W = tf.get_variable("embedding_W",
                                               shape=[vocab_size, embedding_size])
            self.updated_paras.append(self.Embedding_W)

        self.kernels=[]        
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("first_conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, self.num_filters]
                W = tf.get_variable("first_conv_W_{}".format(filter_size), shape=filter_shape,
                                    initializer=initializer)
                b = tf.get_variable("first_conv_b_{}".format(filter_size), shape=[self.num_filters],
                                    initializer=initializer)
                self.kernels.append((W,b))
                self.updated_paras.append(W)
                self.updated_paras.append(b)


        self.kernels_2 = []
        for i, filter_size in enumerate(self.filter_sizes2):
            with tf.variable_scope("second_conv_maxpool_%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, num_filters * len(self.filter_sizes), self.num_filters]
                W = tf.get_variable("second_conv_W_{}".format(filter_size), shape=filter_shape, initializer=initializer)
                b = tf.get_variable("second_conv_b_{}".format(filter_size), shape=[self.num_filters],
                                    initializer=initializer)
                self.kernels_2.append((W, b))
                self.updated_paras.append(W)
                self.updated_paras.append(b)


        self.l2_loss = tf.constant(0.0)
        for para in self.updated_paras:
            self.l2_loss+= tf.nn.l2_loss(para)
        

        with tf.variable_scope("output"):
            q  =self.getRepresentation(self.input_x_1)
            pos=self.getRepresentation(self.input_x_2)
            neg=self.getRepresentation(self.input_x_3)

            self.score12 = self.cosine(q,pos)
            self.score13 = self.cosine(q,neg)

            self.score = tf.identity(self.score12, name="score")

            self.positive= tf.reduce_mean(self.score12)
            self.negative= tf.reduce_mean( self.score13)

        self._model_stats()  # print model statistics info

        self.losses = tf.maximum(0.0, tf.subtract(0.15, tf.subtract(self.score12, self.score13)))
        self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * self.l2_loss

        self.correct = tf.equal(0.0, self.losses)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")


    def getRepresentation(self,sentence):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        embedded_chars_1 = tf.nn.dropout(embedded_chars_1, min(1.0, self.dropout_keep_prob+0.2))



        output_1=[]
        for i, filter_size in enumerate(self.filter_sizes): 
            conv1 = tf.nn.conv1d(
                embedded_chars_1,
                self.kernels[i][0],
                stride=1,
                padding='SAME',
                name="conv-1"
            )
            #conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training)  # axis定的是channel在的维度。
            h = tf.nn.relu(tf.nn.bias_add(conv1, self.kernels[i][1]), name="relu-1")
            output_1.append(h)

        conv2_input = tf.concat(output_1, -1)
        #print "conv2_input shape:", conv2_input.shape




        output = []
        for i, filter_size in enumerate(self.filter_sizes2):
            conv2 = tf.nn.conv1d(conv2_input, self.kernels_2[i][0], stride=1, padding="VALID", name="conv-2")
            #print "conv2 shape:", conv2.shape
            h = tf.nn.relu(tf.nn.bias_add(conv2, self.kernels_2[i][1]), name="relu-2")
            pooled = tf.layers.max_pooling1d(h, pool_size=self.sequence_length - filter_size + 1, strides=1, name="pool")
            output.append(pooled)
        pooled_reshape = tf.reshape(tf.concat(output, -1), [-1, self.num_filters_total])
        #print "pooled_reshape shape:", pooled_reshape.shape
        pooled_flat = tf.nn.dropout(pooled_reshape, self.dropout_keep_prob)
        return pooled_flat


    def cosine(self,q,a):

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))

        pooled_mul_12 = tf.reduce_sum(tf.multiply(q, a), 1)
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2)+1e-8, name="scores")
        return score 


    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))



