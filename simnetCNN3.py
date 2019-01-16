import tensorflow as tf

class MLPTagPosiCnn():

    def __init__(self, sequence_length, vocab_size, tagVocab_size, embedding_size, window_size, num_filters, hidden_size, margin,
                l2_reg_lambda=0.0):

        self.sequence_length = sequence_length
        self.win_size = window_size
        self.num_filters = num_filters
        self.margin = margin
        self.hidden_size = hidden_size
        self.num_filters_total = self.num_filters * len(self.win_size)
        self.l2_reg_lambda = l2_reg_lambda

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_q")
        self.input_x_11 = tf.placeholder(tf.int32, [None, sequence_length], name="input_q_tag")

        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos")
        self.input_x_22 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos_tag")

        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_neg")
        self.input_x_33 = tf.placeholder(tf.int32, [None, sequence_length], name="input_neg_tag")

        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

        initializer = tf.keras.initializers.he_normal()

        # Embedding layer
        with tf.variable_scope("Embedding"):
            self.embedding_W = tf.get_variable("embedding",shape=[vocab_size, embedding_size])

            input_q = tf.nn.embedding_lookup(self.embedding_W, self.input_x_1)
            input_pos = tf.nn.embedding_lookup(self.embedding_W, self.input_x_2)
            input_neg = tf.nn.embedding_lookup(self.embedding_W, self.input_x_3)

        with tf.variable_scope("Tag_Embedding"):
            self.tag_embedding = tf.get_variable("tag_embedding", shape=[tagVocab_size, embedding_size])

            tag_q = tf.nn.embedding_lookup(self.tag_embedding, self.input_x_11)
            tag_pos = tf.nn.embedding_lookup(self.tag_embedding, self.input_x_22)
            tag_neg = tf.nn.embedding_lookup(self.tag_embedding, self.input_x_33)

        input_q = input_q + tag_q
        input_pos = input_pos + tag_pos
        input_neg = input_neg + tag_neg

        with tf.variable_scope("Position_Embedding"):
            self.position_embedding = tf.get_variable("position_embedding", shape=[sequence_length, embedding_size])

            q_position_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x_1)[1]), 0),
                                     [tf.shape(self.input_x_1)[0], 1])
            pos_position_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x_2)[1]), 0),
                                     [tf.shape(self.input_x_2)[0], 1])
            neg_position_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x_3)[1]), 0),
                                     [tf.shape(self.input_x_3)[0], 1])

            position_q = tf.nn.embedding_lookup(self.position_embedding, q_position_ind)
            position_pos = tf.nn.embedding_lookup(self.position_embedding, pos_position_ind)
            position_neg = tf.nn.embedding_lookup(self.position_embedding, neg_position_ind)

        input_q = input_q + position_q
        input_pos = input_pos + position_pos
        input_neg = input_neg + position_neg

        with tf.variable_scope("CNN"):

            self.kernels = []
            for i, filter_size in enumerate(self.win_size):
                filter_shape = [filter_size, embedding_size, 1, self.num_filters]
                W = tf.get_variable("conv_W_{}".format(filter_size), shape=filter_shape,
                                    initializer=initializer)
                b = tf.get_variable("conv_b_{}".format(filter_size), shape=[self.num_filters],
                                    initializer=initializer)
                self.kernels.append((W, b))

            conv_q = self.cnn(input_q)
            conv_pos = self.cnn(input_pos)
            conv_neg = self.cnn(input_neg)

        with tf.variable_scope("FC"):
            self.fc_w = tf.get_variable("fc", [self.num_filters_total, self.hidden_size])
            self.fc_b = tf.get_variable("bias", [self.hidden_size])

            hid1_q = self.fc(conv_q)
            hid1_pos = self.fc(conv_pos)
            hid1_neg = self.fc(conv_neg)

        with tf.variable_scope("Out"):
            pred_pos = self.cos(hid1_q, hid1_pos)
            pred_neg = self.cos(hid1_q, hid1_neg)

            self.output_prob = tf.identity(pred_pos, name="output_prob")

            print "output_prob shape: ", self.output_prob.shape

            self.loss = tf.reduce_mean(tf.maximum(0., pred_neg + self.margin - pred_pos))

        self._model_stats()



    def cnn(self, emb):
        emb_expanded = tf.expand_dims(emb, -1)

        output = []
        for i, filter_size in enumerate(self.win_size):
            conv = tf.nn.conv2d(
                emb_expanded,
                self.kernels[i][0],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="poll-1"
            )
            output.append(pooled)
        pooled_reshape = tf.reshape(tf.concat(output, 3), [-1, self.num_filters_total])
        pooled_flat = tf.nn.dropout(pooled_reshape, self.dropout_keep_prob)
        return pooled_flat


    def fc(self, input):
        out_without_bias = tf.matmul(input, self.fc_w)
        output = tf.nn.bias_add(out_without_bias, self.fc_b)
        out = tf.nn.relu(output)
        return out

    def cos(self, input_a, input_b):
        norm_a = tf.nn.l2_normalize(input_a, dim=1)
        norm_b = tf.nn.l2_normalize(input_b, dim=1)
        cos_sim = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_a, norm_b), 1), -1)
        return cos_sim



    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))


