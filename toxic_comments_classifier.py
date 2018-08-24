import random
import csv
import nltk
import numpy as np
import pickle
import os
import tensorflow as tf
from nltk.tokenize import word_tokenize

nltk.download('punkt')


class TokenEmbedding:
    def __init__(self, embedding, index):
        self._embedding = embedding
        self._index = index

    @property
    def index(self):
        return self._index

    @property
    def embedding(self):
        return self._embedding


class ToxicComment:
    def __init__(self, csv_row, glove_model, comment_max_length):
        self._tokens = word_tokenize(csv_row['comment_text'].strip('"'))
        self._labels = np.array([float(csv_row['toxic']), float(csv_row['severe_toxic']), float(csv_row['obscene']), float(csv_row['threat']), float(csv_row['insult']), float(csv_row['identity_hate'])])
        self._indexed_tokens = np.zeros(shape=[comment_max_length], dtype=np.int32)
        self._token_count = min(len(self._tokens), comment_max_length)
        for i, token in enumerate(self._tokens):
            if i < comment_max_length:
                token = token.lower()
                index = glove_model.token_to_embedding['something'].index
                if token in glove_model.token_to_embedding:
                    index = glove_model.token_to_embedding[token].index
                self._indexed_tokens[i] = index
            else:
                break

    @property
    def tokens(self):
        return self._tokens

    @property
    def labels(self):
        return self._labels

    @property
    def indexed_tokens(self):
        return self._indexed_tokens

    @property
    def token_count(self):
        return self._token_count


class GloveModel:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self._tokens = []
        self._embeddings = []
        self._token_to_embedding = {}

    def append_model(self, token, embedding):
        self._tokens.append(token)
        self._token_to_embedding[token] = TokenEmbedding(embedding, len(self._embeddings))
        self._embeddings.append(embedding)

    def load_glove_model(self, glove_model_file_path):
        valid_model_on_disk = False
        if os.path.exists('.\\glove_model_tokens.pickle'):
            if os.path.exists('.\\glove_model_embeddings.pickle'):
                if os.path.exists('.\\glove_model_token_to_embedding.pickle'):
                    valid_model_on_disk = True

        if valid_model_on_disk is True:
            with open('.\\glove_model_tokens.pickle', 'rb') as handle:
                self._tokens = pickle.load(handle)
            with open('.\\glove_model_embeddings.pickle', 'rb') as handle:
                self._embeddings = pickle.load(handle)
            with open('.\\glove_model_token_to_embedding.pickle', 'rb') as handle:
                self._token_to_embedding = pickle.load(handle)
        else:
            self.initialize()
            file = open(glove_model_file_path, encoding="utf-8")
            _, embedding = self.parse_line(self.peek_line(file))
            self.append_model("<unk>", np.zeros(embedding.size, dtype=float))
            for line in file:
                token, embedding = self.parse_line(line)
                self.append_model(token, embedding)

            with open('.\\glove_model_tokens.pickle', 'wb') as handle:
                pickle.dump(self._tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('.\\glove_model_embeddings.pickle', 'wb') as handle:
                pickle.dump(self._embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('.\\glove_model_token_to_embedding.pickle', 'wb') as handle:
                pickle.dump(self._token_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def peek_line(file):
        pos = file.tell()
        line = file.readline()
        file.seek(pos)
        return line

    @staticmethod
    def parse_line(line):
        splitted_line = line.split()
        token = splitted_line[0]
        embedding = np.array([float(val) for val in splitted_line[1:]])
        return token, embedding

    @property
    def tokens(self):
        return self._tokens

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def token_to_embedding(self):
        return self._token_to_embedding

    @property
    def vocabulary_size(self):
        return len(self._embeddings)

    @property
    def embedding_dimension(self):
        if not self._embeddings:
            return 0
        return len(self._embeddings[0])


class ToxicComments:
    def __init__(self, word_embeddings_model, id, comment_max_length):
        self._id = id
        self._comment_max_length = comment_max_length
        self._toxic_comments_pickle_file_path = '.\\toxic_comments_' + id + '.pickle'
        self._word_embeddings_model = word_embeddings_model
        self.initialize()

    def initialize(self):
        self._epoch_size = 0
        self._current_epoch = 0
        self._current_batch = 0
        self._toxic_comments = []

    def load_toxic_comments(self, toxic_comments_file_path):
        self.initialize()
        self.load_toxic_comments_file(toxic_comments_file_path)
        self._epoch_size = len(self._toxic_comments)
        self.shuffle()

    def load_toxic_comments_file(self, toxic_comments_file_path):
        if os.path.exists(self._toxic_comments_pickle_file_path):
            with open(self._toxic_comments_pickle_file_path, 'rb') as handle:
                self._toxic_comments = pickle.load(handle)
        else:
            with open(toxic_comments_file_path, encoding="ISO-8859-1") as csv_file:
                dict_reader = csv.DictReader(csv_file)
                for csv_row in dict_reader:
                    if csv_row['toxic'] != '-1':
                        toxic_comment = ToxicComment(csv_row, self._word_embeddings_model, self._comment_max_length)
                        self._toxic_comments.append(toxic_comment)

            with open(self._toxic_comments_pickle_file_path, 'wb') as handle:
                pickle.dump(self._toxic_comments, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self._toxic_comments

    def shuffle(self):
        random.shuffle(self._toxic_comments)
        self.cursor = 0

    def get_next_batch(self, batch_size):
        start_range = self.cursor
        end_range = self.cursor + batch_size
        next_batch_indexed_tokens_tensor = self.get_next_batch_indexed_tokens(start_range, end_range, batch_size)
        next_batch_labels_tensor = self.get_next_batch_labels(start_range, end_range, batch_size)
        next_batch_sequence_length = self.get_next_batch_sequence_length(start_range, end_range, batch_size)
        self.cursor += batch_size

        self._current_batch = self._current_batch + 1
        new_epoch = False
        if self.cursor + batch_size > self._epoch_size:
            self._current_epoch += 1
            self._current_batch = 0
            self.shuffle()
            new_epoch = True

        return next_batch_indexed_tokens_tensor, next_batch_labels_tensor, next_batch_sequence_length, new_epoch

    def get_next_batch_indexed_tokens(self, start_range, end_range, batch_size):
        indexed_tokens_list = []
        for index in range(start_range, end_range):
            indexed_tokens_list.append(self._toxic_comments[index].indexed_tokens)
        indexed_tokens_lengths = [len(indexed_tokens) for indexed_tokens in indexed_tokens_list]
        max_length = max(indexed_tokens_lengths)
        next_batch_indexed_tokens_tensor = np.zeros([batch_size, max_length], dtype=np.int32)
        for i, padded_indexed_tokens in enumerate(next_batch_indexed_tokens_tensor):
            padded_indexed_tokens[:len(indexed_tokens_list[i])] = indexed_tokens_list[i]
        return next_batch_indexed_tokens_tensor

    def get_next_batch_labels(self, start_range, end_range, batch_size):
        next_batch_labels_tensor = np.zeros([batch_size, 6])
        for i in range(start_range, end_range):
            next_batch_labels_tensor[i - start_range] = self._toxic_comments[i].labels
        return next_batch_labels_tensor

    def get_next_batch_sequence_length(self, start_range, end_range, batch_size):
        next_batch_sequence_length_tensor = np.zeros([batch_size])
        for i in range(start_range, end_range):
            next_batch_sequence_length_tensor[i - start_range] = self._toxic_comments[i].token_count
        return next_batch_sequence_length_tensor

    @property
    def toxic_comments(self):
        return self._toxic_comments

    @property
    def word_embeddings_model(self):
        return self._word_embeddings_model

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def current_batch(self):
        return self._current_batch


class ToxicCommentsRNN:
    def __init__(self, toxic_comments_train, toxic_comments_test, state_size, batch_size, epochs):
        self._toxic_comments_train = toxic_comments_train
        self._toxic_comments_test = toxic_comments_test
        self._state_size = state_size
        self._batch_size = batch_size
        self._epochs = epochs

    def build_graph(self):
        tf.reset_default_graph()
        vocabulary_size = self._toxic_comments_train.word_embeddings_model.vocabulary_size
        embedding_dimension = self._toxic_comments_train.word_embeddings_model.embedding_dimension
        embeddings_shape = [vocabulary_size, embedding_dimension]

        # Constants
        self._keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Placeholders
        self._x = tf.placeholder(tf.int32, [self._batch_size, 50])  # [batch_size, num_steps]
        self._sequence_length = tf.placeholder(tf.int32, [self._batch_size])
        self._y = tf.placeholder(tf.int32, [self._batch_size, 6])
        self._embeddings_placeholder = tf.placeholder(tf.float32, embeddings_shape)
        self._is_training = tf.placeholder(tf.bool)

        # Embeddings layer
        self._embeddings_variable = tf.Variable(tf.constant(0.0, shape=embeddings_shape), trainable=False, name="embeddings")
        self._embeddings_init = self._embeddings_variable.assign(self._embeddings_placeholder)

        # RNN Inputs
        self._rnn_inputs = tf.nn.embedding_lookup(self._embeddings_variable, self._x)

        # RNN
        self._forward_cell = tf.nn.rnn_cell.GRUCell(self._state_size)
        self._forward_cell = tf.contrib.rnn.DropoutWrapper(self._forward_cell, output_keep_prob=self._keep_prob)

        self._backward_cell = tf.nn.rnn_cell.GRUCell(self._state_size)
        self._backward_cell = tf.contrib.rnn.DropoutWrapper(self._backward_cell, output_keep_prob=self._keep_prob)

        # self._init_state_forward = tf.get_variable('init_state_forward', [1, self._state_size], initializer=tf.constant_initializer(0.0))
        # self._init_state_forward = tf.tile(self._init_state_forward, [self._batch_size, 1])
        #
        # self._init_state_backward = tf.get_variable('init_state_backward', [1, self._state_size], initializer=tf.constant_initializer(0.0))
        # self._init_state_backward = tf.tile(self._init_state_backward, [self._batch_size, 1])

        _init_state_forward = self._forward_cell.zero_state(self._batch_size, dtype=tf.float32)
        _init_state_backward = self._forward_cell.zero_state(self._batch_size, dtype=tf.float32)
        self._rnn_outputs, self._final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self._forward_cell, cell_bw=self._backward_cell, inputs=self._rnn_inputs, sequence_length=self._sequence_length, initial_state_fw=_init_state_forward, initial_state_bw=_init_state_backward)


        # self._rnn_outputs, self._final_states = tf.nn.dynamic_rnn(cell=self._forward_cell, inputs=self._rnn_inputs, sequence_length=self._sequence_length, initial_state=initial_state)


        self._rnn_output_forward = self._rnn_outputs[0]
        self._rnn_output_backward = self._rnn_outputs[1]

        # shape_forward = tf.shape(self._rnn_output_forward)
        # shape_backward = tf.shape(self._rnn_output_backward)


        # self._rnn_output_forward_max_pool = tf.reduce_max(self._rnn_output_forward, 2)

        # self._rnn_output_forward_max_pool = tf.reduce_max(self._rnn_output_forward, 2)
        # self._rnn_output_backward_max_pool = tf.reduce_max(self._rnn_output_backward, 2)
        # self._rnn_output_concatenated = tf.concat([self._rnn_output_forward, self._rnn_output_backward], 1)

        # self._rnn_output_forward_flatten = tf.reshape(self._rnn_output_forward, [shape_forward[0]*shape_forward[1], shape_forward[2]])
        # self._rnn_output_backward_flatten = tf.reshape(self._rnn_output_backward, [shape_backward[0] * shape_backward[1], shape_backward[2]])

        # self._rnn_output_concatenated = tf.concat([self._rnn_output_forward_flatten , self._rnn_output_backward_flatten], 0)
        # self._rnn_output_concatenated_max_pool = tf.reduce_max(self._rnn_output_concatenated, axis=[1])

        # self._myshape = tf.shape(self._rnn_output_concatenated_max_pool)
        # self._rnn_output_concatenated_max_pool = tf.reshape(self._rnn_output_concatenated_max_pool, [self._myshape[0], 1])

        # Add dropout, as the model otherwise quickly overfits

        # self._rnn_output_forward = tf.nn.dropout(self._rnn_output_forward, self._keep_prob)
        # self._rnn_output_backward = tf.nn.dropout(self._rnn_output_backward, self._keep_prob)

        # Get last RNN output

        batch_range = tf.range(self._batch_size)
        indices = tf.stack([batch_range, self._sequence_length - 1], axis=1)
        self._last_rnn_output_forward = tf.gather_nd(self._rnn_output_forward, indices)
        self._last_rnn_output_backward = tf.gather_nd(self._rnn_output_backward, indices)

        # Calculate logits and predictions
        W_forward = tf.get_variable('W_output_forward', [self._state_size, 6])
        W_backward = tf.get_variable('W_output_backward', [self._state_size, 6])
        b = tf.get_variable('b_output', [6], initializer=tf.constant_initializer(0.0))
        self._logits = tf.matmul(self._last_rnn_output_forward, W_forward) + tf.matmul(self._last_rnn_output_backward, W_backward) + b

        # self._flatten = tf.contrib.layers.flatten(self._rnn_output_concatenated)
        # self._fc1 = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(self._rnn_output_concatenated), num_outputs=100)
        # self._logits = tf.contrib.layers.fully_connected(inputs=self._fc1, num_outputs=6)
        # self._fc3 = tf.contrib.layers.fully_connected(inputs=self._fc2, num_outputs=50)
        # self._logits = tf.contrib.layers.fully_connected(inputs=self._fc3, num_outputs=6)
        # self._fc1_dropout = tf.contrib.layers.dropout(inputs=self._fc1, is_training=self._is_training)
        # self._fc5 = tf.contrib.layers.fully_connected(inputs=self._fc4, num_outputs=300)
        # self._fc6 = tf.contrib.layers.fully_connected(inputs=self._fc5, num_outputs=100)
        # self._fc3 = tf.contrib.layers.fully_connected(inputs=self._fc2, num_outputs=50)
        # self._fc4 = tf.contrib.layers.fully_connected(inputs=self._fc3, num_outputs=25)
        # self._fc3_dropout = tf.contrib.layers.dropout(inputs=self._fc3, is_training=self._is_training)

        self._preds = tf.nn.sigmoid(self._logits)

        # Calculate accuracy
        self._threshold_preds = self._preds > 0.5
        self._threshold_int_preds = tf.cast(self._threshold_preds, tf.int32)
        self._correct = tf.reduce_all(tf.equal(self._threshold_int_preds, self._y), 1)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))

        # Setup loss functions and optimizer
        self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=tf.cast(self._y, tf.float32)))
        self._optimizer = tf.train.AdamOptimizer(1e-2).minimize(self._loss)

    def train_graph(self):

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     feed_dict = {
        #         self._embeddings_placeholder: self._toxic_comments_train.word_embeddings_model.embeddings
        #     }
        #     sess.run(self._embeddings_init, feed_dict=feed_dict)
        #
        #     indexed_tokens, labels, sequence_length, new_epoch = self._toxic_comments_train.get_next_batch(self._batch_size)
        #     feed_dict = {
        #         self._x: indexed_tokens,
        #         self._sequence_length: sequence_length,
        #         self._y: labels,
        #         self._is_training: True
        #         # self._embeddings_placeholder: self._toxic_comments_train.word_embeddings_model.embeddings
        #     }
        #
        #     bla = sess.run(self._rnn_inputs, feed_dict=feed_dict)
        #     bla2 = sess.run(self._rnn_outputs, feed_dict=feed_dict)
        #     bla3 = sess.run(self._rnn_output_forward, feed_dict=feed_dict)
        #     bla4 = sess.run(self._rnn_output_backward, feed_dict=feed_dict)
        #
        #     # bla5 = sess.run(self._rnn_output_forward_flatten, feed_dict=feed_dict)
        #     #
        #     # bla6 = sess.run(self._rnn_output_concatenated, feed_dict=feed_dict)
        #     # bla7 = sess.run(self._rnn_output_concatenated_max_pool, feed_dict=feed_dict)
        #
        #
        #     # bla8 = sess.run(self._fc1, feed_dict=feed_dict)
        #
        #     bla9 = sess.run(self._logits, feed_dict=feed_dict)
        #
        #     y = 6

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step, accuracy = 0, 0
            train_accuracy, test_accuracy, threshold_preds = [], [], []
            while self._toxic_comments_train.current_epoch < self._epochs:
                print("Epoch:           ", self._toxic_comments_train.current_epoch)
                print("Batch:           ", self._toxic_comments_train.current_batch)

                feed_dict = {
                    self._embeddings_placeholder: self._toxic_comments_train.word_embeddings_model.embeddings
                }
                sess.run(self._embeddings_init, feed_dict=feed_dict)


                indexed_tokens, labels, sequence_length, new_epoch_train = self._toxic_comments_train.get_next_batch(self._batch_size)
                feed_dict = {
                    self._x: indexed_tokens,
                    self._sequence_length: sequence_length,
                    self._y: labels,
                    self._embeddings_placeholder: self._toxic_comments_train.word_embeddings_model.embeddings,
                    self._keep_prob: 0.5,
                    self._is_training: True
                }
                loss, logits, x, inputs, seq = sess.run([self._loss, self._logits, self._x, self._rnn_inputs, self._sequence_length], feed_dict=feed_dict)
                # print("------------------ threshold_int_preds: ------------------")
                # print(threshold_int_preds)
                # print("------------------ y: ------------------")
                # print(y)
                # print("------------- correct: -------------")
                # print(seq)
                current_accuracy, loss, _ = sess.run([self._accuracy, self._loss, self._optimizer], feed_dict=feed_dict)
                print(loss)
                accuracy += current_accuracy
                step += 1
                print("Accuracy:        ", "{0:.2%}".format(current_accuracy))
                print("Avg. Accuracy:   ", "{0:.2%}".format(accuracy / step))
                print("---------------------------------------------")

                if new_epoch_train:
                    print("Epoch", self._toxic_comments_train.current_epoch, "done.")
                    train_accuracy.append(accuracy / step)
                    step, accuracy = 0, 0

                    while True:
                        feed_dict = {
                            self._embeddings_placeholder: self._toxic_comments_test.word_embeddings_model.embeddings
                        }
                        sess.run(self._embeddings_init, feed_dict=feed_dict)

                        indexed_tokens, labels, sequence_length, new_epoch_test = self._toxic_comments_test.get_next_batch(self._batch_size)
                        feed_dict = {
                            self._x: indexed_tokens,
                            self._sequence_length: sequence_length,
                            self._y: labels,
                            self._embeddings_placeholder: self._toxic_comments_test.word_embeddings_model.embeddings,
                            self._is_training: False
                        }
                        current_threshold_preds = sess.run(self._threshold_preds, feed_dict=feed_dict)
                        threshold_preds.append(current_threshold_preds)
                        current_accuracy = sess.run(self._accuracy, feed_dict=feed_dict)
                        accuracy += current_accuracy
                        step += 1

                        if new_epoch_test:
                            break

                    test_accuracy.append(accuracy / step)
                    step, accuracy = 0, 0
                    print("Test accuracy:", "{0:.2%}".format(test_accuracy[-1]))
                    print("---------------------------------------------")
