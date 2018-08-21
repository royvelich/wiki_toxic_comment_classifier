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
    def __init__(self, csv_row, glove_model):
        self._tokens = word_tokenize(csv_row['comment_text'].strip('"'))
        self._labels = np.array([float(csv_row['toxic']), float(csv_row['severe_toxic']), float(csv_row['obscene']), float(csv_row['threat']), float(csv_row['insult']), float(csv_row['identity_hate'])])
        self._indexed_tokens = np.zeros(shape=[len(self._tokens)], dtype=np.int32)
        for i, token in enumerate(self._tokens):
            token = token.lower()
            index = glove_model.token_to_embedding['something'].index
            if token in glove_model.token_to_embedding:
                index = glove_model.token_to_embedding[token].index
            self._indexed_tokens[i] = index

    @property
    def tokens(self):
        return self._tokens

    @property
    def labels(self):
        return self._labels

    @property
    def indexed_tokens(self):
        return self._indexed_tokens


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
            file = open(glove_model_file_path, encoding="ISO-8859-1")
            _, embedding = self.parse_line(self.peek_line(file))
            self.append_model("<UNK>", np.zeros(embedding.size, dtype=float))
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
    def __init__(self, word_embeddings_model):
        self._word_embeddings_model = word_embeddings_model
        self.initialize()

    def initialize(self):
        self.size = 0
        self.epochs = 0
        self._toxic_comments_train_data = []
        self._toxic_comments_test_data = []

    def load_toxic_comments(self, train_data_file_path, test_data_file_path):
        self.initialize()
        self.load_toxic_comments_train_data(train_data_file_path)
        self.load_toxic_comments_test_data(test_data_file_path)
        self.size = len(self._toxic_comments_train_data)
        self.shuffle()

    def load_toxic_comments_train_data(self, train_data_file_path):
        if os.path.exists('.\\toxic_comments_train_data.pickle'):
            with open('.\\toxic_comments_train_data.pickle', 'rb') as handle:
                self._toxic_comments_train_data = pickle.load(handle)
        else:
            with open(train_data_file_path, encoding="ISO-8859-1") as csv_file:
                dict_reader = csv.DictReader(csv_file)
                for csv_row in dict_reader:
                    toxic_comment = ToxicComment(csv_row, self._word_embeddings_model)
                    self._toxic_comments_train_data.append(toxic_comment)

            with open('.\\toxic_comments_train_data.pickle', 'wb') as handle:
                pickle.dump(self._toxic_comments_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self._toxic_comments_train_data

    def load_toxic_comments_test_data(self, test_data_file_path):
        if os.path.exists('.\\toxic_comments_test_data.pickle'):
            with open('.\\toxic_comments_test_data.pickle', 'rb') as handle:
                self._toxic_comments_test_data = pickle.load(handle)
        else:
            with open(test_data_file_path, encoding="ISO-8859-1") as csv_file:
                dict_reader = csv.DictReader(csv_file)
                for csv_row in dict_reader:
                    if csv_row['toxic'] != '-1':
                        toxic_comment = ToxicComment(csv_row, self._word_embeddings_model)
                        self._toxic_comments_test_data.append(toxic_comment)

            with open('.\\toxic_comments_test_data.pickle', 'wb') as handle:
                pickle.dump(self._toxic_comments_test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self._toxic_comments_test_data

    def shuffle(self):
        random.shuffle(self._toxic_comments_train_data)
        self.cursor = 0

    def get_next_batch(self, batch_size):
        if self.cursor + batch_size - 1 > self.size:
            self.epochs += 1
            self.shuffle()
        start_range = self.cursor
        end_range = self.cursor + batch_size
        next_batch_indexed_tokens_tensor = self.get_next_batch_indexed_tokens(start_range, end_range, batch_size)
        next_batch_labels_tensor = self.get_next_batch_labels(start_range, end_range, batch_size)
        next_batch_sequence_length = self.get_next_batch_sequence_length(start_range, end_range, batch_size)
        self.cursor += batch_size
        return next_batch_indexed_tokens_tensor, next_batch_labels_tensor, next_batch_sequence_length

    def get_next_batch_indexed_tokens(self, start_range, end_range, batch_size):
        indexed_tokens_list = []
        for index in range(start_range, end_range):
            indexed_tokens_list.append(self._toxic_comments_train_data[index].indexed_tokens)
        indexed_tokens_lengths = [len(indexed_tokens) for indexed_tokens in indexed_tokens_list]
        max_length = max(indexed_tokens_lengths)
        next_batch_indexed_tokens_tensor = np.zeros([batch_size, max_length], dtype=np.int32)
        for i, padded_indexed_tokens in enumerate(next_batch_indexed_tokens_tensor):
            padded_indexed_tokens[:len(indexed_tokens_list[i])] = indexed_tokens_list[i]
        return next_batch_indexed_tokens_tensor

    def get_next_batch_labels(self, start_range, end_range, batch_size):
        next_batch_labels_tensor = np.zeros([batch_size, 6])
        for i in range(start_range, end_range):
            next_batch_labels_tensor[i] = self._toxic_comments_train_data[i].labels
        return next_batch_labels_tensor

    def get_next_batch_sequence_length(self, start_range, end_range, batch_size):
        next_batch_sequence_length_tensor = np.zeros([batch_size])
        for i in range(start_range, end_range):
            next_batch_sequence_length_tensor[i] = len(self._toxic_comments_train_data[i].indexed_tokens)
        return next_batch_sequence_length_tensor

    @property
    def toxic_comments_train_data(self):
        return self._toxic_comments_train_data

    @property
    def toxic_comments_test_data(self):
        return self._toxic_comments_test_data

    @property
    def word_embeddings_model(self):
        return self._word_embeddings_model


class ToxicCommentsRNN:
    def __init__(self, toxic_comments, state_size, batch_size):
        self._toxic_comments = toxic_comments
        self._state_size = state_size
        self._batch_size = batch_size

    def build_graph(self):
        tf.reset_default_graph()
        vocabulary_size = self._toxic_comments.word_embeddings_model.vocabulary_size
        embedding_dimension = self._toxic_comments.word_embeddings_model.embedding_dimension
        embeddings_shape = [vocabulary_size, embedding_dimension]

        # Constants
        self._keep_prob = tf.constant(1.0)

        # Placeholders
        self._x = tf.placeholder(tf.int32, [self._batch_size, None])  # [batch_size, num_steps]
        self._sequence_length = tf.placeholder(tf.int32, [self._batch_size])
        self._y = tf.placeholder(tf.int32, [self._batch_size, None])
        self._embeddings_placeholder = tf.placeholder(tf.float32, embeddings_shape)

        # Embeddings layer
        self._embeddings_variable = tf.Variable(tf.constant(0.0, shape=embeddings_shape), trainable=False, name="embeddings")
        self._embeddings_init = self._embeddings_variable.assign(self._embeddings_placeholder)

        # RNN Inputs
        self._rnn_inputs = tf.nn.embedding_lookup(self._embeddings_variable, self._x)

        # RNN
        self._cell = tf.nn.rnn_cell.GRUCell(self._state_size)
        self._init_state = tf.get_variable('init_state', [1, self._state_size], initializer=tf.constant_initializer(0.0))
        self._init_state = tf.tile(self._init_state, [self._batch_size, 1])
        self._rnn_outputs, self._final_state = tf.nn.dynamic_rnn(self._cell, self._rnn_inputs, sequence_length=self._sequence_length, initial_state=self._init_state)

        # Get last RNN output
        batch_range = tf.range(self._batch_size)
        indices = tf.stack([batch_range, self._sequence_length - 1], axis=1)
        self._last_rnn_output = tf.gather_nd(self._rnn_outputs, indices)


        W = tf.get_variable('W_output', [self._state_size, 6])
        b = tf.get_variable('b_output', [6], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(self._last_rnn_output, W) + b
        preds = tf.nn.softmax(logits)

        correct = tf.equal(tf.cast(preds > 0.5, tf.int32), self._y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self._y))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def train_graph(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            # indexed_tokens, labels, sequence_length = self._toxic_comments.get_next_batch(64)
            # feed_dict = {
            #     self._x: indexed_tokens,
            #     self._sequence_length: sequence_length,
            #     self._y: labels,
            #     self._embeddings_placeholder: self._toxic_comments.word_embeddings_model.embeddings
            # }
            #
            # sess.run(self._embeddings_init, feed_dict=feed_dict)
            # bla = sess.run(self._rnn_inputs, feed_dict=feed_dict)
            # bla2 = sess.run(self._rnn_outputs, feed_dict=feed_dict)
            # bla3 = sess.run(self._last_rnn_output, feed_dict=feed_dict)

            step, accuracy = 0, 0
            tr_losses, te_losses = [], []
            current_epoch = 0
            while current_epoch < num_epochs:
                step += 1
                batch = tr.next_batch(batch_size)
                feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
                accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
                accuracy += accuracy_

                if tr.epochs > current_epoch:
                    current_epoch += 1
                    tr_losses.append(accuracy / step)
                    step, accuracy = 0, 0

                    # eval test set
                    te_epoch = te.epochs
                    while te.epochs == te_epoch:
                        step += 1
                        batch = te.next_batch(batch_size)
                        feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                        accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                        accuracy += accuracy_

                    te_losses.append(accuracy / step)
                    step, accuracy = 0, 0
                    print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])



