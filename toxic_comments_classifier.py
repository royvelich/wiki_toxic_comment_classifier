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
            file = open(glove_model_file_path, encoding="utf8")
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


class ToxicComments:
    def __init__(self, glove_model):
        self._glove_model = glove_model
        self.initialize()

    def initialize(self):
        self.size = 0
        self.epochs = 0
        self._toxic_comments_train_data = []

    def load_toxic_comments(self, train_data_file_path, test_data_file_path, test_labels_file_path):
        self.initialize()
        self.load_toxic_comments_train_data(train_data_file_path)
        self.size = len(self._toxic_comments_train_data)
        self.shuffle()

    def load_toxic_comments_train_data(self, train_data_file_path):
        if os.path.exists('.\\toxic_comments_train_data.pickle'):
            with open('.\\toxic_comments_train_data.pickle', 'rb') as handle:
                self._toxic_comments_train_data = pickle.load(handle)
        else:
            with open(train_data_file_path, encoding="utf8") as csv_file:
                dict_reader = csv.DictReader(csv_file)
                for csv_row in dict_reader:
                    toxic_comment = ToxicComment(csv_row, self._glove_model)
                    self._toxic_comments_train_data.append(toxic_comment)

            with open('.\\toxic_comments_train_data.pickle', 'wb') as handle:
                pickle.dump(self._toxic_comments_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return self._toxic_comments_train_data

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
        self.cursor += batch_size
        return next_batch_indexed_tokens_tensor, next_batch_labels_tensor

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

    @property
    def toxic_comments_train_data(self):
        return self._toxic_comments_train_data

class ToxicCommentsRNN:
    # def reset_graph():
    #     if 'sess' in globals() and sess:
    #         sess.close()
    #     tf.reset_default_graph()

    def build_rnn(vocab_size, state_size = 64, batch_size = 256, num_classes = 6):
        # reset_graph()

        # Placeholders
        x = tf.placeholder(tf.int32, [batch_size, None])  # [batch_size, num_steps]
        seqlen = tf.placeholder(tf.int32, [batch_size])
        y = tf.placeholder(tf.int32, [batch_size])
        keep_prob = tf.constant(1.0)

        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

        # RNN
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        init_state = tf.get_variable('init_state', [1, state_size],
                                     initializer=tf.constant_initializer(0.0))
        init_state = tf.tile(init_state, [batch_size, 1])
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                                     initial_state=init_state)

        # Add dropout, as the model otherwise quickly overfits
        rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

        """
        Obtain the last relevant output. The best approach in the future will be to use:

            last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

        which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
        gradient for this op has not been implemented as of this writing.

        The below solution works, but throws a UserWarning re: the gradient.
        """
        idx = tf.range(batch_size) * tf.shape(rnn_outputs)[1] + (seqlen - 1)
        last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

        # Softmax layer
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(last_rnn_output, W) + b
        preds = tf.nn.softmax(logits)
        correct = tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

        return {
            'x': x,
            'seqlen': seqlen,
            'y': y,
            'dropout': keep_prob,
            'loss': loss,
            'ts': train_step,
            'preds': preds,
            'accuracy': accuracy
        }