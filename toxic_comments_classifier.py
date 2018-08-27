import random
import csv
import nltk
import numpy as np
import pickle
import os
import tensorflow as tf
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import os
from shutil import copyfile
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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
    _eng_stopwords = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _tokenizer = TweetTokenizer()
    _appos = {"aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not",
             "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
             "he'd": "he would", "he'll": "he will", "he's": "he is", "i'd": "I would", "i'd": "I had",
             "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is", "it'll": "it will", "i've": "I have",
             "let's": "let us", "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
             "she'd": "she would", "she'll": "she will", "she's": "she is", "shouldn't": "should not",
             "that's": "that is", "there's": "there is", "they'd": "they would", "they'll": "they will",
             "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are",
             "weren't": "were not", "we've": "we have", "what'll": "what will", "what're": "what are",
             "what's": "what is", "what've": "what have", "where's": "where is", "who'd": "who would",
             "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not",
             "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are",
             "you've": "you have", "'re": " are", "wasn't": "was not", "we'll": " will", "didn't": "did not"}

    @staticmethod
    def _clean(comment):
        # make all characters lower cased
        comment = comment.lower()
        # remove new line character
        comment = re.sub('\\n', ' ', comment)
        # remove ip addresses
        comment = re.sub('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', comment)
        # remove usernames
        comment = re.sub('\[\[.*\]', '', comment)
        # split the comment into words
        words = ToxicComment._tokenizer.tokenize(comment)
        # replace that's to that is by looking up the dictionary
        words = [ToxicComment._appos[word] if word in ToxicComment._appos else word for word in words]
        # replace variation of a word with its base form
        words = [ToxicComment._lemmatizer.lemmatize(word, "v") for word in words]
        # eliminate stop words
        words = [w for w in words if not w in ToxicComment._eng_stopwords]
        # now we will have only one string containing all the words
        clean_comment = " ".join(words)
        # remove all non alphabetical characters
        clean_comment = re.sub("\W+", " ", clean_comment)
        clean_comment = re.sub("  ", " ", clean_comment)
        return clean_comment

    def __init__(self, csv_row, glove_model, comment_max_length):
        self._id = csv_row['id']
        self._comment_text = csv_row['comment_text']
        self._tokens = word_tokenize(ToxicComment._clean(csv_row['comment_text']))
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

    @property
    def id(self):
        return self._id

    @property
    def comment_text(self):
        return self._comment_text


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
        if self._id is 'train':
            random.shuffle(self._toxic_comments)
        self.cursor = 0

    def get_next_batch(self, batch_size):
        start_range = self.cursor
        end_range = self.cursor + batch_size
        next_batch = self.get_next_batch_comments(start_range, end_range, batch_size)
        next_batch_indexed_tokens = self.get_next_batch_indexed_tokens(start_range, end_range, batch_size)
        next_batch_labels = self.get_next_batch_labels(start_range, end_range, batch_size)
        self.cursor += batch_size

        self._current_batch = self._current_batch + 1
        new_epoch = False
        if self.cursor + batch_size > self._epoch_size:
            self._current_epoch += 1
            self._current_batch = 0
            self.shuffle()
            new_epoch = True

        return next_batch, next_batch_indexed_tokens, next_batch_labels, new_epoch

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

    def get_next_batch_comments(self, start_range, end_range, batch_size):
        next_batch = []
        for i in range(start_range, end_range):
            next_batch.append(self._toxic_comments[i])
        return next_batch

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

    @property
    def id(self):
        return self._id


class ToxicCommentsRNN:
    def __init__(
            self,
            glove_model_file_path,
            toxic_comments_train_file_path,
            toxic_comments_test_file_path,
            toxic_comment_max_length,
            state_size,
            batch_size,
            epochs,
            learning_rate,
            fc_layer1_size,
            fc_layer2_size,
            fc_layer1_dropout,
            fc_layer2_dropout,
            rnn_dropout):
        self._results_root_directory = ".\\Results"
        if not os.path.exists(self._results_root_directory):
            os.makedirs(self._results_root_directory)
        immidiate_subdirectories = ToxicCommentsRNN._get_immediate_subdirectories(self._results_root_directory)
        if not immidiate_subdirectories:
            self._current_results_id = 1
        else:
            self._current_results_id = max(map(int, immidiate_subdirectories)) + 1
        last_directory_name = str(self._current_results_id)
        self._results_directory = os.path.join(self._results_root_directory, last_directory_name)
        self._glove_model = GloveModel()
        self._glove_model.load_glove_model(glove_model_file_path)
        self._toxic_comments_train = ToxicComments(self._glove_model, 'train', toxic_comment_max_length)
        self._toxic_comments_train.load_toxic_comments(toxic_comments_train_file_path)
        self._toxic_comments_test = ToxicComments(self._glove_model, 'test', toxic_comment_max_length)
        self._toxic_comments_test.load_toxic_comments(toxic_comments_test_file_path)
        self._toxic_comment_max_length = toxic_comment_max_length
        self._state_size = state_size
        self._batch_size = batch_size
        self._epochs = epochs
        self._train_stats = []
        self._test_stats = []
        self._test_results = []
        self._best_test_results = []
        self._max_auc = 0
        self._toxic_correctness_sum = 0
        self._learning_rate = learning_rate
        self._fc_layer1_size = fc_layer1_size
        self._fc_layer2_size = fc_layer2_size
        self._fc_layer1_dropout = fc_layer1_dropout
        self._fc_layer2_dropout = fc_layer2_dropout
        self._rnn_dropout = rnn_dropout

    def _reset_global_variables(self):
        self._sess.run(tf.global_variables_initializer())

    def _reset_local_variables(self):
        self._sess.run(tf.local_variables_initializer())
        feed_dict = {
            self._embeddings_placeholder: self._glove_model.embeddings
        }
        self._sess.run(self._embeddings_init, feed_dict=feed_dict)

    def _optimize(self, feed_dict):
        self._sess.run([self._optimizer], feed_dict=feed_dict)

    def _run_update_ops(self, feed_dict):
        self._sess.run([self._update_op_loss_mean, self._update_op_accuracy, self._update_op_column_0, self._update_op_column_1, self._update_op_column_2, self._update_op_column_3, self._update_op_column_4, self._update_op_column_5], feed_dict=feed_dict)

    def _get_batch_results(self, feed_dict):
         return self._sess.run([self._accuracy, self._loss_mean, self._threshold_int_preds, self._y, self._auc], feed_dict=feed_dict)

    @staticmethod
    def _get_batch_toxic_correctness(preds, y):
        zero_indices = np.where(~y.any(axis=1))[0]
        y_non_zero = np.delete(y, zero_indices, axis=0)
        preds_non_zero = np.delete(preds, zero_indices, axis=0)
        correct_non_zero_pred = np.equal(y_non_zero, preds_non_zero).all(axis=1)
        correct_non_zero_preds_count = correct_non_zero_pred.shape[0]
        return np.sum(correct_non_zero_pred) / correct_non_zero_preds_count

    def _get_next_batch(self, toxic_comments):
        toxic_comments_batch, indexed_tokens, labels, new_epoch = toxic_comments.get_next_batch(self._batch_size)
        feed_dict = {
            self._x: indexed_tokens,
            self._y: labels,
            self._keep_prob: self._rnn_dropout if toxic_comments.id is 'train' else 1.0,
            self._is_training: True if toxic_comments.id is 'train' else False
        }
        return feed_dict, new_epoch, toxic_comments_batch

    @staticmethod
    def _get_immediate_subdirectories(directory_path):
        return [name for name in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, name))]

    def _run_batch(self, toxic_comments):
        print("{:<30}".format(" ".join([toxic_comments.id.capitalize(), "Epoch:"])), toxic_comments.current_epoch)
        print("{:<30}".format("Batch:"), toxic_comments.current_batch)
        is_training = True if toxic_comments.id is 'train' else False
        batch_count = toxic_comments.current_batch + 1
        feed_dict, new_epoch, toxic_comments_batch = self._get_next_batch(toxic_comments=toxic_comments)
        if toxic_comments.id is 'train':
            self._optimize(feed_dict=feed_dict)
        self._run_update_ops(feed_dict=feed_dict)
        accuracy, loss, preds, y, auc = self._get_batch_results(feed_dict=feed_dict)
        toxic_correctness = ToxicCommentsRNN._get_batch_toxic_correctness(preds=preds, y=y)
        self._toxic_correctness_sum = self._toxic_correctness_sum + toxic_correctness
        toxic_correctness = self._toxic_correctness_sum / batch_count
        print("{:<30}".format("Accuracy:"), "{0:.2%}".format(accuracy))
        print("{:<30}".format("Toxic Correctness:"), "{0:.4%}".format(toxic_correctness))
        print("{:<30}".format("Loss:"), loss)
        print("{:<30}".format("Mean Column-Wise ROC AUC:"), auc)
        print("---------------------------------------------")

        if not is_training:
            for i, toxic_comment in enumerate(toxic_comments_batch):
                self._test_results.append([toxic_comment.id] + [toxic_comment.comment_text] + y[i].tolist() + preds[i].tolist())

        if new_epoch:
            _new_stats = [toxic_comments.current_epoch - 1, accuracy, loss, auc, toxic_correctness]
            if is_training:
                self._train_stats.append(_new_stats)
            else:
                self._test_stats.append(_new_stats)
                if auc > self._max_auc:
                    saver = tf.train.Saver()
                    saver.save(self._sess, os.path.join(self._results_directory, "model.ckpt"))
                    self._best_test_results = self._test_results
                    self._max_auc = auc

                    should_rewrite_best_result = True
                    best_results_file_path = os.path.join(self._results_root_directory, "best_results.csv")
                    if os.path.exists(best_results_file_path):
                        with open(best_results_file_path, encoding="ISO-8859-1") as csv_file:
                            dict_reader = csv.DictReader(csv_file)
                            for csv_row in dict_reader:
                                last_best_auc = float(csv_row["auc"])
                                if auc < last_best_auc:
                                    should_rewrite_best_result = False

                    if should_rewrite_best_result:
                        with open(best_results_file_path, 'w', encoding="ISO-8859-1") as myfile:
                            writer = csv.writer(myfile, lineterminator='\n')
                            writer.writerow(["id", "auc"])
                            writer.writerow([self._current_results_id, auc])

                self._test_results = []

            self._toxic_correctness_sum = 0

        return new_epoch

    def build_graph(self):
        tf.reset_default_graph()
        vocabulary_size = self._glove_model.vocabulary_size
        embedding_dimension = self._glove_model.embedding_dimension
        embeddings_shape = [vocabulary_size, embedding_dimension]

        # Constants
        self._keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Placeholders
        self._x = tf.placeholder(tf.int32, [self._batch_size, self._toxic_comment_max_length])
        self._sequence_length = tf.placeholder(tf.int32, [self._batch_size])
        self._y = tf.placeholder(tf.int32, [self._batch_size, 6])
        self._embeddings_placeholder = tf.placeholder(tf.float32, embeddings_shape)
        self._is_training = tf.placeholder(tf.bool)

        # Embeddings layer
        self._embeddings_variable = tf.Variable(tf.constant(0.0, shape=embeddings_shape), trainable=False, name="embeddings")
        self._embeddings_init = self._embeddings_variable.assign(self._embeddings_placeholder)

        # RNN Inputs
        self._rnn_inputs = tf.nn.embedding_lookup(self._embeddings_variable, self._x)

        # RNN forward cell
        self._forward_cell = tf.nn.rnn_cell.GRUCell(self._state_size)
        self._forward_cell = tf.contrib.rnn.DropoutWrapper(self._forward_cell, output_keep_prob=self._keep_prob)

        # RNN backward cell
        self._backward_cell = tf.nn.rnn_cell.GRUCell(self._state_size)
        self._backward_cell = tf.contrib.rnn.DropoutWrapper(self._backward_cell, output_keep_prob=self._keep_prob)

        # RNN cells initial state
        _init_state_forward = self._forward_cell.zero_state(self._batch_size, dtype=tf.float32)
        _init_state_backward = self._backward_cell.zero_state(self._batch_size, dtype=tf.float32)

        # Calculate sequence length for our inputs (text comments have variable length)
        used = tf.sign(tf.reduce_max(tf.abs(self._rnn_inputs), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        self._sequence_length = tf.cast(length, tf.int32)

        # Build bi-directional RNN and extract forward and backward outputs
        self._rnn_outputs, self._final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self._forward_cell, cell_bw=self._backward_cell, inputs=self._rnn_inputs, sequence_length=self._sequence_length, initial_state_fw=_init_state_forward, initial_state_bw=_init_state_backward)
        self._rnn_output_forward = self._rnn_outputs[0]
        self._rnn_output_backward = self._rnn_outputs[1]

        # self._rnn_output_forward_max_pool = tf.reduce_max(self._rnn_output_forward, 2)
        # self._rnn_output_backward_max_pool = tf.reduce_max(self._rnn_output_backward, 2)
        # self._rnn_output_concatenated = tf.concat([self._rnn_output_forward_max_pool, self._rnn_output_backward_max_pool], 1)

        # Get last RNN output
        # batch_range = tf.range(self._batch_size)
        # indices = tf.stack([batch_range, self._seq_len - 1], axis=1)
        # self._last_rnn_output_forward = tf.gather_nd(self._rnn_output_forward, indices)
        # self._last_rnn_output_backward = tf.gather_nd(self._rnn_output_backward, indices)

        shape = [self._rnn_output_forward.shape[0], self._rnn_output_forward.shape[1] * self._rnn_output_forward.shape[2]]
        self._rnn_output_concatenated = tf.concat([tf.reshape(self._rnn_output_forward, shape), tf.reshape(self._rnn_output_backward, shape)], axis=1)
        self._fc1 = tf.contrib.layers.fully_connected(inputs=self._rnn_output_concatenated, num_outputs=self._fc_layer1_size)
        self._fc1_dropout = tf.contrib.layers.dropout(inputs=self._fc1, is_training=self._is_training, keep_prob=self._fc_layer1_dropout)
        self._fc2 = tf.contrib.layers.fully_connected(inputs=self._fc1_dropout, num_outputs=self._fc_layer2_size)
        self._fc2_dropout = tf.contrib.layers.dropout(inputs=self._fc2, is_training=self._is_training, keep_prob=self._fc_layer2_dropout)
        self._logits = tf.contrib.layers.fully_connected(inputs=self._fc2_dropout, num_outputs=6, activation_fn=None)

        # Apply sigmoid on logits to get predictions
        self._preds = tf.nn.sigmoid(self._logits)

        # Split preds and labels (y) by columns
        self._preds_column_0 = tf.slice(self._preds, [0, 0], [self._batch_size, 1])
        self._preds_column_1 = tf.slice(self._preds, [0, 1], [self._batch_size, 1])
        self._preds_column_2 = tf.slice(self._preds, [0, 2], [self._batch_size, 1])
        self._preds_column_3 = tf.slice(self._preds, [0, 3], [self._batch_size, 1])
        self._preds_column_4 = tf.slice(self._preds, [0, 4], [self._batch_size, 1])
        self._preds_column_5 = tf.slice(self._preds, [0, 5], [self._batch_size, 1])

        self._y_column_0 = tf.slice(self._y, [0, 0], [self._batch_size, 1])
        self._y_column_1 = tf.slice(self._y, [0, 1], [self._batch_size, 1])
        self._y_column_2 = tf.slice(self._y, [0, 2], [self._batch_size, 1])
        self._y_column_3 = tf.slice(self._y, [0, 3], [self._batch_size, 1])
        self._y_column_4 = tf.slice(self._y, [0, 4], [self._batch_size, 1])
        self._y_column_5 = tf.slice(self._y, [0, 5], [self._batch_size, 1])

        # Calculate accuracy and mean column-wise AUC
        self._threshold_preds = self._preds > 0.5
        self._threshold_int_preds = tf.cast(self._threshold_preds, tf.int32)
        self._correct = tf.reduce_all(tf.equal(self._threshold_int_preds, self._y), 1)
        self._accuracy, self._update_op_accuracy = tf.metrics.mean(tf.cast(self._correct, tf.float32))
        self._auc_column_0, self._update_op_column_0 = tf.metrics.auc(labels=self._y_column_0, predictions=self._preds_column_0)
        self._auc_column_1, self._update_op_column_1 = tf.metrics.auc(labels=self._y_column_1, predictions=self._preds_column_1)
        self._auc_column_2, self._update_op_column_2 = tf.metrics.auc(labels=self._y_column_2, predictions=self._preds_column_2)
        self._auc_column_3, self._update_op_column_3 = tf.metrics.auc(labels=self._y_column_3, predictions=self._preds_column_3)
        self._auc_column_4, self._update_op_column_4 = tf.metrics.auc(labels=self._y_column_4, predictions=self._preds_column_4)
        self._auc_column_5, self._update_op_column_5 = tf.metrics.auc(labels=self._y_column_5, predictions=self._preds_column_5)
        self._auc_concat = tf.stack([self._auc_column_0, self._auc_column_1, self._auc_column_2, self._auc_column_3, self._auc_column_4, self._auc_column_5], axis=0);
        self._auc = tf.reduce_mean(self._auc_concat)

        # Setup loss functions and optimizer
        self._sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=tf.cast(self._y, tf.float32))
        self._loss_mean, self._update_op_loss_mean = tf.metrics.mean(self._sigmoid_cross_entropy)
        self._loss = tf.reduce_mean(self._sigmoid_cross_entropy)
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

    def train_graph(self):
        with tf.Session() as self._sess:
            self._reset_global_variables()
            self._reset_local_variables()
            while self._toxic_comments_train.current_epoch < self._epochs:
                new_epoch = self._run_batch(toxic_comments=self._toxic_comments_train)
                if new_epoch:
                    self._reset_local_variables()
                    while True:
                        new_epoch = self._run_batch(toxic_comments=self._toxic_comments_test)
                        if new_epoch:
                            break
                    self._reset_local_variables()

        header = "epoch, accuracy, loss, auc, toxic_correctness"
        np.savetxt(os.path.join(self._results_directory, "train_stats.csv"), np.array(self._train_stats), delimiter=",", header=header, comments='')
        np.savetxt(os.path.join(self._results_directory, "test_stats.csv"), np.array(self._test_stats), delimiter=",", header=header, comments='')

        with open(os.path.join(self._results_directory, "best_test_results.csv"), 'w', encoding="ISO-8859-1") as myfile:
            writer = csv.writer(myfile, lineterminator='\n')
            writer.writerow(["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "toxic_preds", "severe_toxic_preds", "obscene_preds", "threat_preds", "insult_preds", "identity_hate_preds"])
            writer.writerows(self._best_test_results)

        copyfile(os.path.join(".", "main.py"), os.path.join(self._results_directory, "main.py"))
        copyfile(os.path.join(".", "toxic_comments_classifier.py"), os.path.join(self._results_directory, "toxic_comments_classifier.py"))

    def test_graph(self, model_file_path):
        with tf.Session() as self._sess:
            self._reset_global_variables()
            self._reset_local_variables()
            saver = tf.train.Saver()
            saver.restore(self._sess, model_file_path)
            while True:
                new_epoch = self._run_batch(toxic_comments=self._toxic_comments_test)
                if new_epoch:
                    break