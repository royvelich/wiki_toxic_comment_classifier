import numpy as np
import random


class TokenEmbedding:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class GloveModel:
    def __init__(self):
        self.reset_model()

    def reset_model(self):
        self.tokens = []
        self.embeddings = []
        self.token_to_embedding = {}

    def append_model(self, token, embedding):
        self.tokens.append(token)
        self.token_to_embedding[token] = TokenEmbedding(embedding, len(self.embeddings))
        self.embeddings.append(embedding)

    def load_glove_model(self, glove_model_file_path):
        self.reset_model()
        print("Loading Glove Model")
        file = open(glove_model_file_path, encoding="utf8")
        _, embedding = self.parse_line(self.peek_line(file))
        self.append_model("<UNK>", np.zeros(embedding.size, dtype=float))
        for line in file:
            token, embedding = self.parse_line(line)
            self.append_model(token, embedding)
        print("Done.", len(self.token_to_embedding), " token embeddings loaded!")

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


class SimpleDataIterator:
    def __init__(self, train_data):
        self.train_data_embeddings = [train_data[i].embeddings for i in train_data]
        self.train_data_labels = [train_data[i].labels for i in train_data]
        self.size = len(self.train_data)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.train_data)
        self.cursor = 0

    def next_batch(self, batch_size):
        if self.cursor + batch_size - 1 > self.size:
            self.epochs += 1
            self.shuffle()
        batch_data_embeddings = self.train_data_embeddings[self.cursor: self.cursor + batch_size - 1]
        batch_data_labels = self.train_data_embeddings[self.cursor: self.cursor + batch_size - 1]
        self.cursor += batch_size
        return batch_data_embeddings, batch_data_labels


class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor + n - 1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['gender'] * 3 + res['age_bracket'], res['length']