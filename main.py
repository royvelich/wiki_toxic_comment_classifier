import csv
import nltk
import numpy as np
import pickle
import os
import toxic_comments_classifier
from nltk.tokenize import word_tokenize

nltk.download('punkt')





def load_train_data(train_data_file_path):
    print("Loading Train Data (Toxic Comments)")
    with open(train_data_file_path, encoding="utf8") as csvFile:
        dict_reader = csv.DictReader(csvFile)
        train_data = []
        for row in dict_reader:
            comment = {
                'tokens': word_tokenize(row['comment_text'].strip('"')),
                'labels': np.array([float(row['toxic']), float(row['severe_toxic']), float(row['obscene']), float(row['threat']), float(row['insult']), float(row['identity_hate'])])
            }
            train_data.append(comment)
        print("Done.", len(train_data), " comments loaded!")
        return train_data


def tokens_to_indices(train_data, glove_model):
    print("Start converting tokens to indices")
    indexed_train_data = []
    for comment in train_data:
        indices = []
        for token in comment['tokens']:
            token = token.lower()
            index = glove_model['tokens'].index('something')
            if token in glove_model['tokens']:
                index = glove_model['tokens'].index(token)
            indices.append(index)

        embedded_comment = {
            'indices': indices,
            'labels': comment['labels']
        }

        indexed_train_data.append(embedded_comment)
    print("Done.", len(indexed_train_data), " comments were indexed!")
    return indexed_train_data


indexed_train_data = []
if os.path.exists('.\\indexed_train_data.pickle'):
    with open('.\\indexed_train_data.pickle', 'rb') as handle:
        indexed_train_data = pickle.load(handle)

if len(indexed_train_data) is 0:
    glove_model = load_glove_model('.\\glove.6B\\glove.6B.50d.txt')
    train_data = load_train_data('.\\all\\train.csv')
    indexed_train_data = tokens_to_indices(train_data, glove_model)
    with open('.\\indexed_train_data.pickle', 'wb') as handle:
        pickle.dump(indexed_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


y = 6
