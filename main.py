import numpy as np
import toxic_comments_classifier
import tensorflow as tf

glove_model = toxic_comments_classifier.GloveModel()
glove_model.load_glove_model('.\\glove.6B\\glove.6B.50d.txt')

toxic_comments_train = toxic_comments_classifier.ToxicComments(glove_model, 'train', 100)
toxic_comments_train.load_toxic_comments('.\\all\\train.csv')

toxic_comments_test = toxic_comments_classifier.ToxicComments(glove_model, 'test', 100)
toxic_comments_test.load_toxic_comments('.\\all\\test_merged.csv')

toxic_comments_rnn = toxic_comments_classifier.ToxicCommentsRNN(toxic_comments_train=toxic_comments_train, toxic_comments_test=toxic_comments_test, state_size=32, batch_size=8192, epochs=8000)
toxic_comments_rnn.build_graph()
toxic_comments_rnn.train_graph()
