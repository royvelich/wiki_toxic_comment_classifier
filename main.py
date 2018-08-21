import numpy as np
import toxic_comments_classifier
import tensorflow as tf

glove_model = toxic_comments_classifier.GloveModel()
glove_model.load_glove_model('.\\glove.6B\\glove.6B.50d.txt')

toxic_comments = toxic_comments_classifier.ToxicComments(glove_model)
toxic_comments.load_toxic_comments('.\\all\\train.csv', '.\\all\\test_merged.csv')

toxic_comments_rnn = toxic_comments_classifier.ToxicCommentsRNN(toxic_comments=toxic_comments, state_size=64, batch_size=64)
toxic_comments_rnn.build_graph()
toxic_comments_rnn.train_graph()
