import numpy as np
import toxic_comments_classifier

glove_model = toxic_comments_classifier.GloveModel()
glove_model.load_glove_model('.\\glove.6B\\glove.6B.50d.txt')

toxic_comments = toxic_comments_classifier.ToxicComments(glove_model)
toxic_comments.load_toxic_comments('.\\all\\train.csv', '.\\all\\test.csv', '.\\all\\test_labels.csv')
_, _ = toxic_comments.get_next_batch(64)
y = 5


