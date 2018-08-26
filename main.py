import toxic_comments_classifier

toxic_comments_rnn = toxic_comments_classifier.ToxicCommentsRNN(
    glove_model_file_path='.\\glove.6B\\glove.6B.50d.txt',
    toxic_comments_train_file_path='.\\all\\train.csv',
    toxic_comments_test_file_path='.\\all\\test_merged.csv',
    toxic_comment_max_length=100,
    state_size=50,
    batch_size=8192,
    epochs=2000)
toxic_comments_rnn.build_graph()
toxic_comments_rnn.train_graph()
