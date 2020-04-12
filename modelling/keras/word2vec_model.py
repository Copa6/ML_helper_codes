def create_model(embeddings_matrix, vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH):
    """

    :param embeddings_matrix: Word2Vec embeddings matrix/ randomly initialized embeddings matric
    :param vocab_size: Size of vocabulary. Ensure to do a +1 to account for reserved token 0
    :param embedding_dim: Dimension of embeddings
    :param MAX_SEQUENCE_LENGTH: Length of maximum sequence
    :return: Keras model instance
    """
    import tensorflow as tf
    # from tensorflow.keras.layers import Input, Bidirectional, Dense, LSTM, Embedding, Concatenate, SpatialDropout1D, \
    #     Reshape, BatchNormalization, Dropout, GlobalMaxPool1D
    # from tensorflow.keras.models import Model
    from keras.layers import Input, Bidirectional, LSTM, Embedding
    from keras.models import Model

    tokenized_sents = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    embedded_sents = Embedding(vocab_size, embedding_dim,
                               weights=[embeddings_matrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)(tokenized_sents)

    lstm_out = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(embedded_sents)
    lstm_out = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(lstm_out)

    model = Model(inputs=tokenized_sents, outputs=lstm_out)

    return model
