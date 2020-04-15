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


def get_toxic_comments_model(embeddings_matrix, vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH):
    import tensorflow as tf
    # from tensorflow.keras.layers import Input, Bidirectional, Dense, LSTM, Embedding, Concatenate, SpatialDropout1D, \
    #     Reshape, BatchNormalization, Dropout, GlobalMaxPool1D
    # from tensorflow.keras.models import Model
    from keras.layers import Input, Bidirectional, SpatialDropout1D, Embedding, Conv1D, GlobalAveragePooling1D, \
        GRU, GlobalMaxPooling1D, Concatenate, Dense, Dropout
    from keras.models import Model

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(vocab_size, embedding_dim,
                           weights=[embeddings_matrix],
                           input_length=MAX_SEQUENCE_LENGTH,
                           trainable=False)(sequence_input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Conv1D(128, kernel_size = 1, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(1)(x)
    model = Model(sequence_input, preds)
    return model


def model_lstm_with_atten(embeddings_matrix, vocab_size, embedding_dim, MAX_SEQUENCE_LENGTH):
    from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
    from keras.models import Model
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(vocab_size, embedding_dim,
                           weights=[embeddings_matrix],
                           input_length=MAX_SEQUENCE_LENGTH,
                           trainable=False)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(63, return_sequences=True))(x)
    x = Attention(MAX_SEQUENCE_LENGTH)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="relu")(x)
    model = Model(inputs=inp, outputs=x)

    return model

