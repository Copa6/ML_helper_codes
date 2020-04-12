def create_categorical_embeddings_model(data, catcols):
    import numpy as np
    # from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, SpatialDropout1D, \
    #     Reshape, BatchNormalization, Dropout, Embedding
    # from tensorflow.keras.models import Model
    from keras.layers import Input, Dense, Concatenate, SpatialDropout1D, \
        Reshape, BatchNormalization, Dropout, Embedding
    from keras.models import Model

    categorical_inputs = []
    categorical_embed_outputs = []

    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values) / 2), 50))
        inp = Input(shape=(1,))
        out = Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = SpatialDropout1D(0.3)(out)
        out = Reshape(target_shape=(embed_dim,))(out)
        categorical_inputs.append(inp)
        categorical_embed_outputs.append(out)

    categorical_out = Concatenate()(categorical_embed_outputs)
    categorical_out = BatchNormalization()(categorical_out)
    x = Dense(32, activation="relu")(categorical_out)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Dense(8, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    model = Model(inputs=categorical_inputs, outputs=categorical_out)
    return model
