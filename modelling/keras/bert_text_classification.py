def bert_create_model(bert_pretrained_model, MAX_SEQUENCE_LENGTH, OUTPUT_SIZE):
    from transformers import BertConfig, TFBertModel
    import tensorflow as tf
    # from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization
    # from tensorflow.keras.models import Model
    from keras.layers import Input, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization
    from keras.models import Model

    config = BertConfig()
    config.output_hidden_states = False
    bert_model = TFBertModel.from_pretrained(bert_pretrained_model, config=config)

    text_id = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    text_mask = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    text_segment = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    text_embedding = bert_model(text_id, attention_mask=text_mask, token_type_ids=text_segment)[0]
    x = GlobalAveragePooling1D()(text_embedding)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    x = BatchNormalization()(x)

    out = Dense(OUTPUT_SIZE, activation='sigmoid')(x)

    model = Model(inputs=[text_id, text_mask, text_segment], outputs=out)

    return model
