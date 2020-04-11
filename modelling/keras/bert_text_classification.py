def bert_create_model():
  config = BertConfig()
  config.output_hidden_states = False
  bert_model = TFBertModel.from_pretrained(bert_pretrained_model, config=config)

  text_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
  text_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
  text_segment = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

  text_embedding = bert_model(text_id, attention_mask=text_mask, token_type_ids=text_segment)[0]
  x = tf.keras.layers.GlobalAveragePooling1D()(text_embedding)
  x = tf.keras.layers.Dropout(0.3)(x)

  x = tf.keras.layers.Dense(32, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.3)(x)

  x = tf.keras.layers.BatchNormalization()(x)

  out = tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid')(x)

  model = tf.keras.models.Model(inputs=[text_id, text_mask, text_segment], outputs=out)

  return model