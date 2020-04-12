import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, Bidirectional, Dense, LSTM, Embedding, Concatenate, SpatialDropout1D, \
    Reshape, BatchNormalization, Dropout, GlobalMaxPool1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def create_tokenizer(data, text_column, vocab_size=5000):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data[text_column])
    word_idx_map = tokenizer.word_index
    return tokenizer, word_idx_map


def tokenize_sentences_and_pad(data, text_column, tokenizer, max_len):
    tokenized_sents = tokenizer.texts_to_sequences(data[text_column])
    return pad_sequences(tokenized_sents, padding='post', maxlen=max_len)


def w2v_create_embeddings_matrix(embeddings_file, word_index_mapping, emb_dim=100):
    vocab_size = len(word_index_mapping) + 1  # Adding 1, as 0 index is reserved for OOV token
    embedding_matrix = np.zeros((vocab_size, emb_dim))
    with open(embeddings_file) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index_mapping:
                idx = word_index_mapping[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:emb_dim]
    return embedding_matrix, vocab_size


def get_max_len(data, text_column=None, cap=512):
    col = text_column if text_column else data.columns[0]
    return min(max([len(t.split()) for t in data[col]]), cap)


def scale_data(data, columns, save_model=True, model_name="standard_scaler"):
    from sklearn.preprocessing import MinMaxScaler
    from joblib import dump

    scaler = MinMaxScaler()
    scaler.fit(data[columns])
    transformed_columns = scaler.transform(data[columns])
    if save_model:
        dump(scaler, f"{model_name}.model")

    return pd.DataFrame(transformed_columns)


def scale_data_from_model(data, columns, model_file=None, model=None):
    from joblib import load
    if model_file:
        scaler = load(model_file)
    else:
        scaler = model
    transformed_columns = scaler.transform(data[columns])

    return pd.DataFrame(transformed_columns)


def label_encode_data(data, columns, save_model=True, model_name="label_encoder"):
    from sklearn.preprocessing import LabelEncoder
    from joblib import dump

    return_data = pd.DataFrame()

    models = {}
    for column in columns:
        encoder = LabelEncoder()
        encoder.fit(data[column])
        return_data[f"transformed_{column}"] = encoder.transform(data[column])
        models[column] = encoder

    if save_model:
        dump(models, f"{model_name}.model")

    return return_data


def label_encode_from_model(data, columns, model_file=None, model=None):
    from joblib import load
    import bisect
    if model_file:
        encoders = load(model_file)
    else:
        encoders = model

    return_data = pd.DataFrame()
    for column in columns:
        encoder = encoders[column]
        data[column] = data[column].map(lambda c: "<unk>" if c not in encoder.classes_ else c)
        encoder_classes = encoder.classes_.tolist()
        bisect.insort_left(encoder_classes, '<unk>')
        encoder.classes_ = encoder_classes
        return_data[f"transformed_{column}"] = encoder.transform(data[column])

    return return_data


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def text_model(embeddings_matrix, vocab_size, embedding_dim):
    tokenized_sents = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    embedded_sents = Embedding(vocab_size, embedding_dim,
                               weights=[embeddings_matrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)(tokenized_sents)

    lstm_out = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedded_sents)
    lstm_out = LSTM(16, dropout=0.2, recurrent_dropout=0.2)(lstm_out)
    out = Dense(4, activation="relu")(lstm_out)

    model = Model(inputs=tokenized_sents, outputs=out)

    return model


def categorical_model(data, catcols):
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


def numeric_model():
    numeric_data = Input((NUM_COL_SIZE,), dtype=tf.float32)

    model = Model(inputs=numeric_data, outputs=numeric_data)
    return model


def create_model(embeddings_matrix, vocab_size, embedding_dim, data, catcols):
    lstm = text_model(embeddings_matrix, vocab_size, embedding_dim)
    categorical = categorical_model(data, catcols)
    numerical = numeric_model()

    non_text_data = Concatenate()([categorical.output, numerical.output])
    non_text_data = Dense(4, activation="relu")(non_text_data)
    non_text_data = Dropout(0.2)(non_text_data)
    x = BatchNormalization()(non_text_data)

    x = Concatenate()([lstm.output, non_text_data])
    x = BatchNormalization()(x)
    # x = BatchNormalization()(lstm.output)

    x = Dense(4, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    y = Dense(1)(x)

    model = Model(inputs=[lstm.input, categorical.input, numerical.input], outputs=y)

    return model


# Read the input data
train = pd.read_csv("processed_data_with_additional_feats.csv")
test = pd.read_csv("processed_test_data_with_additional_feats.csv")
glove_embeddings = "glove.6B.100d.txt"

# Extract names of columns of relevance
text_column = "preprocessed_review_by_patient"
categorical = ["name_of_drug", "use_case_for_drug"]
numerics_data = ["effectiveness_rating", "number_of_times_prescribed"]
target = "base_score"

# Calculate data dependent constants
MAX_SEQUENCE_LENGTH = get_max_len(train, text_column, 512)
CAT_COL_SIZE = len(categorical)
NUM_COL_SIZE = len(numerics_data)

# Prepare data for model
tokenizer, word_idx_map = create_tokenizer(train, text_column)
train_sents_tokenized = tokenize_sentences_and_pad(train, text_column, tokenizer, MAX_SEQUENCE_LENGTH)
outputs = train[target].values
categorical_data = label_encode_data(train, categorical)
scaled_data = scale_data(train, numerics_data)

# Prepare test data
test_sents_tokenized = tokenize_sentences_and_pad(test, text_column, tokenizer, MAX_SEQUENCE_LENGTH)
test_categorical_data = label_encode_from_model(test, categorical, model_file="label_encoder.model")
test_scaled_data = scale_data_from_model(test, numerics_data, model_file="standard_scaler.model")

# Load W2V embeddings
embeddings_matrix, vocab_size = w2v_create_embeddings_matrix(glove_embeddings, word_idx_map, emb_dim=100)

# Train model for multiple iterations and predict in each iteration
val_preds_all = []
test_preds_all = []

all_train_data = pd.concat([pd.DataFrame(train_sents_tokenized), categorical_data, scaled_data], axis=1)
train_X, val_X, train_y, val_y = train_test_split(all_train_data, outputs, test_size=0.1, random_state=44)
all_test_data = pd.concat([pd.DataFrame(test_sents_tokenized), test_categorical_data, test_scaled_data], axis=1)


gkf = GroupKFold(n_splits=2).split(X=train_X.iloc[:, 0], groups=train_X.iloc[:, 0])
for train_index, val_index in gkf:
    train_inputs = train_X.iloc[train_index, :]
    train_outputs = train_y[train_index]

    val_inputs = train_X.iloc[val_index, :]
    val_outputs = train_y[val_index]

    es = callbacks.EarlyStopping(monitor='val_rmse', min_delta=0.0001, patience=5,
                                 verbose=1, mode='min', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.5,
                                      patience=2, min_lr=1e-7, mode='min', verbose=1)

    K.clear_session()
    with tf.device('/gpu:0'):
        model = create_model(embeddings_matrix, vocab_size, 100, train, categorical)
        # print(model.summary())
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])

        h = model.fit([train_X.iloc[:, :100], train_X.iloc[:, 100], train_X.iloc[:, 101], train_X.iloc[:, 102:104]],
                      train_y,
                      validation_data=(
                          [val_X.iloc[:, :100], val_X.iloc[:, 100], val_X.iloc[:, 101], val_X.iloc[:, 102:104]], val_y),
                      epochs=50,
                      batch_size=32,
                      callbacks=[es, rlr])
        val_preds = model.predict([val_X.iloc[:, :100], val_X.iloc[:, 100], val_X.iloc[:, 101], val_X.iloc[:, 102:104]])
        test_preds = model.predict([all_test_data.iloc[:, :100], all_test_data.iloc[:, 100], all_test_data.iloc[:, 101],
                                    all_test_data.iloc[:, 102:104]])
    val_preds_all.append(val_preds)
    test_preds_all.append(test_preds)
    print(f"RMSE on validation set - {metrics.mean_squared_error(val_y, val_preds)}")

# Combine predictions on validation set
val_data = pd.DataFrame(val_X)
for i, vp in enumerate(val_preds_all):
    val_data[f"preds_{i}"] = vp
val_data["actual"] = val_y
val_data.to_csv("val_preds.csv", index=False)

# combine predictions on test set
test_preds_final = []
test_preds_sum = sum(test_preds_all) / 10

for elem in test_preds_sum:
    test_preds_final.append(elem[0])

out_data = pd.DataFrame({
    "patient_id": test["patient_id"],
    "base_score": test_preds_final
})

out_data.to_csv("outputs.csv", index=False)
