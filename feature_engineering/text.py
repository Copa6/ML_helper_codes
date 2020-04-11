import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn import metrics
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from transformers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def preprocess_text(data, column=None, lemmatize=True, remove_punctuations_number_url=True):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    stop_words = set(stopwords.words('english'))

    return_data = pd.DataFrame()
    text_data = data[column] if column else data.iloc[:, 0]
    new_col = []
    for i, row_data in enumerate(text_data):
        row_data = str(row_data).strip().lower()  # Lower case the data
        if len(row_data) != 0:
            if lemmatize:
                parsed_data = nlp(row_data)
                data_lemmatized = [w.lemma_ if '-PRON-' not in w.lemma_ else w.text for w in parsed_data if
                                   not w.is_stop]

                row_data_lemmatized = ' '.join(data_lemmatized)
            else:
                row_data_lemmatized = row_data

            if remove_punctuations_number_url:
                # Remove url, numbers, punct
                row_data_punct_removed = re.sub(
                    r'((http|ftp|https|mailto)*://[A-Za-z0-9._?=+&%-]*(/[A-Za-z0-9._?=+&%-]*)*)|(www.*?\s+)|(.*?@.*?\s+)|[0-9]|[^\w\s]|[\p{P}]+',
                    '',
                    str(row_data_lemmatized))
            else:
                row_data_punct_removed = row_data
        else:
            row_data_punct_removed = row_data

        new_col.append(row_data_punct_removed)
    return_data[f"preprocessed_{col_name}"] = new_col

    return return_data


## For W2V
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
    return embedding_matrix


# For BERT
def bert_initialize_tokenizer_and_base_model():
    bert_pretrained_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
    return tokenizer, bert_pretrained_model


def get_max_len(data, text_column=None, cap=512):
    col = text_column if text_column else data.columns[0]
    return max(max([len(t.split()) for t in data[col]]), cap)


def _bert_convert_to_transformer_input(text, max_len, tokenizer):
    def _get_bert_input_ids(s1, s2, length):
        tokenized_data = tokenizer.encode_plus(s1, s2, add_special_tokens=True, max_length=length)
        input_ids = tokenized_data["input_ids"]
        input_segments = tokenized_data["token_type_ids"]
        num_tokens = len(input_ids)
        padding_length = length - num_tokens
        input_masks = [1] * num_tokens + [0] * (padding_length)
        padding_id = tokenizer.pad_token_id
        input_ids += [padding_id] * padding_length
        input_segments += [0] * padding_length

        return input_ids, input_masks, input_segments

    text_id, text_mask, text_segments = _get_bert_input_ids(text, None, max_len)
    return text_id, text_mask, text_segments


def bert_prepare_input_data(data, tokenizer, max_len):
    input_text_id = []
    input_text_mask = []
    input_text_segment = []

    for _, row in tqdm(data.iterrows()):
        text = row[0]
        text_id, text_mask, text_segment = _convert_to_transformer_input(text, max_len, tokenizer)

        input_text_id.append(text_id)
        input_text_mask.append(text_mask)
        input_text_segment.append(text_segment)

    return [
        np.asarray(input_text_id, dtype=np.int32),
        np.asarray(input_text_mask, dtype=np.int32),
        np.asarray(input_text_segment, dtype=np.int32)
    ]
