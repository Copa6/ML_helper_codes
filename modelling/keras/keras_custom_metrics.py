def rmse(y_true, y_pred):
    import tensorflow.keras.backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def auc(y_true, y_pred):
    from sklearn import metrics
    import tensorflow as tf

    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


def recall(y_true, y_pred):
    # from tensorflow.keras import backend as K
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def precision(y_true, y_pred):
    # from tensorflow.keras import backend as K
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def f1(y_true, y_pred):
    # from tensorflow.keras import backend as K
    from keras import backend as K
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p*r) / (p+r+K.epsilon()))


def f1_sklearn(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)

