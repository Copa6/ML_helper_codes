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
