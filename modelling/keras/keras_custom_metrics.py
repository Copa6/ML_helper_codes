def rmse(y_true, y_pred):
    import tensorflow.keras.backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
