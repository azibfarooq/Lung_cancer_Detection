def model_train(model, train_data_gen, val_data_gen, model_save_dir, model_name, eps, spe):

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.metrics import categorical_crossentropy
    
    with tf.device('/cpu:0'):
        model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics='accuracy')
        callbacks = [keras.callbacks.ModelCheckpoint(model_save_dir+model_name, save_best_only=True)]

    with tf.device('/device:GPU:0'):
        history = model.fit(train_data_gen, steps_per_epoch = spe, epochs=eps, validation_data = val_data_gen, callbacks=callbacks)

    return history