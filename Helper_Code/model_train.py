def model_train(model, train_data_gen, val_data_gen, model_save_dir, model_name, eps, spe,  val_steps):

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.metrics import categorical_crossentropy
    
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics='accuracy')
    callbacks = [keras.callbacks.ModelCheckpoint(model_save_dir+model_name, save_best_only=True, verbose=2)]

    history = model.fit(train_data_gen, validation_data=val_data_gen, validation_steps=val_steps, steps_per_epoch = spe, epochs=eps, callbacks=callbacks)

    return history
