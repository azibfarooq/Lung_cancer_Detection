def train_gen(img_path, mask_path, b_s):
    import tensorflow as tf
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator
    import random
    seed = 909
    wsi_datagen = ImageDataGenerator(rescale=1/255)
    mask_datagen = ImageDataGenerator(rescale=1/255)

    wsi_gen = wsi_datagen.flow_from_directory(img_path, batch_size = b_s, class_mode = None, seed = seed)
    mask_gen = mask_datagen.flow_from_directory(mask_path, batch_size = b_s, class_mode = None, seed = seed, color_mode='grayscale')

    return zip(wsi_gen, mask_gen)