def import_libs():
    import numpy as np, glob, os, shutil, matplotlib.pyplot as plt, cv2 as cv, random, tifffile as tiff, imagecodecs
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,Dense,ReLU,concatenate,Activation, Dropout,Conv2DTranspose, Concatenate, Conv2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.metrics import categorical_crossentropy