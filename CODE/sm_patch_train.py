import numpy as np, os, tifffile as tiff
from patchify import patchify

os.chdir('D:/LCDC/Dataset/Training/Custom_Annos/') # Parent Directory
"""
Directory Structure
Parent Directory
|---Raw_Data
|---|---images [0...20]
|---|---|---1.tif
|---|---|---2.tif ...
|---|---masks [0...20]
|---|---|---1.tif
|---|---|---2.tif...
|---Train
|---|---images
|---|---|---images [15k]
|---|---|---|---1_0.jpg
|---|---|---|---1_1.jpg ...
|---|---masks 
|---|---|---masks [15k]
|---|---|---|---1_0.jpg
|---|---|---|---1_1.jpg...
|---Val
|---|---images
|---|---|---images [3k]
|---|---|---|---17_0.jpg
|---|---|---|---17_1.jpg ...
|---|---masks 
|---|---|---masks [3k]
|---|---|---|---17_0.tif
|---|---|---|---17_1.tif...
|---Test
|---|---images
|---|---|---images [2.5k]
|---|---|---|---19_0.jpg
|---|---|---|---19_1.jpg ...
|---|---masks 
|---|---|---masks [2.5k]
|---|---|---|---19_0.jpg
|---|---|---|---19_1.jpg...
"""


# Making Patches for training, validation and test folders
# Total 20 wsi images and custom annotated masks named from 1 - 10.tif
# Train: 1 - 16.tif 
# Val: 17 - 18.tif 
# Test: 19 - 20.tif

#Only changed range and write folder in below for-loop to make train, val and test folders
for name in range(19,21):
    img = tiff.imread(f'./Raw_Data/images/{name}.tif')
    mask = tiff.imread(f'./Raw_Data/masks/{name}.tif')
    img_patches = patchify(img, (512,512,3), step=512)
    mask_patches = patchify(mask, (512,512), step=512)
    x = 0
    for i in range(mask_patches.shape[0]):
        for j in range(mask_patches.shape[1]):
            temp = mask_patches[i][j]
            if np.max(temp) > 0:
                temp = 255*np.where(temp == 2, 1, 0).astype(np.uint8)
                tiff.imwrite(f'./Test/masks/{name}_{x}.jpg',temp, photometric='minisblack')
                tiff.imwrite(f'./Test/images/{name}_{x}.jpg',img_patches[i][j][0], photometric='rgb')
                x+=1



def train_gen(img_path, mask_path, b_s, kwargs={}):
    import tensorflow as tf
    from tensorflow import keras
    from keras.preprocessing.image import ImageDataGenerator
    import random
    seed = 909
    wsi_datagen = ImageDataGenerator(kwargs, rescale=1/255)
    mask_datagen = ImageDataGenerator(kwargs, rescale=1/255)

    wsi_gen = wsi_datagen.flow_from_directory(img_path, batch_size = b_s, target_size=(512,512), class_mode = None, seed = seed)
    mask_gen = mask_datagen.flow_from_directory(mask_path, batch_size = b_s, target_size=(512,512), class_mode = None, seed = seed, color_mode='grayscale')

    return zip(wsi_gen, mask_gen)




train_imgs = './Train/images/'
train_masks = './Train/masks/'

val_imgs = './Val/images/'
val_masks = './Val/masks/'

test_imgs = './Test/images/'
test_masks = './Test/masks/'


aug = dict(horizontal_filp=True, 
        vertical_flip=True, 
        fill_mode='nearest', 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        brightness_range=[0.4,1.5], 
        zoom_range=0.3)

b_s = 32
train_data = train_gen(train_imgs, train_masks, b_s, aug)
val_data = train_gen(val_imgs, val_masks, b_s)
test_data = train_gen(test_imgs, test_masks, b_s)


import tensorflow as tf, pickle
from tensorflow.keras.optimizers import Adam

# Segmentation model UNET
import segmentation_models as sm
sm.set_framework('tf.keras')
model = sm.Unet('resnet34',input_shape=(512,512,3), encoder_weights = None)


model.compile(optimizer=Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['acc',tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),tf.keras.metrics.MeanIoU(2)],
)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint('./model.h5', save_best_only=True, verbose=2)]


history = model.fit(
      train_data,
      steps_per_epoch= 481, # training_images / batch_size
      epochs=30,
      validation_data=val_data,
      validation_steps = 7,
      callbacks = callbacks
   )

with open('./history.pkl', 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
