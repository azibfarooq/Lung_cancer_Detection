def wsi_prediction(model, img):
    from patchify import patchify, unpatchify
    import numpy as np

    max_sum = np.sum(np.ones((256,256,3)))
    img_patches = patchify(img/255, (256, 256, 3), step=256)
    x,y,z = img_patches.shape[0],img_patches.shape[1],img_patches.shape[2]
    out = np.zeros((x*256, y*256, 1))
    mask_patches = patchify(out, (256, 256, 1), step=256)

    for i in range(x):
        for j in range(y):
            for k in range(z):
                temp = img_patches[i][j][k]
                if np.sum(temp) < 0.9*max_sum:
                    pred = model.predict(np.expand_dims(temp, axis = 0))[0]
                    mask_patches[i][j][k] = 255*((pred > 0.5).astype(np.uint8))

    return unpatchify(mask_patches, out.shape)