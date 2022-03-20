def random_batch_selector(wsi_dir, mask_dir, des_dir, samples, val = 0):
    import random, glob, os, shutil
    img_paths = random.sample(glob.glob(wsi_dir+'*'), samples)
    #os.chdir(mask_dir)
    #x = [os.rename(name, name.split('_')[0]+'_MP_'+name.split('_')[-1]) for name in os.listdir(mask_dir)]
    if os.path.isdir(des_dir+'mask/') is False and os.path.isdir(des_dir+'wsi/') is False:
        os.makedirs(des_dir+'mask/')
        os.makedirs(des_dir+'wsi/')
    if not val:
        y = [shutil.copy(img, des_dir+'wsi/') for img in img_paths]
        mask_names = os.listdir(des_dir+'wsi/')
        z = [shutil.copy(mask_dir+name.replace('C','M'), des_dir+'mask/') for name in mask_names]
    else:
        y = [shutil.move(img, des_dir+'wsi/') for img in img_paths]
        mask_names = os.listdir(des_dir+'wsi/')
        z = [shutil.move(mask_dir+name.replace('C','M'), des_dir+'mask/') for name in mask_names]
    del y, z