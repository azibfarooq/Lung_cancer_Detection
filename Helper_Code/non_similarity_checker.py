def non_similarity_checker(des):
    import os
    wsi = os.listdir(des+'wsi/')
    mask = os.listdir(des+'mask/')
    count = 0
    for i in range(len(wsi)):
        if wsi[i].split('_')[-1] != mask[i].split('_')[-1]:
            count += 1
            print(count, ' ### ',wsi[i],' ### ',mask[i])