from PIL import Image
import numpy as np
import os
Image.MAX_IMAGE_PIXELS = 1000000000
# 元となる画像の読み込み
np.set_printoptions(threshold=np.inf)

for i in range(40):
    num ='{0:02d}'.format(i)
    img = Image.open('./train_images/train_hv_'+ num +'.jpg')
    img2 = Image.open('./train_images/train_hh_'+ num +'.jpg')
    label = Image.open('./train_annotations/train_'+num+'.png')

    blend = Image.blend(img,img2, 0.75)# img*a + img2 *(1-a)
    blend = blend.resize((1024, 1024), Image.ANTIALIAS)
    label = label.resize((1024, 1024))
    number_padded = '{0:02d}'.format(i)
    blend.save(os.path.join("./blend/train_images/", ("train_" + number_padded + ".jpg")))
    label.save(os.path.join("./blend/train_annotations/", ("train_" + number_padded + ".png")))
