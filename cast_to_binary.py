from PIL import Image
import numpy as np
import os
Image.MAX_IMAGE_PIXELS = 1000000000
# 元となる画像の読み込み
np.set_printoptions(threshold=np.inf)
if not os.path.exists("./data_set/binary"):
    os.mkdir("./data_set/binary")
for i in range(10):
    num ='{0:03d}'.format(i)
    label = Image.open('./data_set/train_annotations/train_'+num+'.png')

    label = label.point(lambda x: 255 if (x > 0) and (x < 11) else 0)#, mode="8")   #海氷は1に変換

    label = label.resize((1024, 1024))
    label.save(os.path.join("./data_set/binary/", ("train_" + num + ".png")))
