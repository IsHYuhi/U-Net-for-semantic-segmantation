from PIL import Image
import numpy as np
import os
Image.MAX_IMAGE_PIXELS = 1000000000
# 元となる画像の読み込み
np.set_printoptions(threshold=np.inf)
if not os.path.exists("./data_set/palette"):
    os.mkdir("./data_set/palette")

image_sample_palette = Image.open('./data_set/train_annotations/train_00.png')#基礎の画像、なんでもいいpaletteを取り出す用
image_sample_palette = image_sample_palette.convert("P") #RGBの時はグレースケール化
palette = image_sample_palette.getpalette()

#今回のタスク用,1~10のラベルを(青)全て同じ色で出力するためカラーパレットの一部変更
for i in range(3*1+2, 3*10+3, 3):
    palette[i] = 255
#湖(11)を緑にする
for i in range(3*11+1, 3*11+2, 3):
    palette[i] = 255
#陸(12)を赤にする
for i in range(3*12, 3*12+1, 3):
    palette[i] = 255

#print(palette)

for i in range(80):
    num ='{0:02d}'.format(i)
    label = Image.open('./data_set/train_annotations/train_'+num+'.png')
    label = label.resize((1024, 1024))#リサイズ
    label.putpalette(palette)#.convert("RGB")

    number_padded = '{0:02d}'.format(i)
    label.save(os.path.join("./data_set/palette/", ("train_" + number_padded + ".png")))