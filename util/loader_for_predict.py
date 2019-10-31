from PIL import Image
import numpy as np
import glob
import os

import random
from natsort import natsorted


class LoaderForPredict(object):
    def __init__(self, dir_original, init_size=(128, 128)):
        self._data, self._data_size = LoaderForPredict.import_data(dir_original, init_size)

    def get_dataset(self):
        return self._data, self._data_size

    def import_data(self, dir_original, init_size=None):
        paths_original = LoaderForPredict.generate_paths(dir_original)
        # pathをファイル順にソートする
        paths_original = natsorted(paths_original)

        images_original, images_original_size = LoaderForPredict.extract_images(paths_original, init_size)

        return images_original, images_original_size

    @staticmethod
    def extract_images(paths_original, init_size):
        images_original,images_original_size = [], []

        # Load images
        print("Loading original images", end="", flush=True)
        for image, original_size in LoaderForPredict.image_generator(paths_original, init_size, antialias=True):
            images_original.append(image)
            images_original_size.append(original_size)
            if len(images_original) % 100 == 0:
                print(".", end="", flush=True)
        print(" Completed", flush=True)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)

        return images_original, images_original_size

    @staticmethod
    def image_generator(file_paths, init_size=None, antialias=True):
        '''
        リサイズした画像とサイズを返す
        ラベルはアンチエイリアスとノーマライズはしない
        '''
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                
                image = Image.open(file_path)
                original_size = (image.width, image.height)

                if antialias:
                    image = image.resize(init_size, Image.ANTIALIAS)
                else:
                    image = image.resize(init_size)
                
                # delete alpha channel
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.asarray(image)
                if normalization:
                    image = image / 255.0
                yield image, original_size
    
    @staticmethod
    def generate_paths(dir_original):
        paths_original = glob.glob(str(dir_original) + "/*")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_original))#.以降を削除し、ファイルネームのみにする
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))#ファイルネーム.jpgにする

        return paths_original
