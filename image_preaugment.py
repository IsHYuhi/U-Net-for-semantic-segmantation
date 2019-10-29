from PIL import Image
import numpy as np
import glob
import os

import random
from natsort import natsorted

Image.MAX_IMAGE_PIXELS = 1000000000

CATEGORY = (
        "no sea ice",
        "sea ice density level 1",
        "sea ice density level 2",
        "sea ice density level 3",
        "sea ice density level 4",
        "sea ice density level 5",
        "sea ice density level 6",
        "sea ice density level 7",
        "sea ice density level 8",
        "sea ice density level 9",
        "sea ice density level 10",
        "lake",
        "land"
    )

class ImageLoader(object):
    ROOT_DIR = "augmented_images"
    TRAIN = "train_images"
    LABEL = "label_images"

    def __init__(self, dir_original, dir_segmented, init_size=(128, 128), one_hot=False):
        self._data = ImageLoader.import_data(dir_original, dir_segmented, init_size, one_hot)
        self._root_dir = self.ROOT_DIR
        self._train_dir = os.path.join(self._root_dir, self.TRAIN)
        self._label_dir = os.path.join(self._root_dir, self.LABEL)
        if not os.path.exists(self._root_dir):
            os.makedirs(self._root_dir, exist_ok=True)
        if not os.path.exists(self._train_dir):
            os.makedirs(self._train_dir, exist_ok=True)
        if not os.path.exists(self._label_dir):
            os.makedirs(self._label_dir, exist_ok=True)

    @staticmethod
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=False):
        #pathを作る
        paths_original, paths_segmented = ImageLoader.generate_paths(dir_original, dir_segmented)
        #file name順にソート
        #sort files in file name order
        paths_original = natsorted(paths_original)
        paths_segmented = natsorted(paths_segmented)
        #imageを取り出す
        images_original, images_segmented, images_original_size = ImageLoader.extract_images(paths_original, paths_segmented, init_size, one_hot)

        return [images_original, images_segmented, images_original_size]

    @staticmethod
    def extract_images(paths_original, paths_segmented, init_size, one_hot):
        images_original, images_segmented = [], []
        images_original_size = []

        # Load images
        print("Loading original images", end="", flush=True)
        for image, original_size in ImageLoader.image_generator(paths_original, init_size, antialias=True):
            images_original.append(image)
            images_original_size.append(original_size)##addition
        print(" Done", flush=True)

        print("Loading label images", end="", flush=True)
        for image, _ in ImageLoader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)

        print(" Done")
        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        # Change indices which correspond to "void" from 255
        images_segmented = np.where(images_segmented == 255, len(CATEGORY)-1, images_segmented)

        # One hot encoding using identity matrix.
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(CATEGORY), dtype=np.uint8)
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented, images_original_size

    @staticmethod
    def generate_paths(dir_original, dir_segmented):
        paths_original = glob.glob(dir_original + "/*")
        paths_segmented = glob.glob(dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))#.以降を削除し、ファイルネームのみにする
        paths_original = list(map(lambda filename: dir_original + "/" + filename + ".jpg", filenames))#ファイルネーム.jpgにする

        return paths_original, paths_segmented

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=False, antialias=False):#normalizationはFalse
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
                image = Image.open(file_path)
                # save original size
                original_size = (image.width, image.height)
                #print(original_size)
                image = image.resize(init_size)
                # resize by init_size
                if init_size is not None and init_size != image.size:
                    if antialias:
                        image = image.resize(init_size, Image.ANTIALIAS)
                    else:
                        image = image.resize(init_size)
                # alpha channelを削除
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.asarray(image)
                if normalization: #normalizeは無し
                    image = image / 255.0
                yield image, original_size

    def save_augmented_image(self, original=True):#train:jpg, label:png
        self.flipud_images(len(self._data[0])) # 枚数が倍になる
        self.fliplr_images(len(self._data[0])) #　枚数がさらに倍 -> 4倍
        for i in range(len(self._data[0])):
            train = Image.fromarray(np.uint8(self._data[0][i]), mode="L")
            label = Image.fromarray(np.uint8(self._data[1][i]), mode="L")
            if original:
                train = train.resize((self._data[2][i]), Image.ANTIALIAS)
                label = label.resize((self._data[2][i]), Image.ANTIALIAS)
            number_padded = '{0:03d}'.format(i)
            train.save(os.path.join(self._train_dir, ("train_" + number_padded + ".jpg")))
            label.save(os.path.join(self._label_dir, ("train_" + number_padded + ".png")))

    def flipud_images(self, num):
        for i in range(num):
            fliped = (np.flipud(self._data[0][i]))[np.newaxis,:,:]
            self._data[0] = np.append(self._data[0],fliped, axis=0)
            fliped = (np.flipud(self._data[1][i]))[np.newaxis,:,:]
            self._data[1] = np.append(self._data[1],fliped , axis=0)
            #print(self._data[0].shape)

    def fliplr_images(self, num):
        for i in range(num):
            fliped = (np.fliplr(self._data[0][i]))[np.newaxis,:,:]
            self._data[0] = np.append(self._data[0],fliped, axis=0)
            fliped = (np.fliplr(self._data[1][i]))[np.newaxis,:,:]
            self._data[1] = np.append(self._data[1],fliped , axis=0)
            #print(self._data[0].shape)

if __name__ == "__main__":
    dataset_ImageLoader = ImageLoader(dir_original="./data_set/train_images",
                            dir_segmented="./data_set/train_annotations",
                            init_size=(256, 256))
    dataset_ImageLoader.save_augmented_image(original=False)