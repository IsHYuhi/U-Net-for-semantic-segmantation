from PIL import Image
import numpy as np
import glob
import os

import random
from natsort import natsorted

Image.MAX_IMAGE_PIXELS = 1000000000

class Loader(object):
    def __init__(self, dir_original, dir_segmented, init_size=(128, 128), one_hot=True):
        self._data = Loader.import_data(dir_original, dir_segmented, init_size, one_hot)

    def get_all_dataset(self):
        return self._data

    def load_train_test(self, train_rate=0.80, shuffle=True):
        if train_rate < 0.0 or train_rate > 1.0:
            raise ValueError("input trainrate from 0.0 to 1.0.")
        if shuffle:
            self._data.shuffle()

        train_size = int(self._data.images_original.shape[0] * train_rate)
        data_size = int(len(self._data.images_original))

        train_set = self._data.devide(0, train_size)
        test_set = self._data.devide(train_size, data_size)

        return train_set, test_set


    @staticmethod
    def import_data(dir_original, dir_segmented, init_size=None, one_hot=True):
        # パスリストを作成
        paths_original, paths_segmented = Loader.generate_paths(dir_original, dir_segmented)

        # pathをファイル順にソートする
        paths_original = natsorted(paths_original)
        paths_segmented = natsorted(paths_segmented)

        # 画像データをndarrayで読み込む
        images_original, images_segmented, images_original_size = Loader.extract_images(paths_original, paths_segmented, init_size, one_hot)

        # カラーパレットを取得
        image_sample_palette = Image.open(paths_segmented[0])#一枚の画像からパレットを取り出す
        #im = np.array(image_sample_palette)
        #print(im.shape) #パレットのshapeを確認
        image_sample_palette = image_sample_palette.convert("P")
        palette = image_sample_palette.getpalette()

        #今回のタスク用,1~10のラベルを全て同じ色で出力するためカラーパレットの一部変更
        for i in range(3*2-1, 3*11+1, 3):
            palette[i] = 255

        #パレット確認
        #print(np.array(palette))
        return DataSet(images_original, images_segmented, palette, images_original_size)


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
    def extract_images(paths_original, paths_segmented, init_size, one_hot):
        images_original, images_segmented, images_original_size = [], [], []

        # Load images
        print("Loading original images", end="", flush=True)
        for image, original_size in Loader.image_generator(paths_original, init_size, antialias=True):
            images_original.append(image)
            images_original_size.append(original_size)
            if len(images_original) % 100 == 0:
                print(".", end="", flush=True)
        print(" Completed", flush=True)

        print("Loading segmented images", end="", flush=True)
        for image, _ in Loader.image_generator(paths_segmented, init_size, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 100 == 0:
                print(".", end="", flush=True)
        print(" Completed")

        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        # Change indices which correspond to "void" from 255
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        # One hot encoding using identity matrix.
        if one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented, images_original_size

    @staticmethod
    def cast_to_index(ndarray):
        return np.argmax(ndarray, axis=2)

    @staticmethod
    def cast_to_onehot(ndarray):
        identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
        return identity[ndarray]

    @staticmethod
    def image_generator(file_paths, init_size=None, normalization=True, antialias=False):
        '''
        リサイズした画像とサイズを返す
        ラベルはアンチエイリアスとノーマライズはしない
        '''
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):

                image = Image.open(file_path)
                original_size = (image.width, image.height)

                # to square
                #image = Loader.crop_to_square(image) #戻した時にサイズの比率が変わるのでここでは単純にresize()を使う
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
    def crop_to_square(image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))




class DataSet(object):
    #今回のタスク用それぞれidxのラベルがつく0~12
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

    def __init__(self, images_original, images_segmented, image_palette, images_original_size):
        assert len(images_original) == len(images_segmented), "image's and label's length aren't equal"
        self._images_original = images_original
        self._images_segmented = images_segmented
        self._image_palette = image_palette
        self._images_original_size = images_original_size

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def palette(self):
        return self._image_palette

    @property
    def length(self):
        return len(self._images_original)

    @property
    def images_original_size(self):
        return self._images_original_size

    @staticmethod
    def length_category():
        return len(DataSet.CATEGORY)


    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        return DataSet(images_original, images_segmented, self._image_palette)

    def shuffle(self):
        idx = np.arange(self._images_original.shape[0])
        np.random.shuffle(idx)
        self._images_original, self._images_segmented = self._images_original[idx], self._images_segmented[idx]

    def devide(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end], self._image_palette, self._images_original_size)

    def __call__(self, batch_size=20, shuffle=True):
        if batch_size < 1:
            raise ValueError("input batch_size more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size):
            batch = self.devide(start, start+batch_size)
            yield batch


if __name__ == "__main__":
    dataset_loader = Loader(dir_original="../data_set/train_images",
                            dir_segmented="../data_set/train_annotations")
    train, test = dataset_loader.load_train_test()