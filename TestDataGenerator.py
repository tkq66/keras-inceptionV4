import cv2
import inception_v4
from keras import backend as K
import numpy as np
from os import listdir
from tqdm import tqdm


class DataGenerator:

    def __init__(self, imgFilePathRoot="../../data/transferred_test/"):
        self.imgFilePathRoot = imgFilePathRoot
        self.fileNameList = listdir(imgFilePathRoot)

        self.imShape = [3, 299, 299] if K.image_data_format() == "channels_first" else [299, 299, 3]

    def getFileNameList(self):
        return self.fileNameList

    def loadData(self, verbose=False):
        if verbose:
            print("Loading input data...")
            x = np.empty([len(self.fileNameList)] + self.imShape)
            for i in tqdm(range(len(self.fileNameList))):
                x[i] = self.__getImage(self.fileNameList[i])
            print("Finished loading.")
            return x
        else:
            x = np.empty([len(self.fileNameList)] + self.imShape)
            for i in range(len(self.fileNameList)):
                x[i] = self.__getImage(self.fileNameList[i])
            return x

    def __getImage(self, fileName):
        fullFileName = self.imgFilePathRoot + fileName
        image = self.__get_processed_image(fullFileName)
        return image

    def __get_processed_image(self, img_path):
        # Load image and convert from BGR to RGB
        im = np.asarray(cv2.imread(img_path))[:, :, ::-1]
        im = self.__central_crop(im, 0.875)
        im = cv2.resize(im, (299, 299))
        im = inception_v4.preprocess_input(im)
        if K.image_data_format() == "channels_first":
            im = np.transpose(im, (2, 0, 1))
            im = im.reshape(3, 299, 299)
        else:
            im = im.reshape(299, 299, 3)
        return im

    # This function comes from Google's ImageNet Preprocessing Script
    def __central_crop(self, image, central_fraction):
        """Crop the central region of the image.
        Remove the outer parts of an image but retain the central region of the image
        along each dimension. If we specify central_fraction = 0.5, this function
        returns the region marked with "X" in the below diagram.
         --------
        |        |
        |  XXXX  |
        |  XXXX  |
        |        |   where "X" is the central 50% of the image.
         --------
        Args:
        image: 3-D array of shape [height, width, depth]
        central_fraction: float (0, 1], fraction of size to crop
        Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
        Returns:
        3-D array
        """
        if central_fraction <= 0.0 or central_fraction > 1.0:
            raise ValueError('central_fraction must be within (0, 1]')
        if central_fraction == 1.0:
            return image

        img_shape = image.shape
        depth = img_shape[2]
        fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
        bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
        bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

        bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
        bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

        image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
        return image
