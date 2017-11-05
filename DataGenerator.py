import csv
import cv2
import inception_v4
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
from ThreadSafe import threadsafe_generator


class DataGenerator:

    def __init__(self,
                 validation_split=0.33,
                 num_classes=132,
                 batch_size=32,
                 shuffle=True,
                 trainFileName="../../data/train_overfit.csv",
                 imgFilePathRoot="../../data/transferred_train/"):
        self.validation_split = validation_split
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.trainFileName = trainFileName
        self.imgFilePathRoot = imgFilePathRoot

        # Load the training data file for reference on all training file name and class
        self.trainingDataReference = []
        with open(self.trainFileName, newline='') as fileHandle:
            reader = csv.reader(fileHandle)
            reader.__next__()
            for fileName, label in reader:
                if int(label) > self.num_classes:
                    raise ValueError("Label is greater than specified number of classes.")
                self.trainingDataReference.append(tuple((fileName, label)))

        # Pick the indices for the training and validation data set
        totalInput = len(self.trainingDataReference)
        self.totalValidationInput = int(np.floor(totalInput * validation_split))
        self.totalTrainInput = totalInput - self.totalValidationInput
        self.allIndices = self.__getDataOrder(totalInput)
        validationIndicesOrder = np.random.choice(totalInput, self.totalValidationInput, replace=False)
        self.validationIndices = self.allIndices[validationIndicesOrder]
        self.trainIndices = np.delete(self.allIndices, validationIndicesOrder)
        assert len(self.validationIndices) == self.totalValidationInput
        assert len(self.trainIndices) == self.totalTrainInput

        # Calculate the begin and end index for each training batch
        self.stepsPerEpoch = int(np.ceil(self.totalTrainInput / self.batch_size))
        self.trainBatchOrder = (np.repeat(np.arange(self.stepsPerEpoch), 2).reshape(-1, 2) * self.batch_size) + \
                               (np.tile(np.array([0, self.batch_size]), self.stepsPerEpoch).reshape(-1, 2))
        self.trainBatchOrder[-1, 1] = self.totalTrainInput

        # Calculate the begin and end index for each validation batch
        self.validationSteps = int(np.ceil(self.totalValidationInput / self.batch_size))
        self.validationBatchOrder = (np.repeat(np.arange(self.validationSteps), 2).reshape(-1, 2) * self.batch_size) + \
                                    (np.tile(np.array([0, self.batch_size]), self.validationSteps).reshape(-1, 2))
        self.validationBatchOrder[-1, 1] = self.totalValidationInput

    def getTrainStepsPerEpoch(self):
        return self.stepsPerEpoch

    def getValidationSteps(self):
        return self.validationSteps

    def getTrainingSize(self):
        return self.totalTrainInput

    def getValidationSize(self):
        return self.totalValidationInput

    def loadAll(self):
        x, y = self.__getData(self.allIndices)
        return tuple((x, y))

    def loadTrain(self):
        x, y = self.__getData(self.trainIndices)
        return tuple((x, y))

    def loadValidation(self):
        x, y = self.__getData(self.validationIndices)
        return tuple((x, y))

    @threadsafe_generator
    def generateTrain(self):
        while True:
            if self.shuffle:
                np.random.shuffle(self.trainIndices)
            for begin, end in self.trainBatchOrder:
                batchOrder = self.trainIndices[begin:end]
                x, y = self.__getData(batchOrder)
                yield tuple((x, y))

    @threadsafe_generator
    def generateValidation(self):
        while True:
            for begin, end in self.validationBatchOrder:
                batchOrder = self.validationIndices[begin:end]
                x, y = self.__getData(batchOrder)
                yield tuple((x, y))

    def __getData(self, order):
        x = [self.__getImageFromDataReference(self.trainingDataReference[i]) for i in order]
        y = to_categorical([self.trainingDataReference[i][1] for i in order],
                           num_classes=self.num_classes)
        return np.asarray(x), y

    def __getImageFromDataReference(self, dataReference):
        fileName, label = dataReference
        fullFileName = self.imgFilePathRoot + fileName
        image = self.__get_processed_image(fullFileName)
        return image

    def __getDataOrder(self, totalItems):
        indices = np.arange(totalItems)
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

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
