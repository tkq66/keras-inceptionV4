import csv
import cv2
import inception_v4
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
from ThreadSafe import threadsafe_generator
from tqdm import tqdm


class DataGenerator:

    def __init__(self,
                 validation_split=0.33,
                 num_classes=132,
                 batch_size=32,
                 shuffle=True,
                 trainFileName="../../data/train.csv",
                 imgFilePathRoot="../../data/transferred_train/"):
        self.validation_split = validation_split
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.trainFileName = trainFileName
        self.imgFilePathRoot = imgFilePathRoot

        # Load the training data file for reference on all training file name and class
        classIndexCollection = {}
        self.classCounter = {}
        self.trainingDataReference = []
        with open(self.trainFileName, newline='') as fileHandle:
            reader = csv.reader(fileHandle)
            reader.__next__()
            index = 0
            for fileName, label in reader:
                if int(label) > self.num_classes:
                    raise ValueError("Label is greater than specified number of classes.")
                self.classCounter[label] = self.classCounter.get(label, 0) + 1
                if label not in classIndexCollection:
                    classIndexCollection[label] = []
                classIndexCollection[label].append(index)
                self.trainingDataReference.append(tuple((fileName, label)))
                index += 1

        # Pick the indices for the training and validation data set
        totalInput = len(self.trainingDataReference)
        self.totalValidationInput = int(np.floor(totalInput * validation_split))
        self.totalTrainInput = totalInput - self.totalValidationInput

        self.rawInputIndices = np.arange(totalInput)

        # Sample a truly representative validation data
        count = 1
        tempValSum = 0
        samplesPerClass = []
        for label in self.classCounter:
            if count < self.num_classes:
                samplesPerClass.append(int(np.floor(self.classCounter[label] * validation_split)))
                tempValSum += samplesPerClass[-1]
            elif count == self.num_classes:
                samplesPerClass.append(self.totalValidationInput - tempValSum)
            count += 1
        np.random.shuffle(samplesPerClass)
        count = 0
        self.validationIndices = []
        for label in classIndexCollection:
            self.validationIndices += list(np.random.choice(classIndexCollection[label], samplesPerClass[count], replace=False))
            count += 1

        # Remove the data selected for validation, leaving with training samples
        self.trainIndices = np.delete(self.rawInputIndices, self.validationIndices)

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

        self.imShape = [3, 299, 299] if K.image_data_format() == "channels_first" else [299, 299, 3]

    def getValidationRepresentationError(self):
        validationClassCounter = {}
        for i in self.validationIndices:
            label = self.trainingDataReference[i][1]
            validationClassCounter[label] = validationClassCounter.get(label, 0) + 1
        valPercentage = {}
        for label in validationClassCounter:
            valPercentage[label] = validationClassCounter[label] / np.floor(self.classCounter[label])
        error = 0
        for label in valPercentage:
            error += (valPercentage[label] - self.validation_split) ** 2
        return error

    def getTrainStepsPerEpoch(self):
        return self.stepsPerEpoch

    def getValidationSteps(self):
        return self.validationSteps

    def getTrainingSize(self):
        return self.totalTrainInput

    def getValidationSize(self):
        return self.totalValidationInput

    def loadAll(self, verbose=False):
        if self.shuffle:
            np.random.shuffle(self.rawInputIndices)
        x, y = self.__getData(self.rawInputIndices, verbose)
        return tuple((x, y))

    def loadTrain(self, verbose=False):
        if self.shuffle:
            np.random.shuffle(self.trainIndices)
        x, y = self.__getData(self.trainIndices, verbose)
        return tuple((x, y))

    def loadValidation(self, verbose=False):
        x, y = self.__getData(self.validationIndices, verbose)
        return tuple((x, y))

    @threadsafe_generator
    def generateTrain(self, verbose=False):
        while True:
            if self.shuffle:
                np.random.shuffle(self.trainIndices)
            for begin, end in self.trainBatchOrder:
                batchOrder = self.trainIndices[begin:end]
                x, y = self.__getData(batchOrder, verbose)
                yield tuple((x, y))

    @threadsafe_generator
    def generateValidation(self, verbose=False):
        while True:
            for begin, end in self.validationBatchOrder:
                batchOrder = self.validationIndices[begin:end]
                x, y = self.__getData(batchOrder, verbose)
                yield tuple((x, y))

    def __getData(self, order, verbose=False):
        if verbose:
            print("Loading input data...")
            x = np.empty([len(order)] + self.imShape)
            for j in tqdm(range(len(order))):
                x[j] = self.__getImageFromDataReference(self.trainingDataReference[order[j]])
            print("Loading data label...")
            allLabels = [self.trainingDataReference[order[i]][1] for i in tqdm(range(len(order)))]
            print("One-hot encoding labels...")
            y = to_categorical(allLabels, num_classes=self.num_classes)
            print("Finished loading.")
            return x, y
        else:
            x = np.empty([len(order)] + self.imShape)
            for i in order:
                x[i] = self.__getImageFromDataReference(self.trainingDataReference[i])
            y = to_categorical([self.trainingDataReference[i][1] for i in order],
                               num_classes=self.num_classes)
            return x, y

    def __getImageFromDataReference(self, dataReference):
        fileName, label = dataReference
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
