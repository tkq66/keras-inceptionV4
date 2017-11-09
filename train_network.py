"""
Copyright 2017 TensorFlow Authors and Kent Sommer

Modified by Teekayu Klongtruajrok for the purpose of CS5242 for NUS School of Computing

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import DataGenerator as dg
from TrainingCallback import BatchEval, LossHistory, BatchHistory
import inception_v4
import json
from keras import optimizers
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sys import argv
import uuid


# If you want to use a GPU set its index here
trainingLabelFileName = "../../data/train.csv"
imgFilePathRoot = "../../data/transferred_train/"
testImgFilePathRoot = "../../data/transferred_test/"
mode = ""
recordFilePath = "records/"
cpuCores = 16
trainingEpoch = 200
batchSize = 20
validationPercentage = 0.2
learningRate = 0.00001
momentum = 0.9
dropoutProb = 0.5
optimizer = optimizers.SGD(lr=learningRate, momentum=momentum, nesterov=True)
# optimizer = optimizers.Adam()
loss = losses.categorical_crossentropy


def trainingLabelGenerator(labelFileName):
    with open(labelFileName, newline='') as fileHandle:
        reader = csv.reader(fileHandle)
        reader.__next__()
        for fileLabelTuple in reader:
            yield fileLabelTuple


def main():
    mode = argv[1]
    weightName = argv[2]
    includeTop = True if weightName != 'imagenet' else False
    sessionId = str(uuid.uuid4())
    classes = len({i[1] for i in trainingLabelGenerator(trainingLabelFileName)})
    dataGenerator = dg.DataGenerator(validation_split=validationPercentage,
                                     num_classes=classes,
                                     batch_size=batchSize,
                                     shuffle=True,
                                     trainFileName=trainingLabelFileName,
                                     imgFilePathRoot=imgFilePathRoot)

    # Create model and load pre-trained weights
    model = inception_v4.create_model(num_classes=classes,
                                      dropout_prob=dropoutProb,
                                      weights=weightName,
                                      include_top=includeTop)

    # Configure training hyper-parameters
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    if mode == "train:

        # Train the new model
        # batchEval = BatchEval(validationGenerator=dataGenerator.generateValidation,
        #                       validationSteps=dataGenerator.getValidationSteps(),
        #                       outputFileLocation=recordFilePath,
        #                       sessionId=sessionId,
        #                       cpuCores=cpuCores)
        # lossHistory = LossHistory(sessionId=sessionId)
        batchHistory = BatchHistory(sessionId=sessionId)
        # earlyStopper = EarlyStopping(monitor="val_acc", patience=10)
        checkpointFileName = "checkpoints/weights_" + sessionId + ".hdf5"
        checkpointer = ModelCheckpoint(filepath=checkpointFileName, monitor="val_acc", verbose=1, save_best_only=True)
        x, y = dataGenerator.loadTrain(verbose=True)
        validationData = dataGenerator.loadValidation(verbose=True)
        history = model.fit(x=x,
                            y=y,
                            batch_size=batchSize,
                            epochs=trainingEpoch,
                            verbose=1,
                            callbacks=[batchHistory, checkpointer],
                            validation_data=validationData,
                            shuffle=True)
        # history = model.fit_generator(generator=dataGenerator.generateTrain(),
        #                               steps_per_epoch=dataGenerator.getTrainStepsPerEpoch(),
        #                               epochs=trainingEpoch,
        #                               verbose=1,
        #                               callbacks=[batchHistory, checkpointer],
        #                               validation_data=dataGenerator.generateValidation(),
        #                               validation_steps=dataGenerator.getValidationSteps())
        historyFilePath = recordFilePath + "history_" + sessionId + ".json"
        with open(historyFilePath, "w") as fp:
            json.dumps(history.history, fp)
    elif mode == "test":
    
    else:
        raise ValueError("Mode not recognized! Please specify either 'train' or 'test'.")

if __name__ == "__main__":
    main()
