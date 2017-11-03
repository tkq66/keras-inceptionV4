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
import uuid


# If you want to use a GPU set its index here
trainingLabelFileName = "../../data/train.csv"
imgFilePathRoot = "../../data/transferred_train/"
recordFilePath = "records/"
cpuCores = 16
trainingEpoch = 200
batchSize = 16
validationPercentage = 0.2
momentum = 0.9
optimizer = optimizers.SGD(momentum=momentum, nesterov=True)
loss = losses.categorical_crossentropy


def trainingLabelGenerator(labelFileName):
    with open(labelFileName, newline='') as fileHandle:
        reader = csv.reader(fileHandle)
        reader.__next__()
        for fileLabelTuple in reader:
            yield fileLabelTuple


def main():
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
                                      weights='imagenet',
                                      include_top=False)

    # Configure training hyper-parameters
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    # Train the new model
    batchEval = BatchEval(validationGenerator=dataGenerator.generateValidation,
                          validationSteps=dataGenerator.getValidationSteps(),
                          outputFileLocation=recordFilePath,
                          sessionId=sessionId,
                          cpuCores=cpuCores)
    lossHistory = LossHistory(sessionId=sessionId)
    batchHistory = BatchHistory(sessionId=sessionId)
    earlyStopper = EarlyStopping(monitor="val_acc", patience=10)
    checkpointFileName = "checkpoints/weights_" + sessionId + ".hdf5"
    checkpointer = ModelCheckpoint(filepath=checkpointFileName, monitor="val_acc", verbose=1, save_best_only=True)
    history = model.fit_generator(generator=dataGenerator.generateTrain(),
                                  steps_per_epoch=dataGenerator.getTrainStepsPerEpoch(),
                                  epochs=trainingEpoch,
                                  verbose=1,
                                  callbacks=[batchHistory, earlyStopper, checkpointer],
                                  validation_data=dataGenerator.generateValidation(),
                                  validation_steps=dataGenerator.getValidationSteps())
    historyFilePath = recordFilePath + "history_" + sessionId + ".json"
    with open(historyFilePath, "w") as fp:
        json.dumps(history.history, fp)


if __name__ == "__main__":
    main()
