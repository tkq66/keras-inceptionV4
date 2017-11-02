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

import DataGenerator as dg
import TrainingCallback as tc
from keras import optimizers
from keras import losses
import inception_v4
import csv


# If you want to use a GPU set its index here
trainingLabelFileName = '../../data/train_overfit.csv'
cpuCores = 4
trainingEpoch = 1
batchSize = 32
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
    classes = len({i[1] for i in trainingLabelGenerator(trainingLabelFileName)})
    dataGenerator = dg.DataGenerator(validation_split=validationPercentage,
                                     num_classes=classes,
                                     batch_size=batchSize,
                                     shuffle=True)

    # Create model and load pre-trained weights
    model = inception_v4.create_model(num_classes=classes,
                                      weights='imagenet',
                                      include_top=False)

    # Configure training hyper-parameters
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    # Train the new model
    history = model.fit_generator(generator=dataGenerator.generateTrain(),
                                  steps_per_epoch=dataGenerator.getTrainStepsPerEpoch(),
                                  epochs=trainingEpoch,
                                  verbose=1,
                                  validation_data=dataGenerator.generateValidation(),
                                  validation_steps=dataGenerator.getValidationSteps(),
                                  workers=cpuCores,
                                  use_multiprocessing=True)
    print(history)


if __name__ == "__main__":
    main()
