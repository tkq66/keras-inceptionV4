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
import TestDataGenerator as dg
import inception_v4
import numpy as np
from sys import argv


# If you want to use a GPU set its index here
trainingLabelFileName = "../../data/train.csv"
testImgFilePathRoot = "../../data/transferred_test/"
recordFilePath = "records/"
dropoutProb = 0.5


def trainingLabelGenerator(labelFileName):
    with open(labelFileName, newline='') as fileHandle:
        reader = csv.reader(fileHandle)
        reader.__next__()
        for fileLabelTuple in reader:
            yield fileLabelTuple


def main():
    weightName = argv[1]
    outFileName = argv[2]
    includeTop = True if weightName != 'imagenet' else False
    classes = len({i[1] for i in trainingLabelGenerator(trainingLabelFileName)})
    dataGenerator = dg.DataGenerator(imgFilePathRoot=testImgFilePathRoot)

    # Create model and load pre-trained weights
    model = inception_v4.create_model(num_classes=classes,
                                      dropout_prob=dropoutProb,
                                      weights=weightName,
                                      include_top=includeTop)
    # Make predictions on the test data
    x = dataGenerator.loadData(verbose=True)
    results = model.predict(x=x, verbose=1)

    fileNameList = dataGenerator.getFileNameList()
    with open(outFileName, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image_name", "category"])
        for i in range(len(fileNameList)):
            writer.writerow([fileNameList[i], np.argmax(results[i])])


if __name__ == "__main__":
    main()
