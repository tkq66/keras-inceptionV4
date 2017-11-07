import datetime
from keras.callbacks import Callback


class BatchEval(Callback):

    def __init__(self,
                 validationGenerator,
                 validationSteps,
                 outputFileLocation="records/",
                 sessionId="",
                 cpuCores=4,
                 lossPerBatchOutFileName="val-loss-batch",
                 accPerBatchOutFileName="val-acc-batch"):
        self.validationGenerator = validationGenerator
        self.validationSteps = validationSteps
        self.sessionId = sessionId
        self.cpuCores = cpuCores
        self.lossPerBatchOutFileName = outputFileLocation + lossPerBatchOutFileName + "_" + sessionId + ".txt"
        self.accPerBatchOutFileName = outputFileLocation + accPerBatchOutFileName + "_" + sessionId + ".txt"

    def getSessionId(self):
        return self.sessionId

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        print("\nBatch {}: Begin Processing - {}".format(batch, datetime.datetime.now()))

    def on_batch_end(self, batch, logs={}):
        print("Batch {}: End Processing - {}".format(batch, datetime.datetime.now()))
        print("Batch {}: Begin Evaluation - {}".format(batch, datetime.datetime.now()))
        # loss, acc = self.model.evaluate_generator(generator=self.validationGenerator(),
        #                                           steps=self.validationSteps,
        #                                           workers=self.cpuCores,
        #                                           use_multiprocessing=True)
        loss, acc = self.model.evaluate_generator(generator=self.validationGenerator(),
                                                  steps=self.validationSteps)
        with open(self.lossPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(loss) + "\n")
        with open(self.accPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(acc) + "\n")
        print("Validation loss: {}, acc: {} - {}\n".format(loss, acc, datetime.datetime.now()))


class LossHistory(Callback):

    def __init__(self,
                 outputFileLocation="records/",
                 sessionId="",
                 lossPerBatchOutFileName="val-loss-batch",
                 accPerBatchOutFileName="val-acc-batch"):
        self.sessionId = sessionId
        self.lossPerBatchOutFileName = outputFileLocation + lossPerBatchOutFileName + "_" + sessionId + ".txt"
        self.accPerBatchOutFileName = outputFileLocation + accPerBatchOutFileName + "_" + sessionId + ".txt"

    def getSessionId(self):
        return self.sessionId

    def on_batch_begin(self, batch, logs={}):
        print("\nBatch {}: Begin Processing - {}".format(batch, datetime.datetime.now()))

    def on_batch_end(self, batch, logs={}):
        loss = logs["loss"]
        acc = logs["acc"]
        print("Batch {}: End Processing - {}".format(batch, datetime.datetime.now()))
        print("Batch {}: Begin Evaluation - {}".format(batch, datetime.datetime.now()))
        with open(self.lossPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(loss) + "\n")
        with open(self.accPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(acc) + "\n")
        print("Validation loss: {}, acc: {} - {}\n".format(loss, acc, datetime.datetime.now()))


class BatchHistory(Callback):

    def __init__(self,
                 outputFileLocation="records/",
                 sessionId="",
                 lossPerEpochOutFileName="val-loss-epoch",
                 accPerEpochOutFileName="val-acc-epoch",
                 lossPerBatchOutFileName="loss-batch",
                 accPerBatchOutFileName="acc-batch"):
        self.sessionId = sessionId
        self.lossPerBatchOutFileName = outputFileLocation + lossPerBatchOutFileName + "_" + sessionId + ".txt"
        self.accPerBatchOutFileName = outputFileLocation + accPerBatchOutFileName + "_" + sessionId + ".txt"
        self.lossPerEpochOutFileName = outputFileLocation + lossPerEpochOutFileName + "_" + sessionId + ".txt"
        self.accPerEpochOutFileName = outputFileLocation + accPerEpochOutFileName + "_" + sessionId + ".txt"

    def getSessionId(self):
        return self.sessionId

    def on_epoch_end(self, epoch, logs={}):
        loss = logs["val_loss"]
        acc = logs["val_acc"]
        with open(self.lossPerEpochOutFileName, "a") as fileHandle:
            fileHandle.write(str(loss) + "\n")
        with open(self.accPerEpochOutFileName, "a") as fileHandle:
            fileHandle.write(str(acc) + "\n")

    # def on_batch_end(self, batch, logs={}):
        # loss = logs["loss"]
        # acc = logs["acc"]
        # with open(self.lossPerBatchOutFileName, "a") as fileHandle:
        #     fileHandle.write(str(loss) + "\n")
        # with open(self.accPerBatchOutFileName, "a") as fileHandle:
        #     fileHandle.write(str(acc) + "\n")
        # print("\n")
