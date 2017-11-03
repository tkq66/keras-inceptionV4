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
        print(f"\nBatch {batch}: Begin Processing - {datetime.datetime.now()}")

    def on_batch_end(self, batch, logs={}):
        print(f"Batch {batch}: End Processing - {datetime.datetime.now()}")
        print(f"Batch {batch}: Begin Evaluation - {datetime.datetime.now()}")
        loss, acc = self.model.evaluate_generator(generator=self.validationGenerator(),
                                                  steps=self.validationSteps,
                                                  workers=self.cpuCores,
                                                  use_multiprocessing=True)
        with open(self.lossPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(loss) + "\n")
        with open(self.accPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(acc) + "\n")
        print(f"Validation loss: {loss}, acc: {acc} - {datetime.datetime.now()}\n")


class LossHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
