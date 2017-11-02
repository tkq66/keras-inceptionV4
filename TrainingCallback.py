from keras.callbacks import Callback


class BatchEval(Callback):

    def __init__(self,
                 validationGenerator,
                 validationSteps,
                 cpuCores=4,
                 lossPerBatchOutFileName="loss-batch.txt",
                 accPerBatchOutFileName="acc-batch.txt"):
        self.validationGenerator = validationGenerator
        self.validationSteps = validationSteps
        self.cpuCores = cpuCores
        self.lossPerBatchOutFileName = lossPerBatchOutFileName
        self.accPerBatchOutFileName = accPerBatchOutFileName

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        print(f"Batch {batch}: Begin Processing\n")

    def on_batch_end(self, batch, logs={}):
        print(f"Batch {batch}: End Processing\n")
        print(f"Batch {batch}: Begin Evaluation\n")
        loss, acc = self.model.evaluate_generator(generator=self.validationGenerator(),
                                                  steps=self.validationSteps,
                                                  workers=self.cpuCores,
                                                  use_multiprocessing=True)
        with open(self.lossPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(loss) + "\n")
        with open(self.accPerBatchOutFileName, "a") as fileHandle:
            fileHandle.write(str(acc) + "\n")
        print(f"\nTesting loss: {loss}, acc: {acc}\n")
