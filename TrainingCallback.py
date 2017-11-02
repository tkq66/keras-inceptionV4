from keras.callbacks import Callback


class TrainingCallback(Callback):

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return
