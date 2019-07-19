import tensorflow as tf


class AMLCallback(tf.keras.callbacks.Callback):

    def __init__(self, run):
        self.run = run
        self.offline = self.run.id.startswith('OfflineRun')
        print('Offline AML Context: {}'.format(self.offline))

    def set_params(self, params):
        self.params = params
        self.metics = self.params['metrics']
        if not self.offline:
            for item in self.params:
                self.run.log(item, self.params[item])

    def set_model(self, model):
        self.model = model

    def on_train_batch_end(self, batch, logs=None):
        if not self.offline:
            for item in self.metrics:
                self.run.log(item, logs[item])

    def on_epoch_end(self, epoch, logs=None):
        if not self.offline:
            for item in self.metrics:
                self.run.log('epoch_{}'.format(item), logs[item])