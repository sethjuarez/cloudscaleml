import tensorflow as tf


class AMLCallback(tf.keras.callbacks.Callback):

    def __init__(self, run):
        self.run = run
        self.offline = self.run.id.startswith('OfflineRun')
        print('Offline AML Context: {}'.format(self.offline))

    def set_params(self, params):
        self.params = params
        print(self.params)
        if not self.offline:
            for item in self.params:
                self.run.log(item, self.params[item])

    def set_model(self, model):
        self.model = model
        print(self.model)

    def on_train_batch_end(self, batch, logs=None):
        if not self.offline:
            self.run.log('loss', logs['loss'])
            self.run.log('accuracy', logs['accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        if not self.offline:
            self.run.log('epoch_loss', logs['loss'])
            self.run.log('epoch_accuracy', logs['accuracy'])