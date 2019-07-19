import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from azureml.core.run import Run
from amlcallback import AMLCallback
from tensorflow.keras.optimizers import Adam
from six.moves.urllib.request import urlretrieve
from tensorflow.keras.callbacks import ModelCheckpoint

###################################################################
# Helpers                                                         #
###################################################################
def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def download(data_path):
    mnist_file = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    target = Path(data_path).joinpath('mnist.npz')
    if not os.path.exists(target):
        print('downloading {} ...'.format(mnist_file))
        file, output = urlretrieve(mnist_file, target)
    else:
        print('{} already exists, skipping step'.format(str(target)))
    return target

def draw_samples(run, log_path, title, X, y):
    fig, axes = plt.subplots(6, 20, figsize=(18, 7),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        ax.imshow(255 - X[i], cmap='gray')
        ax.set_title('{:.0f}'.format(y[i]))

    fig.title = title

    if not run.id.lower().startswith('offline'):
        run.log_image(title, plot=plt)
    else:
        plt.savefig(os.path.join(log_path, '{}.png'.format(title)), bbox_inches='tight')

    plt.close()

###################################################################
# Training                                                        #
###################################################################
def main(run, data_path, output_path, log_path, layer_width, batch_size, epochs, learning_rate):
    info('Data')
    file = download(data_path)

    with np.load(file) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    print('Drawing samples...')
    draw_samples(run, log_path, 'training_set', x_train, y_train)
    draw_samples(run, log_path, 'test_set', x_test, y_test)
    print('Done!')

    train_set = x_train.reshape((len(x_train), -1)) / 255.

    info('Training')

    # function shape
    model = keras.Sequential([
        keras.layers.Dense(layer_width, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # how to optimize the function
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # callbacks
    logaml = AMLCallback(run)
    filename = datetime.now().strftime("%d.%b.%Y.%H.%M")
    checkpoint = ModelCheckpoint(os.path.join(output_path, filename + '.e{epoch:02d}-{accuracy:.2f}.hdf5'))

    model.fit(train_set, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[logaml, checkpoint])

    info('Test')
    test_set = x_test.reshape((len(x_test), -1)) / 255.
    test_loss, test_acc = model.evaluate(test_set, y_test)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scikit-learn MNIST')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    parser.add_argument('-o', '--outputs', help='output directory', default='outputs')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=5, type=int)
    parser.add_argument('-l', '--layer', help='number nodes in internal layer', default=128, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
    parser.add_argument('-r', '--lr', help='learning rate', default=0.001, type=float)
    args = parser.parse_args()

    run = Run.get_context()
    offline = run.id.startswith('OfflineRun')
    print('AML Context: {}'.format(run.id))

    args = {
        'run': run,
        'data_path': check_dir(args.data).resolve(),
        'output_path': check_dir(args.outputs).resolve(),
        'log_path': check_dir(args.logs).resolve(),
        'epochs': args.epochs,
        'layer_width': args.layer,
        'batch_size': args.batch,
        'learning_rate': args.lr
    }

    # log output
    if not offline:
        for item in args:
            if item != 'run':
                run.log(item, args[item])

    info('Args')
    print('Using Tensorflow {}'.format(tf.__version__))
    for i in args:
        print('{} => {}'.format(i, args[i]))

    main(**args)
