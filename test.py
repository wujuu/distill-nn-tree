import numpy as np
import os
import tensorflow as tf

from models import ConvNet, SoftBinaryDecisionTree
from models.utils import brand_new_tfsession, draw_tree
from tensorflow.keras.callbacks import EarlyStopping, Callback
import glob


if __name__ == '__main__':
    sess = brand_new_tfsession()

    # load MNIST data
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # add channel dim
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

    # hold out last 10000 training samples for validation
    x_valid, y_valid = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)

    # retrieve image and label shapes from training data
    img_rows, img_cols, img_chans = x_train.shape[1:]
    n_classes = np.unique(y_train).shape[0]

    print(img_rows, img_cols, img_chans, n_classes)

    # convert labels to 1-hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    print(y_train.shape, y_valid.shape, y_test.shape)

    # normalize inputs and cast to float
    x_train = (x_train / np.max(x_train)).astype(np.float32)
    x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
    x_test = (x_test / np.max(x_test)).astype(np.float32)

    nn = ConvNet(img_rows, img_cols, img_chans, n_classes)
    nn.maybe_train(data_train=(x_train, y_train),
                   data_valid=(x_valid, y_valid),
                   batch_size=16, epochs=12)

    # nn.evaluate(x_valid, y_valid)
    # nn.evaluate(x_test, y_test)

    y_train_soft = nn.predict(x_train)
    print(y_train_soft.shape)

    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_valid_flat = x_valid.reshape((x_valid.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    print(x_train_flat.shape, x_valid_flat.shape, x_test_flat.shape)

    n_features = img_rows * img_cols * img_chans
    tree_depth = 4
    penalty_strength = 1e+1
    penalty_decay = 0.25
    ema_win_size = 1000
    inv_temp = 0.01
    learning_rate = 5e-03
    batch_size = 4

    sess = brand_new_tfsession(sess)

    tree = SoftBinaryDecisionTree(tree_depth, n_features, n_classes,
                                  penalty_strength=penalty_strength, penalty_decay=penalty_decay,
                                  inv_temp=inv_temp, ema_win_size=ema_win_size, learning_rate=learning_rate)
    tree.build_model()

    epochs = 40

    es = EarlyStopping(monitor='val_acc', patience=20, verbose=1)

    # os.remove('assets/distilled/checkpoint')
    # for f in glob.glob('assets/distilled/tree-model*'):
    #     os.remove(f)

    tree.maybe_train(
        sess=sess, data_train=(x_train_flat, y_train_soft), data_valid=(x_valid_flat, y_valid),
        batch_size=batch_size, epochs=epochs, callbacks=[es], distill=True)

    tree.evaluate(x=x_valid_flat, y=y_valid, batch_size=batch_size)
    tree.evaluate(x=x_test_flat, y=y_test, batch_size=batch_size)

    # draw_tree(sess, tree, img_rows, img_cols, img_chans)



