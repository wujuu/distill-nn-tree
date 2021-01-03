import numpy as np
import os
import tensorflow as tf

from models import ConvNet, SoftBinaryDecisionTree
from models.utils import brand_new_tfsession, draw_tree
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import plot_model
import glob


if __name__ == '__main__':
    sess = brand_new_tfsession()

    # load MNIST data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # add channel dim
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

    # hold out last 10000 training samples for validation
    x_valid, y_valid = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    # print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)

    # retrieve image and label shapes from training data
    img_rows, img_cols, img_chans = x_train.shape[1:]
    n_classes = np.unique(y_train).shape[0]

    # print(img_rows, img_cols, img_chans, n_classes)

    # convert labels to 1-hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    # print(y_train.shape, y_valid.shape, y_test.shape)

    # normalize inputs and cast to float
    x_train = (x_train / np.max(x_train)).astype(np.float32)
    x_valid = (x_valid / np.max(x_valid)).astype(np.float32)
    x_test = (x_test / np.max(x_test)).astype(np.float32)

    model_names = [
        'model_1_x_64_5_x_512_14_x_256_dense',
        # 'model_1_x_64_7_x_512_dense',
        # 'model_1_x_64_30_x_256_dense',
        # 'model_1_x_128_2_x_1024_dense',
        # 'model_1_x_128_5_x_512_dense',
        # 'model_1_x_128_15_x_256_dense',
        # 'model_1_x_256_dense'
    ]

    results = []

    for model_name in model_names:
        print(f"MODEL: {model_name}")
        nn = ConvNet(img_rows, img_cols, img_chans, n_classes)
        nn.build_model()
        # print(nn.model.summary())
        nn.maybe_train(data_train=(x_train, y_train),
                       data_valid=(x_valid, y_valid),
                       batch_size=32, epochs=12, model_name=model_name, callbacks=[])

        # # plot_model(nn.model, show_shapes=True)
        #
        # print(f"NN SCORE: {nn.evaluate(x_test, y_test)}")
        nn_acc = nn.evaluate(x_test, y_test)
        y_train_soft = nn.predict(x_train)
        #
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_valid_flat = x_valid.reshape((x_valid.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))
        #
        n_features = img_rows * img_cols * img_chans
        tree_depth = 4
        penalty_strength = 1e+1
        penalty_decay = 0.25
        ema_win_size = 1000
        inv_temp = 0.01
        learning_rate = 5e-03
        batch_size = 4
        #
        sess = brand_new_tfsession(sess)

        tree = SoftBinaryDecisionTree(tree_depth, n_features, n_classes,
                                      penalty_strength=penalty_strength, penalty_decay=penalty_decay,
                                      inv_temp=inv_temp, ema_win_size=ema_win_size, learning_rate=learning_rate, model_name=model_name)
        tree.build_model()

        epochs = 40

        es = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

        tree.maybe_train(
            sess=sess, data_train=(x_train_flat, y_train_soft), data_valid=(x_valid_flat, y_valid),
            batch_size=batch_size, epochs=epochs, callbacks=[es], distill=True)
        #

        tree_acc = tree.evaluate(x=x_test_flat, y=y_test, batch_size=batch_size)
        print(f"TREE SCORE: {tree_acc}")

        results.append((model_name, nn_acc, tree_acc))

        # draw_tree(sess, tree, img_rows, img_cols, img_chans)

    print(results)



