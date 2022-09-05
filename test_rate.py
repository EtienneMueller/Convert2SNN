import tensorflow as tf
import numpy as np
import convert2snn
#from convert2snn import utils


tf.random.set_seed(1234)
batch_size = 512
epochs = 5
act = 'relu'


def main():
    # Import Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    # Unroll (60000, 28, 28) to (60000, 784)
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    # Analog model
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(500, activation=act)(inputs)
    # x = tf.keras.layers.ReLU()(x)  # max_value=1
    x = tf.keras.layers.Dense(100, activation=act)(x)
    # x = tf.keras.layers.Activation(tf.nn.relu)(x)  # not implemented yet
    x = tf.keras.layers.Dense(10, activation=act)(x)
    x = tf.keras.layers.Softmax()(x)
    ann = tf.keras.Model(inputs=inputs, outputs=x)

    ann.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    ann.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs)

    _, testacc = ann.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    # weights = ann.get_weights()
    weights = convert2snn.get_normalized_weights(ann, x_train, percentile=85)

    # Preprocessing for RNN
    x_train = np.expand_dims(x_train, axis=1)  # (60000, 784)->(60000, 1, 784)
    x_test = np.expand_dims(x_test, axis=1)
    # x_rnn = np.tile(x_train, (1, 1, 1))
    # y_rnn = y_train  # np.tile(x_test, (1, timesteps, 1))

    # Conversion to spiking model
    snn = convert2snn.convert2rate(ann, weights, x_test, y_test, err="SparseCategoricalCrossentropy")
    # evaluate_conversion(snn, ann, x_test, y_test, testacc, timesteps=100)
    convert2snn.evaluate_conversion(snn, ann, x_test, y_test, testacc)


if __name__ == "__main__":
    main()
